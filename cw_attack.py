import numpy as np
from tqdm import tqdm
import tensorflow as tf

from cw_attack_utils import to_tanh_space, from_tanh_space, set_with_mask


class CWAttack:
    def __init__(
        self, model, targeted=False, batch_size=256, 
        eta = 1, learning_rate=0.001, max_iterations=1000, 
        abort_early=True, 
        initial_const=0.001, largest_const=1e9, 
        binary_search_steps=5, const_factor=2.0
    ):
        """ The CW attack class. 
        Args:
            model: Instance of the Model class
            targeted: True if we should perform a targetted attack, False otherwise.
            batch_size: Number of attacks to run simultaneously.
            eta: linf constraint
            learning_rate: The learning rate for the attack algorithm. Smaller values
                           produce better results but are slower to converge.
            max_iterations: The maximum number of iterations to perform gradient descent. 
                            Larger values are more accurate; setting too small will require 
                            a large learning rate and will produce poor results.
            abort_early: If true, allows early aborts if gradient descent gets stuck.
            initial_const: The initial tradeoff-constant c to use to tune the relative
                           importance of distance and confidence. Should be set to a 
                           very small value (but positive).
            largest_const: The largest constant c to use until we report failure. Should
                           be set to a very large value.
            binary_search_steps: The number of times we perform binary search to find the
                                 optimal c. (For l2 attack)
            const_factor: The factor f at which we should increase the constant, when the
                          previous constant failed. Should be greater than one, smaller 
                          is better. (For l0 attack)
        """
        self.model = model
        self.targeted = targeted
        self.batch_size = batch_size

        self.eta = eta
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
        self.abort_early = abort_early

        self.initial_const = initial_const
        self.largest_const = largest_const
        self.binary_search_steps = binary_search_steps
        self.const_factor = const_factor

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)


    def gradient(
        self, x_original, valid_mask, 
        x_start_tanh, modifier, 
        y, const
    ):
        ''' Compute the gradient of the given loss function
        Args:
            x_original: original sample
            valid_mask: mask of valid features which could be changed
            x_start_tanh: For l0 attack, x_start_tanh is the latest x_tanh
                          For l2 attack, x_start_tanh is always x_original_tanh
            modifier: the modifier (delta) of the input image in tanh space
                      i.e. w = x + modifier
            y: label
            const: the constant c
        '''
        with tf.GradientTape() as tape:
            # Convert current adv sample to regular space
            x_new_tanh = x_start_tanh + modifier
            x_new = from_tanh_space(x_new_tanh)

            # Combine it with original sample given valid mask
            x_new = set_with_mask(x_original, x_new, valid_mask)   

            # Clip to make sure the difference is within 
            # [-eta/49, eta/49] for x and [-eta/92, eta/92] for y
            diff = x_new - x_original
            diff_x, diff_y, diff_t = tf.split(diff, 3, axis=3)
            diff_x = tf.clip_by_value(diff_x, -self.eta/49, self.eta/49)
            diff_y = tf.clip_by_value(diff_y, -self.eta/92, self.eta/92)
            diff = tf.concat([diff_x, diff_y, diff_t], axis=3)
            x_new = x_original + diff

            # Compute loss
            y_hat = self.model(x_new)
            loss, target_loss, l2_dist = self.l2_loss(
                x_original, x_new, y, y_hat, const, kappa=0.02
            )

        # Compute gradients
        grads = tape.gradient(loss, x_new_tanh)
        return x_new, y_hat, grads, loss, target_loss, l2_dist


    def l2_loss(self, x, x_new, y, y_hat, const, kappa):
        # Define loss
        # 1, Target loss
        # if binary
        if y_hat.shape[1] == 1:
            y_hat = y * y_hat + (1 - y) * (1 - y_hat)
            real = tf.reduce_sum(y_hat, axis=1) # F(t)
            other = tf.reduce_sum(1.0 - y_hat, axis=1) # F(i) (i != t)
        # if multi-class
        else:
            real = tf.reduce_sum((y) * y_hat, axis=1) # F(t)
            other = tf.reduce_max((1-y) * y_hat, axis=1) # max F(i) (i != t)

        # if targetted, optimize for making the other class most likely
        if self.targeted:
            target_loss = tf.maximum(0, other - real + kappa) # + 0.01)
        # if untargeted, optimize for making this class least likely.
        else:
            target_loss = tf.maximum(0, real - other + kappa) # + 0.01)

        # 2, Perturbation loss (L2)
        l2_dist = tf.reduce_sum(
            tf.square(x_new - x), list(range(1, len(x.shape))) 
        )

        # sum up the losses and return
        loss = const*target_loss + l2_dist
        return loss, target_loss, l2_dist


    def attack(self, samples, labels, valid, norm='l2'):
        """
        Perform attack with given norm on the given samples.
        """
        # Select attack method
        if norm == 'l2':
            attack_method = self.l2_attack_batch
        elif norm == 'l0':
            attack_method = self.l0_attack_batch

        # Attack on batch
        adv_samples = np.zeros_like(samples)
        for i in range(0, samples.shape[0], self.batch_size):
            
            adv_samples[i : i + self.batch_size] = \
                attack_method(
                    samples[i : i + self.batch_size], 
                    labels[i : i + self.batch_size],
                    valid[i : i + self.batch_size]
                )
        return adv_samples


    def l2_attack_batch(self, x, y, valid):
        """
        Perform the L2 attack on the given batch of samples.
        """
        # Cast original samples to tensor
        original_x = tf.cast(x, tf.float32)
        shape = original_x.shape
        # Convert to tanh-space
        x_tanh = to_tanh_space(original_x)  

        # Placeholder variables
        # for binary search of the const
        const = tf.ones(shape[:1]) * self.initial_const
        lower_bound = tf.zeros(shape[:1])
        upper_bound = tf.ones(shape[:1]) * self.largest_const
        # for best values
        best_l2 = tf.fill(shape[:1], 1e10)
        best_label = tf.fill(shape[:1], -1)
        best_attack = tf.identity(original_x)
        # for pertubation
        modifier = tf.Variable(tf.zeros(shape, dtype=x.dtype), trainable=True)

        # Comparing function 
        compare_fn = tf.equal if self.targeted else tf.not_equal

        # At each outer iteration, use different consts 
        # to find the best perturbation
        for outer_step in tqdm(range(self.binary_search_steps)):
            # Reset optimization variable state
            modifier.assign(tf.zeros(shape, dtype=x.dtype))
            for var in self.optimizer.variables():
                var.assign(tf.zeros(var.shape, dtype=var.dtype))

            # early stopping criteria
            prev_loss = np.inf
            # Given current const, find the best perturbation
            for iteration in tqdm(range(self.max_iterations)):

                x_new, preds, grads, loss, target_loss, l2_dist = \
                    self.gradient(original_x, valid, x_tanh, modifier, y, const)
                self.optimizer.apply_gradients( [(grads, modifier)] )

                # check if we made progress, abort otherwise
                if (self.abort_early and \
                    iteration % ((self.max_iterations // 10) or 1) == 0
                ):
                    # TODO TEMP FIX
                    if np.all(loss >= prev_loss):
                        break
                    prev_loss = loss

                # update best results
                # TODO TEMP FIX
                label = tf.cast(y, tf.int32)
                label = tf.reduce_sum(label, 1)
                # label = tf.argmax(y, axis=1)

                # Adjust the best result found so far
                # TODO TEMP FIX
                preds = tf.where(preds > 0.5, 1, 0)
                preds = tf.reduce_sum(preds, 1)
                # preds = tf.argmax(preds, axis=1)

                # if the l2 distance is better than the one found before
                # and if the example is a correct example (with regards to the labels)
                mask = tf.math.logical_and(
                    tf.less(l2_dist, best_l2),
                    compare_fn(preds, label)
                )
                best_l2 = set_with_mask(best_l2, l2_dist, mask)
                best_label = set_with_mask(best_label, preds, mask)
                
                # mask is of shape [batch_size]; best_attack is [batch_size, image_size]
                mask = tf.reshape(mask, [-1, 1, 1, 1])
                mask = tf.tile(mask, [1, *best_attack.shape[1:]])
                best_attack = set_with_mask(best_attack, x_new, mask)

            # Search for perturbation with the current const is done
            # adjust constant with binary search

            # successful attack, adjust upper bound
            upper_mask = tf.math.logical_and(
                compare_fn(best_label, label),
                tf.not_equal(best_label, -1),
            )
            upper_bound = set_with_mask(
                upper_bound, tf.math.minimum(upper_bound, const), upper_mask
            )
            # based on this mask compute const mask
            const_mask = tf.math.logical_and(
                upper_mask,
                tf.less(upper_bound, 1e9),
            )
            const = set_with_mask(const, (lower_bound + upper_bound) / 2.0, const_mask)

            # failure attack, adjust lower bound
            lower_mask = tf.math.logical_not(upper_mask)
            lower_bound = set_with_mask(
                lower_bound, tf.math.maximum(lower_bound, const), lower_mask
            )
            # based on this mask compute const mask
            const_mask = tf.math.logical_and(
                lower_mask,
                tf.less(upper_bound, 1e9),
            )
            const = set_with_mask(const, (lower_bound + upper_bound) / 2.0, const_mask)

            # if already at the maximum upper bound (1e9), multiply by 10
            const_mask = tf.math.logical_not(const_mask)
            const = set_with_mask(const, const * 10, const_mask)

        return best_attack.numpy()


    def l0_attack_batch(self, x, y, valid):
        """
        Perform the L0 attack on the given single sample.
        """
        # Cast original samples to tensor
        original_x = tf.cast(x, tf.float32)
        shape = original_x.shape
        
        # Convert to tanh-space
        x_tanh = to_tanh_space(original_x)

        # Placeholder variables
        # for binary search of the const
        const = tf.ones(shape[:1]) * self.initial_const
        
        upper_bound = tf.ones(shape[:1]) * self.largest_const
        # for best values
        best_attack = tf.identity(original_x)
        # for pertubation
        modifier = tf.Variable(tf.zeros(shape, dtype=x.dtype), trainable=True)

        # In each while loop
        # Find if current const and valid map provide a solution
        # |--> if yes, remove some waypoints from the valid mask
        # |--> if no, increase the constant
        # Until the constant is too large
        while tf.reduce_any(const < upper_bound):
            print("Current average const: ", float(tf.reduce_mean(const)))

            # Given current const and valid map, optimize the perturbation
            for iteration in tqdm(range(self.max_iterations)):

                x_new, preds, grads, loss, target_loss, l2_dist = \
                    self.gradient(original_x, valid, x_tanh, modifier, y, const)
                
                # check samples that are still correctly classified
                unsuccess_mask = tf.cast(target_loss > 0, modifier.dtype)
                success_mask = 1.0 - unsuccess_mask
                # mask is of shape [batch_size]; best_attack is [batch_size, image_size]
                success_mask = tf.reshape(success_mask, [-1, 1, 1, 1])
                # print(success_mask.shape)
                success_mask = tf.tile(success_mask, [1, *best_attack.shape[1:]])
                # print(best_attack.shape)
                # print(best_attack.shape[1:])
                # print(success_mask.shape)

                # check if we misclassify the samples
                if (self.abort_early and tf.reduce_sum(unsuccess_mask) == 0):
                    # abort if true
                    break
                # if not, update modifiers that are still correctly classified
                else:
                    old_modifier = tf.identity(modifier)
                    self.optimizer.apply_gradients( [(grads, modifier)] )
                    set_with_mask(modifier, old_modifier, success_mask, assign=True)
            
            # The attack failed, adjust the const by increasing const
            const = set_with_mask(const, const * self.const_factor, unsuccess_mask)
            print(const)
            const = tf.clip_by_value(const, 0, self.largest_const)
            print(const)
            
            # The attack succeeded, keep current result
            best_attack = set_with_mask(best_attack, x_new, success_mask)

            # Adjust the valid mask
            prev_valid = valid.copy()
            # Compute total change
            total_change = np.sum( np.abs(x_new - original_x), axis=3 ) * np.sum( np.abs(grads), axis=3 ) 
            # Set some of the pixels to 0 depending on their total change
            # only change the valid mask of those that are successfully attacked
            success_indices = np.where(unsuccess_mask.numpy() == 0)[0]
            # print the total number of the success_indices
            print("Total number of success indices: ", len(success_indices))
            unsuccess_indices = np.where(unsuccess_mask.numpy() == 1)[0]
            for i in success_indices:
                # 1, if the change is insignificant
                insignificant_indices = list( np.where( total_change[i] <= 1e-5 ) )
                insignificant_indices.insert(0, i)
                valid[tuple(insignificant_indices)] = 0
                # 2, set 20% of the current valid waypoints to 0
                unchangable_count = np.sum(total_change[i] <= 1e-5)
                changing_count = int( 0.2 * (total_change[i].size - unchangable_count))
                changing_count = max(changing_count, 1)
                sorted_indices = np.unravel_index(np.argsort(total_change[i], axis=None), total_change[i].shape)
                sorted_indices = [indices[unchangable_count:unchangable_count + changing_count] 
                                  for indices in sorted_indices]
                sorted_indices.insert(0, i)
                valid[tuple(sorted_indices)] = 0
            print(np.sum(valid))
            
            # Exit if 
            # all unsuccessfully attacked samples have reached maximum const and
            # all successufully attacked samples have no more valid waypoints to remove
            if tf.reduce_all(tf.gather(const, unsuccess_indices) >= self.largest_const ) and \
               np.abs(valid[success_indices] - prev_valid[success_indices]).sum() == 0:
                break

        # Attack failed eventually
        # Return the last adv sample
        return best_attack.numpy()
