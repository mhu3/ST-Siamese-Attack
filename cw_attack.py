import tensorflow as tf
import numpy as np
from tqdm import tqdm
from cw_attack_utils import to_tanh_space, from_tanh_space, set_with_mask


class CWAttack:
    def __init__(self, model, targeted=False, 
                 batch_size=256, learning_rate=1e-2,
                 max_iterations=1000, abort_early=True,
                 initial_const=1e-3, largest_const=1e6,
                 binary_search_steps=5, const_factor=2.0
                ):
        """ The CW attack class. 
        Args:
            model: Instance of the Model class
            targeted: True if we should perform a targetted attack, False otherwise.
            batch_size: Number of attacks to run simultaneously.
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
        self.learning_rate = learning_rate

        self.max_iterations = max_iterations
        self.abort_early = abort_early

        self.initial_const = initial_const
        self.largest_const = largest_const
        self.binary_search_steps = binary_search_steps
        self.const_factor = const_factor

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)


    def gradient(self, x_original, valid_mask, 
                 x_start_tanh, modifier, 
                 y, const):
        ''' Compute the gradient of the loss function
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
            
            # Compute loss
            y_hat = self.model(x_new)
            loss, target_loss, l2_dist = \
                self.l2_loss(x_original, x_new, y, y_hat, const, kappa=0.01)

        # Compute gradients
        grads = tape.gradient(loss, x_new_tanh)
        return x_new, y_hat, grads, loss, target_loss, l2_dist

    def l2_loss(self, x, x_new, y, y_hat, const, kappa):
        # Define loss
        # 1, Target loss
        # TODO TEMP FIX
        y_hat = y * y_hat + (1 - y) * (1 - y_hat)
        real = tf.reduce_sum(y_hat, axis=1) # F(t)
        other = tf.reduce_sum(1.0 - y_hat, axis=1) # F(i) (i != t)
        # real = tf.reduce_sum((y) * y_hat, 1) # Z(t)
        # other = tf.reduce_max((1-y) * y_hat, 1) # max Z(i) (i != t)

        # if targetted, optimize for making the other class most likely
        if self.targeted:
            target_loss = tf.maximum(0, other - real + kappa) # + 0.01)
        # if untargeted, optimize for making this class least likely.
        else:
            target_loss = tf.maximum(0, real - other + kappa) # + 0.01)

        # 2, Perturbation loss (L2)
        l2_dist = tf.reduce_sum( tf.square(x_new - x), 
                                 list(range(1, len(x.shape))) )

        # sum up the losses
        loss = const*target_loss + l2_dist

        return loss, target_loss, l2_dist

    
    def l2_attack(self, samples, labels):
        """
        Perform the L2 attack on the given samples.
        """
        # Attack batch sample
        adv_samples = np.zeros_like(samples)
        for i in range(0, samples.shape[0], self.batch_size):
            adv_samples[i : i + self.batch_size] = \
                self.l2_attack_batch(samples[i : i + self.batch_size], 
                                     labels[i : i + self.batch_size])

        return adv_samples

    
    def l0_attack(self, samples, labels):
        """
        Perform the L) attack on the given samples.
        """
        # Attack each individual sample
        adv_samples = np.zeros_like(samples)
        for i in tqdm(range(len(samples))):
            adv_sample = self.l0_attack_single(samples[i:i+1], labels[i])
            adv_samples[i:i+1] = adv_sample

        return adv_samples

    
    def l2_attack_batch(self, x, y):
        """
        Perform the L2 attack on the given batch of samples.
        """
        # Cast original samples to tensor
        original_x = tf.cast(x, tf.float32)
        shape = original_x.shape
        # Convert to tanh-space
        x_tanh = to_tanh_space(original_x)

        # Define the valid mask
        valid = np.ones(shape, dtype=np.int32)
        # set padding to non changable
        samples, trajs, rows = np.where( np.sum(x, axis=3) == 0 )
        valid[samples, trajs, rows, :] = 0
        # set time to non changable
        valid[:, :, :, 2] = 0

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


    def l0_attack_single(self, x, y):
        """
        Perform the L0 attack on the given single sample.
        """
        # Cast original samples to tensor
        original_x = tf.cast(x, tf.float32)
        shape = original_x.shape
        # Convert to tanh-space
        x_tanh = to_tanh_space(original_x)

        # Define the valid mask
        valid = np.ones(shape, dtype=np.int32)
        # set padding to non changable
        samples, trajs, rows = np.where( np.sum(x, axis=3) == 0 )
        valid[samples, trajs, rows, :] = 0
        # set time to non changable
        valid[:, :, :, 2] = 0

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
        while const < upper_bound:

            # early stopping criteria
            prev_loss = np.inf
            # Given current const and valid map, optimize the perturbation
            for iteration in tqdm(range(self.max_iterations)):

                x_new, preds, grads, loss, target_loss, l2_dist = \
                    self.gradient(original_x, valid, x_tanh, modifier, y, const)
                
                # check if we misclassify the sample
                if (self.abort_early and target_loss <= 0):
                    # abort if true
                    break
                # if not update modifier and keep optimizing
                else:
                    self.optimizer.apply_gradients( [(grads, modifier)] )
            
            # The attack failed, adjust the const by increasing const
            if (target_loss > 0):
                const = const * self.const_factor
                continue
            
            # The attack succeeded, keep current result
            best_attack = tf.identity(x_new)

            # Adjust the valid mask
            prev_valid = valid.copy()

            # Compute total change
            total_change = np.sum( np.abs(x_new - original_x), axis=3 ) * np.sum( np.abs(grads), axis=3 ) 
            # Set some of the pixels to 0 depending on their total change
            # 1, if the change is insignificant
            valid[total_change <= 1e-5] = 0
            # 2, set 20% of the current valid waypoints to 0
            unchangable_count = np.sum(total_change <= 1e-5)
            changing_count = int( 0.2 * (total_change.size - unchangable_count) )
            sorted_indices = np.unravel_index(np.argsort(total_change, axis=None), total_change.shape)
            sorted_indices = [indices[unchangable_count:unchangable_count + changing_count] 
                              for indices in sorted_indices]
            valid[tuple(sorted_indices)] = 0

            # if no more change happens to the valid mask
            if (valid - prev_valid).sum() == 0:
                break

        # Attack failed eventually
        # Return the last adv sample
        return best_attack.numpy()
