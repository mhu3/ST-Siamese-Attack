import tensorflow as tf
import numpy as np
from tqdm import tqdm


class CWAttack:
    def __init__(self, sess, model, sample_shape,
                 targeted = False, learning_rate = 1e-2,
                 max_iterations = 1000, abort_early = True,
                 initial_const = 1e-3, largest_const = 2e6,
                 reduce_const = False, const_factor = 2.0,
                 independent_channels = False):
        """ The L_0 optimized attack. 
        Args:
            sess: TF session
            model: Instance of the Model class
            targeted: True if we should perform a targetted attack, False otherwise.
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
            reduce_const: Try to lower c each iteration; faster to set to false.
            const_factor: The factor f at which we should increase the constant, when the
                          previous constant failed. Should be greater than one, smaller 
                          is better.
            independent_channels: Set to false optimizes for number of pixels changed,
                                  set to true (not recommended) returns number of channels
                                  changed.
        Returns:
            adversarial examples for the supplied model.
        """
        self.model = model
        self.sess = sess

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.REDUCE_CONST = reduce_const
        self.CONST_FACTOR = const_factor
        self.INDEPEDENT_CHANNELS = independent_channels

        self.grad = self.gradient_descent(sess, model, sample_shape)


    def gradient_descent(self, sess, model, shape):
        # Variables and placeholders
        # the variable w to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))
        # the variables we're going to hold, use for efficiency
        original = tf.Variable(np.zeros(shape, dtype=np.float32))
        changable = tf.Variable(np.zeros(shape, dtype=np.float32))
        start = tf.Variable(np.zeros(shape, dtype=np.float32))
        sample = tf.Variable(np.zeros(shape, dtype=np.float32))
        label = tf.Variable(np.zeros((1, self.model.output.shape[1]), dtype=np.float32))

        # placeholder to set the variables
        assign_modifier = tf.placeholder(np.float32, shape)
        assign_original = tf.placeholder(np.float32, shape)
        assign_changable = tf.placeholder(np.float32, shape)
        assign_start = tf.placeholder(np.float32, shape)
        assign_sample = tf.placeholder(np.float32, shape)
        assign_label = tf.placeholder(np.float32, (1, self.model.output.shape[1]))
        const = tf.placeholder(tf.float32, [])

        # setup to assign the variables with placeholders
        setup_modifier = tf.assign(modifier, assign_modifier)
        setup = []
        setup.append(tf.assign(changable, assign_changable))
        setup.append(tf.assign(sample, assign_sample))
        setup.append(tf.assign(original, assign_original))
        setup.append(tf.assign(start, assign_start))
        setup.append(tf.assign(label, assign_label))
        
        # Predictions
        new_sample = changable * ((tf.tanh(modifier + start) + 1)/2) + \
                     (1-changable) * original
        output = model.predict([new_sample])
        
        # Define loss
        # TODO TEMP FIX
        # 1, target loss
        # real = tf.reduce_sum((label) * output, 1) # Z(t)
        # other = tf.reduce_max((1-label) * output, 1) # max Z(i) (i != t)
        real = (output)
        other = (1 - output)

        # if targetted, optimize for making the other class most likely
        kappa = 0.01
        confidence = 0.01
        if self.TARGETED:
            loss1 = tf.maximum(0, other - real + kappa) # + 0.01)
        # if untargeted, optimize for making this class least likely.
        else:
            loss1 = tf.maximum(0, real - other + kappa) # + 0.01)

        # 2, loss
        loss2 = tf.reduce_sum( tf.square(new_sample - tf.tanh(sample)/2) )

        # sum up the losses
        loss = const*loss1 + loss2
        
        # Gradient
        grad = tf.gradients(loss, [modifier])[0]
        
        # setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train = optimizer.minimize(loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        init = tf.variables_initializer(var_list=[modifier, changable, start, original, sample, label] 
                                                 + new_vars)

        # Actual optimization process
        def optimize(samples, labels, starts, valid, const):
            # convert to tanh-space
            samples = np.arctanh( np.array(samples) * 2 - 1 ) # *1.999999 )
            starts = np.arctanh( np.array(starts) * 2 - 1 ) # *1.999999 )

            # initialize the variables
            sess.run(init)
            sess.run(setup, {assign_sample: samples, 
                             assign_label: labels, 
                             assign_start: starts, 
                             assign_original: samples,
                             assign_changable: valid})

            # try solving for each value of the constant
            while const < self.LARGEST_CONST:
                print('try const', const)

                for step in range(self.MAX_ITERATIONS):
                    feed_dict = {const: const}
                    # remember the old value
                    old_modifier = self.sess.run(modifier)

                    # logging
                    if step%(self.MAX_ITERATIONS // 10) == 0:
                        print(step, *sess.run((loss1, loss2), feed_dict=feed_dict))

                    # perform the update step
                    _, success, scores = sess.run([train, loss1, output], feed_dict=feed_dict)

                    '''
                    # presoftmax score check
                    if np.all(scores >= -0.0001) and np.all(scores <= 1.0001):
                        if np.allclose(np.sum(scores,axis=1), 1.0, atol=1e-3):
                            if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                                raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")
                    '''
                    # if this step is successful
                    if success < 0 and self.ABORT_EARLY:
                        # it worked previously, restore the old value and finish
                        self.sess.run(setup_modifier, {assign_modifier: old_modifier})
                        grads, scores, new_samples = sess.run((grad, output, new_sample), feed_dict=feed_dict)

                        # l2s = np.square(new_samples - np.tanh(samples)/2).sum(axis=(1, 2, 3))
                        return grads, scores, new_samples, const

                # we didn't succeed within iteration, increase constant and try again
                const *= self.CONST_FACTOR
            # cannot find a solution within the largest constant
            return None

        return optimize


    def l0_attack(self, samples, targets):
        """
        Perform the L_0 attack on the given samples for the given targets.
            If self.targeted is true, then the targets represents the target labels.
            If self.targeted is false, then targets are the original class labels.
        """
        # Attack each individual sample
        adv_samples = []
        for i in tqdm(range(len(samples))):
            adv_sample = self.l0_attack_single(samples[i], targets[i])
            adv_samples.append(adv_sample)

        return np.array(adv_samples)


    def l0_attack_single(self, sample, target):
        """
        Run the attack on a single sample and label
        """
        # Define the valid waypoints we can change
        valid = np.ones(sample.shape)
        # find unchangable padding features and time features
        trajs, padding_rows = np.where(np.sum(sample, axis=2) == 0)
        valid[trajs, padding_rows, :] = 0
        valid[:, :, 2] = 0
        # reshape to flattened feature vector or falttened waypoint matrix
        if self.INDEPEDENT_CHANNELS:
            valid = valid.flatten()
        else:
            valid = valid.reshape(-1, sample.shape[-1])

        # Initially set the solution to None.
        # return None if we can't find an adversarial sample.
        last_adv_sample = None
        # set original sample as previous sample to begin with
        prev = np.copy(sample)
        
        const = self.INITIAL_CONST
        change_count = None
        while True:
            # Try to solve given this valid map
            res = self.grad([np.copy(sample)], [target], np.copy(prev), 
                             valid, const)
            
            # The attack failed, can not set more pixels to be 0
            if res == None:
                return last_adv_sample

            # The attack succeeded, pick new pixels to set to 0
            grads, scores, new_sample, const = res
            new_sample = new_sample[0] # single
            grads = grads[0] # single
            
            # Compute total change to decide which waypoints to set to not changable
            # we are allowed to change each channel independently
            if self.INDEPEDENT_CHANNELS:
                change_count = np.sum( np.abs(new_sample - sample) > 1e-5 )
                total_change = np.abs(new_sample - sample) * np.abs(grads)
            # we care only about the waypoint change, not each feature independently
            else:
                # compute total change as sum of change for each feature
                change_count = np.sum( np.sum(np.abs(new_sample - sample), axis=2) > 1e-5 )
                total_change = np.sum( np.abs(new_sample - sample), axis=2 ) * np.sum( np.abs(grads), axis=2 )
            total_change = total_change.flatten()

            # Set some of the pixels to 0 depending on their total change
            set_count = 0           
            for i in np.argsort(total_change):
                # already not changable, skip
                if not np.all(valid[i]):
                    continue
                # set to not changable
                valid[i] = 0
                set_count += 1

                # if this waypoint changed a lot, break
                if total_change[i] > .01:
                    break
                # if we changed too many waypoints at a time, break
                if set_count > 0.2 * change_count:
                    break

            print("Now forced equal: ", np.sum(1 - valid))
            last_adv_sample = new_sample
            prev = new_sample
            if self.REDUCE_CONST: 
                const /= 2
