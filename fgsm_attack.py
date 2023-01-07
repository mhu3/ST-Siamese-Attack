import numpy as np
import tensorflow as tf
from copy import copy
from tqdm import tqdm

class FGSM():
    """
    Running Fast Gradient Sign Method 
    adversarial attacks on a model for Sequential data.
    """
    def __init__(self, model, abort_early=True):
        self.model = model
        self.abort_early = abort_early

    def linf_attack(self, samples, labels, eps=0.001, iteration=20):
        # normalization factor  49, 92
        # Binary cross entropy (0/1)
        loss_object = tf.keras.losses.BinaryCrossentropy()

        # Change label size to [batch_size, 1]
        # labels = labels[:, np.newaxis]

        pairs_ori = tf.Variable(samples, dtype=tf.float32)
        pairs_adv = tf.Variable(samples, dtype=tf.float32)

        shape = pairs_ori.shape

        # Define the valid mask
        valid = np.ones(shape, dtype=np.int32)
        # set padding to non changable
        samples, trajs, rows = np.where(np.sum(pairs_ori, axis=3) == 0)
        valid[samples, trajs, rows, :] = 0
        # set time to non changable
        valid[:, :, :, 2] = 0

        # Iterative FGSM
        for i in tqdm(range(iteration)):
            # Compute current loss
            with tf.GradientTape() as tape:
                
                # Watch the input
                tape.watch(pairs_adv)
                
                # Compute loss
                prediction = self.model(pairs_adv)
                loss = loss_object(labels, prediction)

            # Get the gradients of the loss w.r.t to the input
            gradients = tape.gradient(loss, pairs_adv)
            # Get the sign of the gradients to create the perturbation
            signed_grads = tf.sign(gradients)

            pairs_adv = pairs_adv + eps * valid * signed_grads
            
            # compute the difference between the original and the perturbed trajectory
            diff = pairs_adv.numpy() - pairs_ori.numpy()

            # Clip the diff to be in the range of [-1/49, 1/49], [-1/92, 1/92] on x, y 
            diff[:, :, :, 0] = np.clip(diff[:, :, :, 0], -1/49, 1/49)
            diff[:, :, :, 1] = np.clip(diff[:, :, :, 1], -1/92, 1/92)
            diff = tf.convert_to_tensor(diff)
            
            # Add the difference to the original input
            pairs_adv = pairs_ori + diff

        return pairs_adv.numpy()

    def l2_attack(self, samples, labels, eps=0.05, iteration=100, p_norm=2):
        # normalization factor  49, 92
        # Binary cross entropy (0/1)
        loss_object = tf.keras.losses.BinaryCrossentropy()

        pairs_ori = tf.Variable(samples, dtype=tf.float32)
        pairs_adv = tf.Variable(samples, dtype=tf.float32)
        
        # Iterative FGSM
        for i in tqdm(range(iteration)):
            # Compute current loss
            with tf.GradientTape() as tape:

                # Watch the input
                tape.watch(pairs_adv)

                # Compute loss
                prediction = self.model(pairs_adv)
                loss = loss_object(labels, prediction)

            # Get the gradients of the loss w.r.t to the input
            gradients = tape.gradient(loss, pairs_adv)

            # get the l2-norm of the gradients
            norm = tf.norm(tf.reshape(gradients, [gradients.shape[0], -1]), ord=p_norm, axis=1)
            norm = tf.reshape(norm, [gradients.shape[0], 1, 1, 1])

            # l2 norm of the gradients
            l2_grads = gradients / norm

            # Constraint on sequential data
            # we should not change the padding
            l2_grads = l2_grads.numpy()
            for i in range(l2_grads.shape[0]):
                # find rows with all 0s
                trajs, rows = np.where( np.sum(samples[i], axis=2) == 0 )
                l2_grads[i, trajs, rows, :] = 0

                # avoid changing time
                l2_grads[i, :, :, 2] = 0

                # acoid changing the serve trajectory
                l2_grads[i, 10:20, :, :] = 0

            # Perturb the input
            l2_grads = tf.convert_to_tensor(l2_grads)
            pairs_adv = pairs_adv + eps * l2_grads
            
            # compute the difference between the original and the perturbed trajectory
            diff = pairs_adv.numpy() - pairs_ori.numpy()

            # Clip the diff to be in the range of [-1/49, 1/49], [-1/92, 1/92] on x, y 
            diff[:, :, :, 0] = np.clip(diff[:, :, :, 0], -1/49, 1/49)
            diff[:, :, :, 1] = np.clip(diff[:, :, :, 1], -1/92, 1/92)
            diff = tf.convert_to_tensor(diff)
            
            # Add the difference to the original input
            pairs_adv = pairs_ori + diff

        return pairs_adv.numpy()

    def l1_attack(self, samples, labels, eps=0.05, iteration=500):
        return self.l2_attack(samples, labels, eps, iteration, p_norm=1)
    
    def l0_attack(self, samples, labels):
        """
        Perform the L) attack on the given samples.
        """
        # Attack each individual sample
        adv_samples = np.zeros_like(samples)
        '''
        for i in range(0, samples.shape[0], self.batch_size):
            adv_samples[i : i + self.batch_size] = \
                self.l0_attack_batch(samples[i : i + self.batch_size], 
                                     labels[i : i + self.batch_size])
        '''
        for i in tqdm(range(len(samples))):
            adv_sample = self.l0_attack_single(samples[i:i+1], labels[i])
            adv_samples[i:i+1] = adv_sample
        
        return adv_samples
    
    def l0_attack_single(self, x, y, iteration=100, eps=0.05):
        """
        Perform the L0 attack on the given single sample.
        """
        loss_object = tf.keras.losses.BinaryCrossentropy()

        # Cast original samples to tensor
        original_x = tf.Variable(x, dtype=tf.float32)
        adv_x = tf.Variable(x, dtype=tf.float32)
        shape = original_x.shape

        # Define the valid mask
        valid = np.ones(shape, dtype=np.int32)
        # set padding to non changable
        samples, trajs, rows = np.where(np.sum(x, axis=3) == 0 )
        valid[samples, trajs, rows, :] = 0
        # set time to non changable
        valid[:, :, :, 2] = 0

        # Iterative FGSM
        for i in tqdm(range(iteration)):
            # Compute current loss
            with tf.GradientTape() as tape:
                
                # Watch the input
                tape.watch(adv_x)
                
                # Compute loss
                prediction = self.model(adv_x)
                loss = loss_object(y, prediction)

            # Get the gradients of the loss w.r.t to the input
            gradients = tape.gradient(loss, adv_x)
            # Get the sign of the gradients to create the perturbation
            signed_grads = tf.sign(gradients)

            # Perturb the input
            adv_x = adv_x + eps * signed_grads
            
            # compute the difference between the original and the perturbed trajectory
            diff = adv_x.numpy() - original_x.numpy()

            # Clip the diff to be in the range of [-1/49, 1/49], [-1/92, 1/92] on x, y 
            diff[:, :, :, 0] = np.clip(diff[:, :, :, 0], -1/49, 1/49)
            diff[:, :, :, 1] = np.clip(diff[:, :, :, 1], -1/92, 1/92)
            diff = tf.convert_to_tensor(diff)
            
            # Add the difference to the original input
            adv_x = original_x + valid * diff

        #     y_hat = self.model(pairs_adv)

        #     y_hat = y * y_hat + (1 - y) * (1 - y_hat)

        #     if yhat < 0.5:


        #     # Adjust the valid mask
        #     prev_valid = valid.copy()

        #     # Compute total change
        #     total_change = np.sum(np.abs(gradients), axis=3) 
        #     # Set some of the pixels to 0 depending on their total change
        #     # 1, if the change is insignificant
        #     valid[total_change <= 1e-5] = 0
        #     # 2, set 20% of the current valid waypoints to 0
        #     unchangable_count = np.sum(total_change <= 1e-5)
        #     changing_count = int( 0.2 * (total_change.size - unchangable_count) )
        #     sorted_indices = np.unravel_index(np.argsort(total_change, axis=None), total_change.shape)
        #     sorted_indices = [indices[unchangable_count:unchangable_count + changing_count] 
        #                         for indices in sorted_indices]
        #     valid[tuple(sorted_indices)] = 0
        #     # if no more change happens to the valid mask
        #     if (valid - prev_valid).sum() == 0:
        #         break
        # # Attack failed eventually
        # # Return the last adv sample
        return adv_x.numpy()

