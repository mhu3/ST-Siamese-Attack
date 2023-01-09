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
        # set serve to non changable
        valid[:, 10:20:, :, :] = 0

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
        # set serve to non changable
        valid[:, 10:20:, :, :] = 0
        
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

            pairs_adv = pairs_adv + eps * valid * l2_grads
            
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
            adv_sample = self.l0_attack_single(samples[i:i+1], labels[i:i+1])
            adv_samples[i:i+1] = adv_sample
            
        return adv_samples
    
    def l0_attack_single(self, x, y, eps=0.001, iteration=20):
        """
        Perform the L0 attack on the given single sample.
        """
        # normalization factor  49, 92
        # Binary cross entropy (0/1)
        loss_object = tf.keras.losses.BinaryCrossentropy()

        # Change label size to [batch_size, 1]
        # labels = labels[:, np.newaxis]

        pairs_ori = tf.Variable(x, dtype=tf.float32)
        pairs_adv = tf.Variable(x, dtype=tf.float32)

        shape = pairs_ori.shape

        # Define the valid mask
        valid = np.ones(shape, dtype=np.int32)
        # set padding to non changable
        samples, trajs, rows = np.where(np.sum(pairs_ori, axis=3) == 0)
        valid[samples, trajs, rows, :] = 0
        # set time to non changable
        valid[:, :, :, 2] = 0

        # # set serve to non changable
        # valid[:, 10:20:, :, :] = 0

        # iterative until the attack is not successful
        last_adv = pairs_adv.numpy()
        
        while True:
            diff = tf.zeros(shape, dtype=tf.float32)
            # Iterative FGSM
            for i in tqdm(range(iteration)):
                # Compute current loss
                with tf.GradientTape() as tape:
                    
                    # Watch the input
                    tape.watch(pairs_adv)
                    
                    # Compute loss
                    prediction = self.model(pairs_adv)
                    loss = loss_object(y, prediction)

                # Get the gradients of the loss w.r.t to the input
                gradients = tape.gradient(loss, pairs_adv)
                # Get the sign of the gradients to create the perturbation
                signed_grads = tf.sign(gradients)

                # Add perturbation to the adversarial example
                diff += valid * eps * signed_grads

                # Clip the diff to be in the range of [-1/49, 1/49], [-1/92, 1/92] on x, y 
                diff_np = diff.numpy()
                diff_np[:, :, :, 0] = np.clip(diff_np[:, :, :, 0], -2/49, 2/49)
                diff_np[:, :, :, 1] = np.clip(diff_np[:, :, :, 1], -2/92, 2/92)
                diff = tf.convert_to_tensor(diff_np) 

                # Add the difference to the original input
                pairs_adv = pairs_ori + valid * diff

            # Check if the attack is successful
            print(gradients)
            prediction = self.model(pairs_adv)
            print(prediction)
            

            # FGSM attack failed
            if np.all( np.rint(prediction) == y ):
                break
            # FGSM attack successed, keep record of the last successful attack
            # else:
            #     last_adv = pairs_adv.numpy()
            
            # Remove valid points
            # previous valid
            prev_valid = valid.copy()

            # Compute total change
            total_change = np.sum(np.abs(gradients * valid), axis=3)
            # 1, Consider valid
            # gradient * valid
            # 2, Use CW?
            # gradient * diff * valid

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
            
            last_adv = pairs_adv.numpy()

            # if no more change happens to the valid mask
            if np.abs(valid - prev_valid).sum() == 0: # 
                break

        # Attack failed eventually
        # Return the last adv sample
        return last_adv

    def l0_attack_batch(self, samples, labels, eps=0.001, iteration=20):
        """
        Perform the L0 attack on the given batch of samples.
        """
        # normalization factor  49, 92
        # Binary cross entropy (0/1)
        loss_object = tf.keras.losses.BinaryCrossentropy()

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

        # # set serve to non changable
        # valid[:, 10:20:, :, :] = 0

        # iterative until the attack is not successful for all samples
        last_adv = pairs_adv.numpy()

        while True:
            # Iterative FGSM
            diff = tf.zeros(shape, dtype=tf.float32)
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

                # Add perturbation to the adversarial example
                diff += valid * eps * signed_grads

                # Clip the diff to be in the range of [-1/49, 1/49], [-1/92, 1/92] on x, y
                diff_np = diff.numpy()
                diff_np[:, :, :, 0] = np.clip(diff_np[:, :, :, 0], -2/49, 2/49)
                diff_np[:, :, :, 1] = np.clip(diff_np[:, :, :, 1], -2/92, 2/92)
                diff = tf.convert_to_tensor(diff_np)

                # Add the difference to the original input
                pairs_adv = pairs_ori + valid * diff

            # Check if the attack is successful
            print(shape[0]*gradients[0, 0, 59, :])
            prediction = self.model(pairs_adv)
            print(prediction)

            # find the samples that are successfully attacked
            success = np.logical_not(np.rint(prediction) == labels)
            #find the indices of the samples that are successfully attacked
            success_indices = np.where(success)[0]

            # update the last adv sample
            last_adv[success_indices] = pairs_adv.numpy()[success_indices]
        
            # Remove valid points
            # previous valid
            prev_valid = valid.copy()

            # Compute total change
            total_change = np.sum(np.abs(gradients * valid), axis=3)
            # 1, Consider valid
            # gradient * valid
            # 2, Use CW?
            # gradient * diff * valid   

            # every successfully attacked sample, set 20% of the current valid waypoints to 0
            for i in success_indices:
                # 1, if the change is insignificant
                insignificant_indices = list( np.where( total_change[i] <= 1e-5 ) )
                insignificant_indices.insert(0, i)
                valid[tuple(insignificant_indices)] = 0
                # 2, set 20% of the current valid waypoints to 0
                unchangable_count = np.sum(total_change[i] <= 1e-5)
                print(unchangable_count)
                changing_count = int( 0.2 * (total_change[i].size - unchangable_count) )
                changing_count = max(changing_count, 1) # at least 1
                sorted_indices = np.unravel_index(np.argsort(total_change[i], axis=None), total_change[i].shape)
                sorted_indices = [indices[unchangable_count:unchangable_count + changing_count] 
                                  for indices in sorted_indices]
                sorted_indices.insert(0, i)
                valid[tuple(sorted_indices)] = 0

                print(valid[i].sum())
            
            # Exit if the attack is not successful
            # all successufully attacked samples have no more valid waypoints to remove
            if np.abs(valid[success_indices] - prev_valid[success_indices]).sum() == 0:
                break

        # Attack failed eventually
        # Return the last adv sample
        return last_adv