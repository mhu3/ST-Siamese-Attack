import numpy as np
import tensorflow as tf
from copy import copy
from tqdm import tqdm
import time

class FGSM():
    """
    Running Fast Gradient Sign Method 
    adversarial attacks on a model for Sequential data.

    Args:
        model: A model object.
        batch_size: Batch size.
    """
    def __init__(self, model,
                 batch_size = 1000, eta = 1, eps = 0.001, iteration = 20):
        self.model = model

        self.loss_object = model.loss

        self.batch_size = batch_size
        self.eta = eta
        self.eps = eps
        self.iteration = iteration


    def linf_attack_batch(self, samples, labels, valid):
        # normalization factor 49, 92

        # Change label size to [batch_size, 1]
        # labels = labels[:, np.newaxis]
        pairs_ori = tf.Variable(samples, dtype=tf.float32)
        pairs_adv = tf.Variable(samples, dtype=tf.float32)

        shape = pairs_ori.shape

        # Iterative FGSM
        for i in tqdm(range(self.iteration)):
            # Compute current loss
            with tf.GradientTape() as tape:
                
                # Watch the input
                tape.watch(pairs_adv)
                
                # Compute loss
                prediction = self.model(pairs_adv)
                loss = self.loss_object(labels, prediction)

            # Get the gradients of the loss w.r.t to the input
            gradients = tape.gradient(loss, pairs_adv)
            # Get the sign of the gradients to create the perturbation
            signed_grads = tf.sign(gradients)

            pairs_adv = pairs_adv + self.eps * valid * signed_grads
            
            # compute the difference between the original and the perturbed trajectory
            diff = pairs_adv.numpy() - pairs_ori.numpy()

            # Clip the diff to be in the range of [-1/49, 1/49], [-1/92, 1/92] on x, y 
            diff[:, :, :, 0] = np.clip(diff[:, :, :, 0], -self.eta/49, self.eta/49)
            diff[:, :, :, 1] = np.clip(diff[:, :, :, 1], -self.eta/92, self.eta/92)
            diff = tf.convert_to_tensor(diff)
            
            # Add the difference to the original input
            pairs_adv = pairs_ori + diff

        return pairs_adv.numpy()


    def l2_attack_batch(self, samples, labels, valid, p_norm=2):
        # normalization factor  49, 92

        # Change label size to [batch_size, 1]
        # labels = labels[:, np.newaxis]

        pairs_ori = tf.Variable(samples, dtype=tf.float32)
        pairs_adv = tf.Variable(samples, dtype=tf.float32)

        shape = pairs_ori.shape
        
        # Iterative FGSM
        for i in tqdm(range(self.iteration)):
            # Compute current loss
            with tf.GradientTape() as tape:

                # Watch the input
                tape.watch(pairs_adv)

                # Compute loss
                prediction = self.model(pairs_adv)
                loss = self.loss_object(labels, prediction)

            # Get the gradients of the loss w.r.t to the input
            gradients = tape.gradient(loss, pairs_adv)

            # get the l2-norm of the gradients
            norm = tf.norm(tf.reshape(gradients, [gradients.shape[0], -1]), ord=p_norm, axis=1)
            norm = tf.reshape(norm, [gradients.shape[0], 1, 1, 1])

            # l2 norm of the gradients
            l2_grads = gradients / norm

            pairs_adv = pairs_adv + self.eps * valid * l2_grads
            
            # compute the difference between the original and the perturbed trajectory
            diff = pairs_adv.numpy() - pairs_ori.numpy()

            # Clip the diff to be in the range of [-1/49, 1/49], [-1/92, 1/92] on x, y 
            diff[:, :, :, 0] = np.clip(diff[:, :, :, 0], -self.eta/49, self.eta/49)
            diff[:, :, :, 1] = np.clip(diff[:, :, :, 1], -self.eta/92, self.eta/92)
            diff = tf.convert_to_tensor(diff)
            
            # Add the difference to the original input
            pairs_adv = pairs_ori + diff

        return pairs_adv.numpy()


    def l1_attack_batch(self, samples, labels, valid, eps=0.05, iteration=500):
        ''' Perform the L1 attack on the given batch of samples. '''
        return self.l2_attack_batch(samples, labels, valid, eps, iteration, p_norm=1)


    def l0_attack_batch(self, samples, labels, valid):
        """
        Perform the L0 attack on the given batch of samples.
        """
        
        pairs_ori = tf.Variable(samples, dtype=tf.float32)
        pairs_adv = tf.Variable(samples, dtype=tf.float32)

        shape = pairs_ori.shape

        # iterative until the attack is not successful for all samples
        last_adv = pairs_adv.numpy()
        diff = tf.zeros(shape, dtype=tf.float32)

        while True:
            # Iterative FGSM
            # diff = tf.zeros(shape, dtype=tf.float32)
            # pairs_adv = pairs_ori
            for i in tqdm(range(self.iteration)):
                # Compute current loss
                with tf.GradientTape() as tape:
                    
                    # Watch the input
                    tape.watch(pairs_adv)

                    # Compute loss
                    prediction = self.model(pairs_adv)
                    loss = self.loss_object(labels, prediction)

                # Get the gradients of the loss w.r.t to the input
                gradients = tape.gradient(loss, pairs_adv)
            
                # Get the sign of the gradients to create the perturbation
                signed_grads = tf.sign(gradients)

                # Add perturbation to the adversarial example
                diff += valid * self.eps * signed_grads

                # Clip the diff to be in the range of [-1/49, 1/49], [-1/92, 1/92] on x, y
                diff_np = diff.numpy()
                diff_np[:, :, :, 0] = np.clip(diff_np[:, :, :, 0], -self.eta/49, self.eta/49)
                diff_np[:, :, :, 1] = np.clip(diff_np[:, :, :, 1], -self.eta/92, self.eta/92)
                diff = tf.convert_to_tensor(diff_np)

                # Add the difference to the original input
                pairs_adv = pairs_ori + valid * diff

            # Check if the attack is successful
        
            #print(shape[0]*gradients[0, 0, 59, :])
            prediction = self.model(pairs_adv)
            
            # find the samples that are successfully attacked
            if prediction.shape[1] > 1:
                success = np.argmax(prediction, axis=1) != np.argmax(labels, axis=1)
            else:
                success = np.rint(prediction) != labels

            success_indices = np.where(success)[0]

            # update the last adv sample
            last_adv[success_indices] = pairs_adv.numpy()[success_indices]
        
            # Remove valid points
            # previous valid
            prev_valid = valid.copy()

            # Compute total change
            total_change = np.sum(np.abs(gradients * valid), axis=3) 

            # every successfully attacked sample, set 20% of the current valid waypoints to 0
            for s in success_indices:
                # 1, if the change is insignificant
                insignificant_indices = list( np.where( total_change[s] <= 1e-5 ) )
                insignificant_indices.insert(0, s)
                valid[tuple(insignificant_indices)] = 0
                # 2, set 20% of the current valid waypoints to 0
                unchangable_count = np.sum(total_change[s] <= 1e-5)
                #print(unchangable_count)
                changing_count = int( 0.2 * (total_change[s].size - unchangable_count) )
                changing_count = max(changing_count, 1) # at least 1
                sorted_indices = np.unravel_index(np.argsort(total_change[s], axis=None), total_change[s].shape)
                sorted_indices = [indices[unchangable_count:unchangable_count + changing_count] 
                                  for indices in sorted_indices]
                sorted_indices.insert(0, s)
                valid[tuple(sorted_indices)] = 0
            print('total change: ', np.sum(valid))

                # print(valid[s].sum())
            
            # Exit if the attack is not successful
            # all successufully attacked samples have no more valid waypoints to remove
            if np.abs(valid[success_indices] - prev_valid[success_indices]).sum() == 0:
                break

        # Attack failed eventually
        # Return the last adv sample
        return last_adv
    

    def attack(self, samples, labels, valid, norm='linf'):
        """
        Perform attack with given norm on the given samples.
        """
        # Select attack method
        if norm == 'linf':
            attack_method = self.linf_attack_batch
        elif norm == 'l2':
            attack_method = self.l2_attack_batch
        elif norm == 'l1':
            attack_method = self.l1_attack_batch
        elif norm == 'l0':
            attack_method = self.l0_attack_batch

        # Tik
        start = time.time()
        # Attack on batch
        adv_samples = np.zeros_like(samples)
        for i in range(0, samples.shape[0], self.batch_size):
            
            adv_samples[i : i + self.batch_size] = \
                attack_method(samples[i : i + self.batch_size], 
                              labels[i : i + self.batch_size],
                              valid[i : i + self.batch_size])
        # Tok
        end = time.time()
        print("running time: ", end - start)

        return adv_samples
