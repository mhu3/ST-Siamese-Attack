import numpy as np
import tensorflow as tf
from copy import copy
from tqdm import tqdm

class FGSM():
    """
    Running Fast Gradient Sign Method 
    adversarial attacks on a model for Sequential data.
    """
    def __init__(self, model):
        self.model = model

    def linf_attack(self, samples, labels, eps=0.001, iteration=20):
        # normalization factor  49, 92
        # Binary cross entropy (0/1)
        loss_object = tf.keras.losses.BinaryCrossentropy()

        # Change label size to [batch_size, 1]
        labels = labels[:, np.newaxis]

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
            # Get the sign of the gradients to create the perturbation
            signed_grads = tf.sign(gradients)

            # Constraint on sequential data
            # we should not change the padding
            np_grads = signed_grads.numpy()
            for i in range(np_grads.shape[0]):
                # find rows with all 0s
                trajs, rows = np.where(np.sum(samples[i], axis=2) == 0)
                np_grads[i, trajs, rows, :] = 0

                # avoid changing time
                np_grads[i, :, :, 2] = 0

                # acoid changing the serve trajectory
                np_grads[i, 10:20, :, :] = 0

            # Perturb the input
            signed_grads = tf.convert_to_tensor(np_grads)
            pairs_adv = pairs_adv + eps * signed_grads
            
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
        labels = labels[:, np.newaxis]

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
    
    def l0_attack(self, samples, labels, eps=0.05, iteration=500):
              # normalization factor  49, 92
        # Binary cross entropy (0/1)
        loss_object = tf.keras.losses.BinaryCrossentropy()
        
        # Change label size to [batch_size, 1]
        labels = labels[:, np.newaxis]

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

            purtabation = np.sum(np.abs(diff), axis=(1,2,3))
            purtabation = purtabation.reshape(-1, 1)    
            
            # in each iteration,we only allow to change one point\
            # so we set the purtabation of the other points to be 0
            for i in range(purtabation.shape[0]):
                purtabation[i, np.where(purtabation[i] > 0)] = 1
            purtabation = tf.convert_to_tensor(purtabation)
            pairs_adv = pairs_adv * purtabation
            
            
