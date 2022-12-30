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

    def linf_attack(self, samples, labels, eps=0.001, iteration=10):
        # Binary cross entropy (0/1)
        loss_object = tf.keras.losses.BinaryCrossentropy()

        # Change label size to [batch_size, 1]
        labels = labels[:, np.newaxis]

        pair_input = tf.Variable(samples, dtype=tf.float32)

        # Iterative FGSM
        for i in tqdm(range(iteration)):
            # Compute current loss
            with tf.GradientTape() as tape:
                
                # Watch the input
                tape.watch(pair_input)
                
                # Compute loss
                prediction = self.model(pair_input)
                loss = loss_object(labels, prediction)

            # Get the gradients of the loss w.r.t to the input
            gradient = tape.gradient(loss, pair_input)
            # Get the sign of the gradients to create the perturbation
            signed_grad = tf.sign(gradient)

            # Constraint on sequential data
            # we should not change the padding
            for s in range(signed_grad.shape[0]):
                # Find rows with all 0s
                np_grad = signed_grad[s].numpy()
                trajs, rows = np.where( np.sum(np_grad, axis=2) == 0 )
                np_grad[trajs, rows, :] = 0
                
                # Avoid changing time
                np_grad[:, :, 2] = 0
                
                # Perturb the sample
                delta = eps * tf.convert_to_tensor(np_grad)
                pair_input[s] = pair_input[s] + delta

        return pair_input
    
    # def l2_attack(model, pairs, labels, eps=0.1, iteration=10, p_norm = 2):
