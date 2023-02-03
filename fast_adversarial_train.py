import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import load_data, set_seed, \
                  normalize_trajectory_data, denormalize_trajectory_data
from argument import create_parser
from models import load_model


def train(opts, tag_suffix="", retrain=False):
    """ Fast adversarial training. """
    # Define the tag
    tag = str(opts.num_plates) + '_plates_' + \
          str(opts.num_days) +  '_days_' + \
          str(opts.traj_type) + '_traj' + \
          tag_suffix  # for different conditions

    # Load data
    # load trajectory
    if opts.traj_type == 'seek':
        pass #TODO To be generated
    elif opts.traj_type == 'serve':
        pass #TODO To be generated
    elif opts.traj_type == 'all':
        samples, labels = load_data(opts.data_path + 'classification_training_set.pkl')
    # normalize data
    samples = normalize_trajectory_data(samples)
    # one-hot encoding
    num_classes = np.unique(labels).shape[0]
    labels = tf.keras.utils.to_categorical(labels, num_classes)

    # Load model for ST-Siamese
    model = load_model(opts.model_path + 'model_' + tag + '_last.h5')
    # define the loss function
    if model.loss == 'binary_crossentropy':
        loss_object = tf.keras.losses.BinaryCrossentropy()
    else:
        loss_object = tf.keras.losses.CategoricalCrossentropy()            
    
    # Hyper-parameters
    num_batch = len(samples) // opts.batch_size
    ALPHA_DELTA = 0.02
    ETA = 2

    # Start training
    for epoch in tqdm(range(opts.epochs)):
        # shuffle index
        idx = np.arange(len(samples))
        np.random.shuffle(idx)

        # train on each batch
        for n in tqdm(range(num_batch)):
            # save intermediate model when batch size is too larage
            if (n+1) % 1000 == 0:
                model.save(
                    opts.model_path + 'model_' + tag + 
                    '_epoch_' + str(epoch) + 
                    '_batch_' + str((n+1)//1000) + '.h5'
                )
            
            # Get the current batch
            x_batch = samples[idx[n*opts.batch_size:(n+1)*opts.batch_size]]
            x_batch = tf.convert_to_tensor(x_batch)
            y_batch = labels[idx[n*opts.batch_size:(n+1)*opts.batch_size]]
            shape = x_batch.shape

            # Predict on the current batch
            pred_batch = model(x_batch)
            # Find the correctly predicted ones for multi-class classification
            correct = np.argmax(pred_batch, axis=1) == \
                      np.argmax(y_batch, axis=1)

            # Define the valid mask
            valid = np.ones(shape, dtype=np.int32)
            # set padding to non changable
            plates, trajs, rows = np.where(np.sum(x_batch, axis=3) == 0)
            valid[plates, trajs, rows, :] = 0
            # set time to non changable
            valid[:, :, :, 2] = 0
            # # set serve to non changable
            valid[:, 5:10, :, :] = 0
            
            # Initialize uniform noise
            delta = np.zeros(shape, dtype=np.float32)
            delta[:,:,:,0] = np.random.uniform(low=-ETA/49, high=ETA/49, size=shape[:3])
            delta[:,:,:,1] = np.random.uniform(low=-ETA/92, high=ETA/92, size=shape[:3])
            delta = tf.convert_to_tensor(delta)
            x_adv_batch = x_batch + delta

            # while attack is still successful
            while True:
                x_adv_batch = tf.convert_to_tensor(x_adv_batch)
                with tf.GradientTape() as tape:
                    # Watch the input
                    tape.watch(x_adv_batch)
                    # Compute loss
                    prediction = model(x_adv_batch)
                    loss = loss_object(y_batch, prediction)

                # Get the gradients of the loss w.r.t to the input
                gradients = tape.gradient(loss, x_adv_batch)
                # Get the sign of the gradients to create the perturbation
                signed_grads = tf.sign(gradients)

                # Add perturbation to the adversarial example
                delta += valid * ALPHA_DELTA * signed_grads

                # Clip the adversarial example to be in the range
                delta_np = delta.numpy()
                delta_np[:, :, :, 0] = np.clip(delta_np[:, :, :, 0], -ETA/49, ETA/49)
                delta_np[:, :, :, 1] = np.clip(delta_np[:, :, :, 1], -ETA/92, ETA/92)
                delta = tf.convert_to_tensor(delta_np)

                # Add the noise to the original sample
                x_adv_batch = x_batch + delta
                # change the type of the tensor to numpy
                x_adv_batch = x_adv_batch.numpy()
                # denormalize and normalize the trajectory
                # to fit back to the grid
                x_adv_batch = denormalize_trajectory_data(x_adv_batch)
                x_adv_batch = normalize_trajectory_data(x_adv_batch)

                # Find the current samples that are successfully attacked
                prediction = model(x_adv_batch)
                success = np.logical_and(
                    correct,
                    np.argmax(prediction, axis=1) != np.argmax(y_batch, axis=1),
                )
                # find the indices of the samples that are successfully attacked
                success_indices = np.where(success)[0]

                # End if no more sample is successfully attacked
                if len(success_indices) == 0:
                    break

                # Update model based on the successfully attacked samples
                model.train_on_batch(x_adv_batch[success_indices], y_batch[success_indices])

                # Check gradient to update the mask
                # compute total change
                total_change = np.sum(np.abs(gradients * valid), axis=3) 
                
                # for every successfully attacked sample, 
                # set 20% of the current valid waypoints to 0
                for s in success_indices:
                    # 1, if the change is insignificant
                    insignificant_indices = list( np.where( total_change[s] <= 1e-5 ) )
                    insignificant_indices.insert(0, s)
                    valid[tuple(insignificant_indices)] = 0
                    
                    # 2, set 20% of the current valid waypoints to 0
                    unchangable_count = np.sum(total_change[s] <= 1e-5)
                    changing_count = int( 0.2 * (total_change[s].size - unchangable_count) )
                    changing_count = max(changing_count, 1)  # at least 1

                    sorted_indices = np.unravel_index(np.argsort(total_change[s], axis=None), total_change[s].shape)
                    sorted_indices = [indices[unchangable_count:unchangable_count + changing_count] 
                                        for indices in sorted_indices]
                    sorted_indices.insert(0, s)
                    valid[tuple(sorted_indices)] = 0

        # End of epoch
        # save current model
        model.save(opts.model_path + 'model_' + tag + 'epoch' + str(epoch) + '.h5')


if __name__ == "__main__":
    # Load arguments
    parser = create_parser()
    opts = parser.parse_args()
    # Set random seed
    set_seed(1)

    # Run training
    train(opts)
