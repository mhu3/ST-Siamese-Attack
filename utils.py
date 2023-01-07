import os
import pickle, bz2
import random
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences


def create_dir(directory):
    """Creates a directory if it does not already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(file_path):
    return pickle.load(bz2.open(file_path, 'rb'))


### For data normalization
def normalize_trajectory_data(data):
    X = data.astype(np.float32)
    # Normalize data
    X[:, :, :, 0] = X[:, :, :, 0] / 49    # x
    X[:, :, :, 1] = X[:, :, :, 1] / 92    # y
    X[:, :, :, 2] = X[:, :, :, 2] / 288   # t
    return X

def denormalize_trajectory_data(data):
    X = data.copy()
    # Denormalize data
    X[:, :, :, 0] = X[:, :, :, 0] * 49    # x
    X[:, :, :, 1] = X[:, :, :, 1] * 92    # y
    X[:, :, :, 2] = X[:, :, :, 2] * 288   # t
    X = np.round(X).astype(np.int32)
    return X

### For data visualization
def visualize_trajectory(traj, linetype='o-', shown_in_whole_map=False):
    """Draw given trajectory
    Args:
        traj: trajectory
        linetype: line type
    """
    # Skip padding zeros
    zero_rows = np.where(np.sum(traj, axis=1) == 0)[0]
    if len(zero_rows) != 0:
        start_row = zero_rows[-1] + 1
    else:
        start_row = 0

    # Get trajectory
    x = traj[start_row:, 1]
    y = traj[start_row:, 0]
    t = traj[start_row:, 2]
    # plot trajectory
    plt.plot(x, y, linetype, markersize=3)
    # plot start and end point
    plt.scatter(x[0], y[0], c='k', marker='x')
    plt.scatter(x[-1], y[-1], c='k', marker='*')
    
    # Figure settings
    if shown_in_whole_map:
        plt.xlim(0, 92)
        plt.ylim(0, 49)
    # set x, y axis scale to be the same
    plt.gca().set_aspect('equal', adjustable='box')


### For data generation
def get_all_trajectories(raw_trajs, plates, plate_indices, selected_days, traj_types):
    """Get trajectory of specified days and input types of given drivers 
    Args:
        raw_trajs: raw trajectories data
        plates: plates data
        plate_indices: specified plate indices
        selected_days: specified days
        traj_types: specified trajectory types
    Returns:
        trajectories: trajectories of specified days and input types of given drivers
                      [Driver1's trajectories, Driver2's trajectories, ...]
                          |-> [Day1's trajectories, Day2's trajectories, ...]
                                |-> [Trajectory1, Trajectory2, ...]
    """
    # Get trajectories of each plate
    result_trajs = []
    for plate in [plates[i] for i in plate_indices]:

        # Get trajectories of each day
        plate_trajs = []
        for day in selected_days:

            # Get trajectories of each input type
            day_trajs = []
            for traj_type in traj_types:
                
                trajs = raw_trajs[plate][day][traj_type]

                day_trajs.extend(trajs)
            plate_trajs.append(day_trajs)
        # Result trajectories of each plate's N days' trajectories
        result_trajs.append(plate_trajs)

    return result_trajs


def get_trajectories(raw_trajs, plates, plate_idx, selected_day, traj_type):
    ''' Same as get_all_trajectories, but only getting trajectories of one plate on one day of one type'''
    return get_all_trajectories(raw_trajs, plates, [plate_idx], [selected_day], [traj_type])[0][0]


def create_dataset(raw_trajs, plates, num_samples, 
                   plate_indices, selected_days, traj_type, num_trajs=10,
                   padding_length=60, pos_ratio=0.5):
    '''Create dataset for training
    Args:
        raw_trajs: raw trajectories data
        plates: name of drivers' plates
        plate_indices: selected plate indices
        selected_days: selected from given days
        traj_type: spcified trajectory type
        num_trajs: number of trajectories per sample per driver
    
    Returns:
        X: data
        y: labels
    '''
    # Sample size
    num_pos_samples = int(num_samples * pos_ratio)
    num_neg_samples = num_samples - num_pos_samples

    # Create total dataset
    traj_feature_size = len( get_trajectories(raw_trajs, plates, 0, 0, 'seek')[0][0] )
    X = np.zeros((num_samples, num_trajs*2, padding_length, traj_feature_size))
    y = np.zeros((num_samples, 1))

    # Positive samples
    for i in range(num_pos_samples):
        # randomly select one driver
        plate_idx1 = random.sample(plate_indices, 1)[0]
        plate_idx2 = plate_idx1
        # randomly select two different days
        selected_day1, selected_day2 = random.sample(selected_days, 2)

        # get one trajectory pair sample
        sample = get_trajectory_pair_sample(raw_trajs, plates, 
                                            plate_idx1, plate_idx2, selected_day1, selected_day2,
                                            traj_type, num_trajs, padding_length)
        X[i] = sample
        y[i] = 1

    # Negative samples
    for i in range(num_neg_samples):
        # randomly select two drivers
        plate_idx1, plate_idx2 = random.sample(plate_indices, 2)
        # randomly select two days separately
        selected_day1 = random.sample(selected_days, 1)[0]
        selected_day2 = random.sample(selected_days, 1)[0]
        
        # get one trajectory pair sample
        sample = get_trajectory_pair_sample(raw_trajs, plates, 
                                            plate_idx1, plate_idx2, selected_day1, selected_day2,
                                            traj_type, num_trajs, padding_length)
        X[num_pos_samples+i] = sample
        y[num_pos_samples+i] = 0

    return X, y


def get_trajectory_pair_sample(raw_trajs, plates,
                               plate_idx1, plate_idx2, selected_day1, selected_day2,
                               traj_type, num_trajs=10, padding_length=60):
    '''Get one sample of trajectory pair, given a pair of plates and days'''

    # Get desired trajectories
    if traj_type != 'all':
        trajs1 = get_trajectories(raw_trajs, plates, plate_idx1, selected_day1, traj_type)
        trajs2 = get_trajectories(raw_trajs, plates, plate_idx2, selected_day2, traj_type)
        all_trajs = [trajs1, trajs2]
        num_sample_trajs = num_trajs
        
    else:
        trajs1_seek = get_trajectories(raw_trajs, plates, plate_idx1, selected_day1, 'seek')
        trajs1_serve = get_trajectories(raw_trajs, plates, plate_idx1, selected_day1, 'serve')
        trajs2_seek = get_trajectories(raw_trajs, plates, plate_idx2, selected_day2, 'seek')
        trajs2_serve = get_trajectories(raw_trajs, plates, plate_idx2, selected_day2, 'serve')
        all_trajs = [trajs1_seek, trajs2_seek, trajs1_serve, trajs2_serve]
        num_sample_trajs = num_trajs // 2

    # Process and combine all trajectories into sample
    sample = []
    for trajs in all_trajs:
        # sample N number of trajectories from all trajectories
        # sample_trajs = random.sample(trajs, num_sample_trajs)
        indices = random.sample(list(range(len(trajs))), num_sample_trajs)
        indices.sort()
        sample_trajs = [trajs[i] for i in indices]
        # padding
        sample_trajs = pad_sequences(sample_trajs, maxlen=padding_length)
        # decouple and stack
        sample.extend(sample_trajs)

    return np.array(sample)
