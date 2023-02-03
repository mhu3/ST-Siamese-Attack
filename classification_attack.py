import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle, bz2

from utils import load_data, normalize_trajectory_data, denormalize_trajectory_data, \
                  visualize_trajectory, set_seed
from argument import create_parser
from models import load_model

from fgsm_attack import FGSM
from cw_attack import CWAttack


def create_fgsm_attack_samples(model, samples, labels, attack_type="l0"):
    # Create attack class
    attack = FGSM(model, batch_size=5000, n=2, eps=0.05, iteration=50)
    
    # Define initial attackable mask
    # ignore padding, no time change, attack seek trajectories of one driver
    valid = get_valid_mask(
        samples, remove_padding=True, change_time=False, 
        unchanged_traj_indices= np.arange(5, 10)
    )

    # Create adversarial samples
    adv_samples = attack.attack(samples, labels, valid, attack_type)
    
    # Validate the adversarial samples 
    adv_samples = denormalize_trajectory_data(adv_samples) # fit back to grid
    adv_samples = normalize_trajectory_data(adv_samples) # normalize again

    return adv_samples

def create_cw_attack_samples(model, samples, labels, attack_type="l0"):
    
    # Create attack class
    attack = CWAttack(model, batch_size=5000, learning_rate=0.01,
                      max_iterations=8, initial_const=0.125/8, largest_const=1e9,
                      binary_search_steps=20, const_factor=2.0)
                
    
    # Define initial attackable mask
    # ignore padding, no time change, attack seek trajectories of one driver
    valid = get_valid_mask(
        samples, remove_padding=True, change_time=False, 
        unchanged_traj_indices=np.arange(5, 10)
    )

    # Create adversarial samples
    adv_samples = attack.attack(samples, labels, valid, attack_type)

    # Validate the adversarial samples 
    adv_samples = denormalize_trajectory_data(adv_samples)  # fit back to grid
    adv_samples = normalize_trajectory_data(adv_samples)  # normalize again

    return adv_samples

def get_valid_mask(
    samples, 
    remove_padding=True, change_time=False, 
    unchanged_traj_indices=[]
):
    """ Generate valid mask for attack """
    valid = np.ones(samples.shape, dtype=np.int32)
    # set padding to non changable
    if remove_padding:
        plates, trajs, rows = np.where(np.sum(samples, axis=3) == 0)
        valid[plates, trajs, rows, :] = 0
    # set time to non changable
    if not change_time:
        valid[:, :, :, 2] = 0
    # change only some of the trajectories
    if len(unchanged_traj_indices) > 0:
        valid[:, unchanged_traj_indices, :, :] = 0
    return valid
    

def main(opts, method, norm, tag_suffix=""):
    """ Launch attack against the model 
        Args:
            opts: opts from argument parser
            method: attack method (fgsm, cw)
            norm: attack norm (linf, l2, l0)
            tag_suffix: suffix for file naming
    """
    # Checking parameters
    assert(opts.traj_type in ['seek', 'serve', 'all'])
    # prepare file naming
    tag = str(opts.num_plates) + '_plates_' + \
          str(opts.num_days) +  '_days_' + \
          str(opts.traj_type) + '_traj' + \
          tag_suffix  # for different conditions

    # Load data
    # load trajectory
    if opts.traj_type == 'seek':
        pass  #TODO To be generated
    elif opts.traj_type == 'serve':
        pass  #TODO To be generated
    elif opts.traj_type == 'all':
        X_test, y_test = load_data(opts.data_path + 'classification_testing_set.pkl')

    # Normalize data
    X_test = normalize_trajectory_data(X_test)
    num_class = np.unique(y_test).shape[0]
    y_test = tf.keras.utils.to_categorical(y_test, num_classes = num_class)

    # Load model
    # Load model
    model = load_model(opts.model_path + 'model_' + tag + '_best.h5')
    
    # Attack method
    if method == "fgsm":
        attack_fun = create_fgsm_attack_samples
    elif method == "cw":
        attack_fun = create_cw_attack_samples

    X_adv = attack_fun(model, X_test, y_test, attack_type=norm)
 
    # Test attack
    # original samples
    model.evaluate(X_test, y_test, batch_size=5000)
    # adversarial samples
    model.evaluate(X_adv, y_test, batch_size=5000)
 
    # new prediction
    y_adv_seen = model.predict(X_adv, batch_size=5000)
    
    # save adversarial samples
    pickle.dump(
        [X_adv, y_adv_seen], 
        bz2.open(
            opts.data_path + method + "_" + norm +
            '_testing_set_' + tag_suffix + '.pkl', 'wb'
        )
    )

if __name__ == "__main__":
    # Argument
    parser = create_parser()
    opts = parser.parse_args()
    
    set_seed(1)
    # main(opts, method="fgsm", norm="l0", tag_suffix="_class")
    main(opts, method="fgsm", norm="linf", tag_suffix="_class")
    