import pickle, bz2
import numpy as np

from argument import create_parser
from utils import load_data, normalize_trajectory_data,\
                  denormalize_trajectory_data, set_seed
from models import load_model

from fgsm_attack import FGSM
from cw_attack import CWAttack


def create_fgsm_attack_samples(model, samples, labels, attack_type="linf"):
    # Create attack class
    attack = FGSM(model, batch_size=5000, eta=1, eps=0.01, iteration=2)
    
    # Define initial attackable mask
    # ignore padding, no time change, attack seek trajectories of one driver
    valid = get_valid_mask(
        samples, remove_padding=True, change_time=False, 
        unchanged_traj_indices=np.hstack((np.arange(0, 5), np.arange(10, 20)))
    )  

    # Create adversarial samples
    adv_samples = attack.attack(samples, labels, valid, attack_type)
    
    # Validate the adversarial samples 
    adv_samples = denormalize_trajectory_data(adv_samples)  # fit back to grid
    adv_samples = normalize_trajectory_data(adv_samples)  # normalize again

    return adv_samples


def create_cw_attack_samples(model, samples, labels, attack_type="l0"):
    
    # Create attack class
    attack = CWAttack(
        model, batch_size=5000, eta=1, 
        learning_rate=0.01, max_iterations=2, 
        initial_const=0.125/8, largest_const=1e9,
        binary_search_steps=20, const_factor=2.0
    )
    
    # Define initial attackable mask
    # ignore padding, no time change, attack seek trajectories of one driver
    valid = get_valid_mask(
        samples, remove_padding=True, change_time=False, 
        unchanged_traj_indices=np.hstack((np.arange(0, 5), np.arange(10, 20)))
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
        X_test_seen, y_test_seen = load_data(opts.data_path + 'testing_set_seen.pkl')
        X_test_unseen, y_test_unseen = load_data(opts.data_path + 'testing_set_unseen.pkl')
    # normalize data
    X_test_seen = normalize_trajectory_data(X_test_seen)
    X_test_unseen = normalize_trajectory_data(X_test_unseen)

    # Load model
    model = load_model(opts.model_path + 'model_' + tag + '_best.h5')

    # Attack method
    if method == "fgsm":
        attack_fun = create_fgsm_attack_samples
    elif method == "cw":
        attack_fun = create_cw_attack_samples

    # Create adversarial samples
    X_adv_seen = attack_fun(model, X_test_seen, y_test_seen, attack_type=norm)
    X_adv_unseen = attack_fun(model, X_test_unseen, y_test_unseen, attack_type=norm)

    # Test attack
    # original samples
    model.evaluate(X_test_seen, y_test_seen, batch_size=5000)
    model.evaluate(X_test_unseen, y_test_unseen, batch_size=5000)
    # adversarial samples
    model.evaluate(X_adv_seen, y_test_seen, batch_size=5000)
    model.evaluate(X_adv_unseen, y_test_unseen, batch_size=5000)
 
    # New prediction on attacked samples
    y_adv_seen = model.predict(X_adv_seen, batch_size=5000)
    y_adv_unseen = model.predict(X_adv_unseen, batch_size=5000)

    # Save adversarial samples with related predictions
    pickle.dump(
        [X_adv_seen, y_adv_seen], 
        bz2.open(
            opts.data_path + method + "_" + norm +
            '_testing_set_seen_' + tag_suffix + '.pkl', 'wb'
        )
    )
    pickle.dump(
        [X_adv_unseen, y_adv_unseen], 
        bz2.open(
            opts.data_path + method + "_" + norm +
            '_testing_set_unseen_' + tag_suffix + '.pkl', 'wb'
        )
    )   


if __name__ == '__main__':
    # Load arguments
    parser = create_parser()
    opts = parser.parse_args()
    # Set random seed
    set_seed(1)

    # Run training
    main(opts)
