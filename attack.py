import keras
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, normalize_trajectory_data, denormalize_trajectory_data, \
                  visualize_trajectory
from argument import create_parser
from models import load_model

from fgsm_attack import FGSM
from cw_attack import CWAttack


def create_fgsm_attack_samples(model, samples, labels, attack_type="linf"):
    # Create attack class
    attack = FGSM(model)
    
    # Create adversarial samples
    if attack_type == "linf":
        adv_samples = attack.linf_attack(samples, labels)
    elif attack_type == "l2":
        pass
        # adv_samples = attack.l2_attack(samples, labels)
    elif attack_type == "l0":
        pass
        # adv_samples = attack.l0_attack(samples, labels)
    
    # Validate the adversarial samples 
    adv_samples = denormalize_trajectory_data(adv_samples) # fit back to grid
    adv_samples = normalize_trajectory_data(adv_samples) # normalize again

    return adv_samples


def create_cw_attack_samples(model, samples, labels, attack_type="l0"):
    # Create attack class
    attack = CWAttack(model, batch_size=1024, learning_rate=1e-2,
                      max_iterations=20, initial_const=0.01, largest_const=1,
                      binary_search_steps=1, const_factor=5.0)
    
    # Create adversarial samples
    if attack_type == "linf":
        pass
        # adv_samples = attack.linf_attack(samples, labels)
    elif attack_type == "l2":
        adv_samples = attack.l2_attack(samples, labels)
    elif attack_type == "l0":
        adv_samples = attack.l0_attack(samples, labels)

    # Validate the adversarial samples 
    adv_samples = denormalize_trajectory_data(adv_samples) # fit back to grid
    adv_samples = normalize_trajectory_data(adv_samples) # normalize again        

    return adv_samples
    

def main(opts):
    # Load data
    # load trajectory
    if opts.traj_type == 'seek':
        pass #TODO To be generated
    elif opts.traj_type == 'serve':
        pass #TODO To be generated
    elif opts.traj_type == 'all':
        X_test_seen, y_test_seen = load_data(opts.data_path + 'testing_set_seen.pkl')
        X_test_unseen, y_test_unseen = load_data(opts.data_path + 'testing_set_unseen.pkl')

    # Normalize data
    X_test_seen = normalize_trajectory_data(X_test_seen)
    X_test_unseen = normalize_trajectory_data(X_test_unseen)

    # Load model
    model = load_model(opts.model_path + 'model_500_plates_8_days_all_traj_best.h5')
    
    # Create adversarial samples
    # X_cw_adv_seen = create_cw_attack_samples(model, X_test_seen, y_test_seen, "l2")
    # X_cw_adv_unseen = create_cw_attack_samples(model, X_test_unseen, y_test_unseen, "l2")
    X_cw_adv_seen = create_cw_attack_samples(model, X_test_seen[505:510], y_test_seen[505:510], "l0")
    # X_cw_adv_unseen = create_cw_attack_samples(model, X_test_unseen, y_test_unseen, "l0")
    # X_fgsm_linf_adv_seen = create_fgsm_attack_samples(model, X_test_seen, y_test_seen, "linf")
    # X_fgsm_linf_adv_unseen = create_fgsm_attack_samples(model, X_test_unseen, y_test_unseen, "linf")

    # Test attack
    # original samples
    loss_seen, acc_seen = model.evaluate(X_test_seen[505:510], y_test_seen[505:510])
    # loss_unseen, acc_unseen = model.evaluate(X_test_unseen, y_test_unseen)
    # adversarial samples
    loss_cw_adv_seen, acc_cw_adv_seen = model.evaluate(X_cw_adv_seen, y_test_seen[505:510])
    # loss_cw_adv_unseen, acc_cw_adv_unseen = model.evaluate(X_cw_adv_unseen, y_test_unseen)
    # loss_fgsm_linf_adv_seen, acc_fgsm_linf_adv_seen = model.evaluate(X_fgsm_linf_adv_seen, y_test_seen)
    # loss_fgsm_linf_adv_unseen, acc_fgsm_linf_adv_unseen = model.evaluate(X_fgsm_linf_adv_unseen, y_test_unseen)

    # Visualize result
    # # print the unseen prediction
    # print(y_test_seen[:10].T)
    # print((model.predict(X_test_seen[:10, :, :, :])).T)
    # print((model.predict(X_cw_adv_seen[:10, :, :, :])).T)

    # # print the unseen prediction
    # print(y_test_unseen[:10].T)
    # print((model.predict(X_test_unseen[:10, :, :, :])).T)
    # print((model.predict(X_cw_adv_unseen[:10, :, :, :])).T)

    visualize_attack_result(X_test_seen[505:510], X_cw_adv_seen, plate_idx=2)
    # visualize_attack_result(X_test_unseen, X_cw_adv_unseen, plate_idx=2)


def visualize_attack_result(X_ori, X_adv, plate_idx, fig_name="attack_result"):
    # denormalize the data
    X_ori = denormalize_trajectory_data(X_ori)
    X_adv = denormalize_trajectory_data(X_adv)

    # Plot the trajectories
    plt.figure(figsize=(50, 30))
    plt.subplots_adjust(hspace=0.5)
    for t in range(X_ori.shape[1]):
        plt.subplot(4, 5, t+1)

        # Plot the original trajectory and the adversarial trajectory
        visualize_trajectory(X_ori[plate_idx, t], 'o-', False)
        visualize_trajectory(X_adv[plate_idx, t], 'r-', False)
        plt.legend(['original', 'adversarial'])

    plt.savefig(fig_name + '.png')


if __name__ == "__main__":
    # Argument
    parser = create_parser()
    opts = parser.parse_args()

    main(opts)
    