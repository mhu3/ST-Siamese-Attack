import keras
import argparse
import numpy as np
from fgsm import FGSM
from utils import load_data


def create_fgsm_attack_samples(model, samples, labels, attack_type="linf"):
    # Create attack class
    attack = FGSM(model)
    
    # Create adversarial samples
    if attack_type == "linf":
        adv_samples = attack.linf_attack(samples, labels)
    elif attack_type == "l2":
        adv_samples = attack.l2_attack(samples, labels)
    elif attack_type == "l0":
        adv_samples = attack.l0_attack(samples, labels)

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

    # Normlization
    def normalize_trajectory_data(X):
        X = X.astype(np.float32)
        # Normalize data
        X[:, :, :, 0] = X[:, :, :, 0] / 49    # x
        X[:, :, :, 1] = X[:, :, :, 1] / 92    # y
        X[:, :, :, 2] = X[:, :, :, 2] / 288   # t
        return X
    # normalize data
    X_test_seen = normalize_trajectory_data(X_test_seen)
    X_test_unseen = normalize_trajectory_data(X_test_unseen)

    # Load model
    model = keras.models.load_model(opts.model_path + 'model_500_plates_8_days_all_traj_best.h5')
    
    # Create adversarial samples
    X_fgsm_linf_adv_seen = create_fgsm_attack_samples(model, X_test_seen, y_test_seen, "linf")
    X_fgsm_linf_adv_unseen = create_fgsm_attack_samples(model, X_test_seen, y_test_seen, "linf")

    # Test attack
    # original samples
    loss_seen, acc_seen = model.evaluate(X_test_seen, y_test_seen)
    loss_unseen, acc_unseen = model.evaluate(X_test_unseen, y_test_unseen)
    # adversarial samples
    loss_fgsm_linf_adv_seen, acc_fgsm_linf_adv_seen = model.evaluate(X_fgsm_linf_adv_seen, y_test_seen)
    loss_fgsm_linf_adv_unseen, acc_fgsm_linf_adv_unseen = model.evaluate(X_fgsm_linf_adv_unseen, y_test_unseen)


if __name__ == "__main__":
    # Argument
    parser = argparse.ArgumentParser()
    opts = parser.parse_args()

    main(opts)
    