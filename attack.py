import keras
import numpy as np

from utils import load_data, normalize_trajectory_data, denormalize_trajectory_data
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


def create_cw_attack_samples(model, samples, labels, attack_type="linf"):
    # CW attack requires the model output to be logits (not softmax)
    # model_log = keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
    # Create attack class
    sess = keras.backend.get_session()
    attack = CWAttack(sess, model, samples.shape[1:])
    
    # Create adversarial samples
    if attack_type == "linf":
        pass
        # adv_samples = attack.linf_attack(samples, labels)
    elif attack_type == "l2":
        pass
        # adv_samples = attack.l2_attack(samples, labels)
    elif attack_type == "l0":
        adv_samples = attack.l0_attack(samples[0:1], labels)

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
    # Split trajectories into multiple input list
    # X_test_seen = [X_test_seen[:, i, :, :] for i in range(X_test_seen.shape[1])]
    # X_test_unseen = [X_test_unseen[:, i, :, :] for i in range(X_test_unseen.shape[1])]

    # Load model
    model = load_model(opts.model_path + 'model_500_plates_8_days_all_traj_best.h5')
    
    # Create adversarial samples
    # X_cw_l0_adv_seen = create_cw_attack_samples(model, X_test_seen, y_test_seen, "l0")
    # X_cw_l0_adv_unseen = create_cw_attack_samples(model, X_test_seen, y_test_seen, "l0")
    X_fgsm_linf_adv_seen = create_fgsm_attack_samples(model, X_test_seen, y_test_seen, "linf")
    X_fgsm_linf_adv_unseen = create_fgsm_attack_samples(model, X_test_unseen, y_test_unseen, "linf")

    # Test attack
    # original samples
    loss_seen, acc_seen = model.evaluate(X_test_seen, y_test_seen)
    loss_unseen, acc_unseen = model.evaluate(X_test_unseen, y_test_unseen)
    # adversarial samples
    # loss_cw_l0_adv_seen, acc_cw_l0_adv_seen = model.evaluate(X_cw_l0_adv_seen, y_test_seen)
    # loss_cw_l0_adv_unseen, acc_cw_l0_adv_unseen = model.evaluate(X_cw_l0_adv_unseen, y_test_unseen)
    loss_fgsm_linf_adv_seen, acc_fgsm_linf_adv_seen = model.evaluate(X_fgsm_linf_adv_seen, y_test_seen)
    loss_fgsm_linf_adv_unseen, acc_fgsm_linf_adv_unseen = model.evaluate(X_fgsm_linf_adv_unseen, y_test_unseen)


if __name__ == "__main__":
    # Argument
    parser = create_parser()
    opts = parser.parse_args()

    main(opts)
    