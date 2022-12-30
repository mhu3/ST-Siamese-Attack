import logging
import pickle, bz2
import numpy as np
import tensorflow as tf
# from sklearn.model_selection import train_test_split

from argument import create_parser
from utils import create_dir, load_data, get_trajectories, get_trajectory_pair_sample
from models import build_lstm_siamese


def get_classification_pair_samples(raw_trajs, plates, selected_plates, selected_days, 
                                    test_plate, test_day,
                                    traj_type='all', num_trajs=10, padding_length=60):
    """Get classification pair samples for a given plate index."""
    
    # 

    # Get trajectory pair of the given plate
    X = []
    num_samples = len(selected_plates) * len(selected_days)
    for i in range(num_samples):
        # continuously select one driver
        plate_idx1 = test_plate
        plate_idx2 = selected_plates[i // len(selected_days)]
        # continuously select two different days
        selected_day1 = test_day
        selected_day2 = selected_days[i % len(selected_days)]

        # get one trajectory pair sample
        sample = get_trajectory_pair_sample(raw_trajs, plates, 
                                            plate_idx1, plate_idx2, selected_day1, selected_day2,
                                            traj_type, num_trajs, padding_length)
        X.append(sample)
    return np.array(X)


def main(opts):
    """Load data, model and train the model."""
    # Checking
    assert(opts.traj_type in ['seek', 'serve', 'all'])

    # Create directories for model and log
    create_dir(opts.log_path)
    create_dir(opts.model_path)
    # Prepare logging file
    tag = str(opts.num_plates) + '_plates_' + \
          str(opts.num_days) +  '_days_' + \
          str(opts.traj_type) + '_traj' + \
          "_3"
    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s %(levelname)s %(message)s',
                        filename = opts.log_path + 'log_' + tag + '.log',
                        filemode = 'a')

    # Load original data
    raw_trajs = pickle.load(open('./dataset/trajs_without_speed500.pkl', 'rb'))
    plates = pickle.load(open('./dataset/plates.pkl', 'rb'))

    # Load trained model
    model = tf.keras.models.load_model(opts.model_path + 'model_' + tag + '_best.h5')
    model.summary()

    # Start testing
    acc_count = 0
    # selected_plates = np.arange(opts.num_plates)
    # selected_days = np.arange(opts.num_days)
    selected_plates = np.random.randint(0, opts.num_plates, 10)
    selected_days = np.arange(opts.num_days)
    for test_plate in selected_plates:
        # get paried classfication test samples
        samples = get_classification_pair_samples(raw_trajs, plates, selected_plates, selected_days,
                                                  test_plate, test_day=9,
                                                  traj_type=opts.traj_type, num_trajs=opts.num_trajs, 
                                                  padding_length=opts.padding_length)
        yhat = model.predict(samples, batch_size=512)
        # get the mean of similarity score of each plate
        yhat = yhat.reshape(len(selected_plates), len(selected_days)).mean(axis=1)
        print(yhat)
        # check accuracy
        if selected_plates[yhat.argmax()] == test_plate:
            acc_count += 1

    # print testing accuracy and loss
    logging.info("Classification accuracy: {0}".format(acc_count / len(selected_plates)))


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    main(opts)
