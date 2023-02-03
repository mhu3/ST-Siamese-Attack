import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from argument import create_parser
from utils import create_dir, load_data, normalize_trajectory_data, set_seed
from models import build_lstm, load_model


def train(opts, tag_suffix="class", retrain=False):
    """ Load data, model and train the model. """
    # Checking parameters
    assert(opts.traj_type in ['seek', 'serve', 'all'])

    # Create directories for model and log
    create_dir(opts.log_path)
    create_dir(opts.model_path)
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
        X_train, y_train = load_data(opts.data_path + 'classification_training_set.pkl') 
        X_val, y_val = load_data(opts.data_path + 'classification_validation_set.pkl') 
        X_test, y_test = load_data(opts.data_path + 'classification_testing_set.pkl')
    # Normalize data
    X_train = normalize_trajectory_data(X_train)
    X_val = normalize_trajectory_data(X_val)
    X_test = normalize_trajectory_data(X_test)
    # one-hot encoding
    num_class = np.unique(y_train).shape[0]
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_class)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_class)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_class)

    # Train from a new model
    if not retrain:
        num_traj_feature = 3 if not opts.with_speed else 4
        model = build_lstm(
            num_class=num_class,
            num_trajs=opts.num_trajs, 
            padding_length=opts.padding_length, 
            num_traj_feature=num_traj_feature,
            traj_type=opts.traj_type
        )
    # Continue training
    else:
        model = load_model(opts.model_path + 'model_' + tag + '_best.h5')
    model.summary()

    # Start training
    callbacks = [
        tf.keras.callbacks.CSVLogger(
            opts.log_path + 'log_' + tag + '.csv', separator=",", append=True
        ),  # log training history
        tf.keras.callbacks.ModelCheckpoint(
            opts.model_path + 'model_' + tag + '_best.h5',
            monitor='val_loss', save_best_only=True
        ),  # save the best model
        tf.keras.callbacks.ModelCheckpoint(
            opts.model_path + 'model_' + tag + '_last.h5',
            monitor='val_loss', save_freq="epoch"
        ),  # save the latest model
    ]

    # TODO Hyperparameters used in classification but not siamese
    EPOCH = 100
    BATCH_SIZE = 16
    model.fit(
        X_train, y_train, 
        batch_size=BATCH_SIZE, epochs=EPOCH,
        callbacks=callbacks,
        validation_data=[X_val, y_val], 
    )
    
    # Start testing
    # load best model
    model = load_model(opts.model_path + 'model_' + tag + '_best.h5')
    # test
    model.evaluate(X_test, y_test, batch_size=5000)


if __name__ == '__main__':
    # Load arguments
    parser = create_parser()
    opts = parser.parse_args()
    # Set random seed
    set_seed(1)

    # Run training
    train(opts)
