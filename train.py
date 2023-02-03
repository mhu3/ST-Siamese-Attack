import tensorflow as tf
from sklearn.model_selection import train_test_split

from argument import create_parser
from utils import create_dir, load_data, normalize_trajectory_data, set_seed
from models import build_lstm_siamese, load_model


def train(opts, tag_suffix="", retrain=False):
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
        pass  #TODO to be generated
    elif opts.traj_type == 'serve':
        pass  #TODO to be generated
    elif opts.traj_type == 'all':
        X_train, y_train = load_data(opts.data_path + 'training_set.pkl')  
        X_val, y_val = load_data(opts.data_path + 'validating_set.pkl')
        X_test_seen, y_test_seen = load_data(opts.data_path + 'testing_set_seen.pkl')
        X_test_unseen, y_test_unseen = load_data(opts.data_path + 'testing_set_unseen.pkl')
    # normalize data
    X_train = normalize_trajectory_data(X_train)
    X_test_seen = normalize_trajectory_data(X_test_seen)
    X_test_unseen = normalize_trajectory_data(X_test_unseen)
    # split training set into training and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=1
    )

    # Train from a new model
    if not retrain:
        num_traj_feature = 3 if not opts.with_speed else 4
        model = build_lstm_siamese(
            num_trajs=opts.num_trajs, 
            padding_length=opts.padding_length, 
            num_traj_feature=num_traj_feature, 
            traj_type=opts.traj_type
        )
    # Continue training
    else:
        model = load_model(opts.model_path + 'model_' + tag + '_last.h5')
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
    model.fit(
        X_train, y_train, 
        batch_size=opts.batch_size, epochs=opts.epochs,
        callbacks=callbacks,
        validation_data = [X_val, y_val], 
    )

    # Start testing
    # load best model
    model = load_model(opts.model_path + 'model_' + tag + '_best.h5')
    # test
    model.evaluate(X_test_seen, y_test_seen, batch_size=5000)
    model.evaluate(X_test_unseen, y_test_unseen, batch_size=5000)


if __name__ == '__main__':
    # Load arguments
    parser = create_parser()
    opts = parser.parse_args()
    # Set random seed
    set_seed(1)

    # Run training
    train(opts)
