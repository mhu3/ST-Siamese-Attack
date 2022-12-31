import logging
import pickle, bz2
import numpy as np
import tensorflow as tf
# from sklearn.model_selection import train_test_split

from argument import create_parser
from utils import create_dir, load_data, normalize_trajectory_data
from models import build_lstm_siamese, load_model


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
          "_2"
    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s %(levelname)s %(message)s',
                        filename = opts.log_path + 'log_' + tag + '.log',
                        filemode = 'a')
    
    # Load data
    # load trajectory
    if opts.traj_type == 'seek':
        pass #TODO To be generated
    elif opts.traj_type == 'serve':
        pass #TODO To be generated
    elif opts.traj_type == 'all':
        X_train, y_train  = load_data(opts.data_path + 'training_set.pkl')
        X_test_seen, y_test_seen = load_data(opts.data_path + 'testing_set_seen.pkl')
        X_test_unseen, y_test_unseen = load_data(opts.data_path + 'testing_set_unseen.pkl')

    # Normalize data
    X_train = normalize_trajectory_data(X_train)
    X_test_seen = normalize_trajectory_data(X_test_seen)
    X_test_unseen = normalize_trajectory_data(X_test_unseen)

    # Build model
    num_traj_feature = 3 if not opts.with_speed else 4
    model = build_lstm_siamese(num_trajs=opts.num_trajs, 
                               padding_length=opts.padding_length, 
                               num_traj_feature=num_traj_feature,
                               traj_type=opts.traj_type)
    model.summary()

    # Continue training
    # model = load_model(opts.model_path + 'model_' + tag + '_best.h5')

    # Start training
    callbacks = \
    [
        tf.keras.callbacks.CSVLogger(opts.log_path + 'log_' + tag + '.csv', separator=",", append=True),
        tf.keras.callbacks.ModelCheckpoint(opts.model_path + 'model_' + tag + '_best.h5',
                                           monitor='val_loss', save_best_only=True),
        # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3),
    ]
    history = model.fit(X_train, y_train, validation_data = [X_test_seen, y_test_seen], 
                        batch_size=opts.batch_size, epochs=opts.epochs,
                        callbacks=callbacks)

    # Log training history
    history_dict = history.history
    accuracy = history_dict["accuracy"]
    loss = history_dict["loss"]
    val_accuracy = history_dict["val_accuracy"]
    val_loss = history_dict["val_loss"]
    for i in range(len(accuracy)):
        # logging.info("Epoch: {0}, accuracy: {1}, loss: {2}".format(i, accuracy[i], loss[i])) # val_accuracy[i], val_loss[i]
        logging.info("Epoch: {0}, accuracy: {1}, loss: {2}, val_accuracy: {3}, val_loss: {4}".format(i, accuracy[i], loss[i], val_accuracy[i], val_loss[i]))
    # pickle.dump([accuracy, loss], open(opts.log_path +'history_{0}.pkl'.format(tag), 'wb'))
    pickle.dump([accuracy, loss, val_accuracy, val_loss], open(opts.log_path +'history_{0}.pkl'.format(tag), 'wb'))
    
    # Save latest model
    model.save(opts.model_path + 'model_' + tag + '_last.h5')
    
    # Start testing
    test_loss_seen, test_acc_seen = model.evaluate(X_test_seen, y_test_seen)
    test_loss_unseen, test_acc_unseen = model.evaluate(X_test_unseen, y_test_unseen)
    # print testing accuracy and loss
    logging.info("seen accuracy: {0}, unseen accuracy: {1}".format(test_acc_seen, test_acc_unseen))


if __name__ == '__main__':
    parser = create_parser()
    opts = parser.parse_args()
    main(opts)
