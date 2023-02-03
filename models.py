import tensorflow as tf
import keras
from keras import optimizers
from keras.layers import Input, Dense, Bidirectional, LSTM, \
                         Concatenate, Activation
from keras.layers import Dropout, BatchNormalization, \
                         Conv1D, Flatten, GlobalMaxPooling1D
from keras.models import Model, Sequential


### Layer blocks ###
def seq_lstm():
    ''' trajectory features representation lstm '''
    lstm = Sequential()
    lstm.add(LSTM(200, return_sequences=True))
    lstm.add(LSTM(100))
    lstm.add(Dense(48, activation='relu'))
    return lstm


def seq_profile():
    '''model profile features representation mlp'''
    profile = Sequential()
    profile.add(Dense(64, activation='relu'))
    profile.add(Dense(32, activation='relu'))
    profile.add(Dense(8, activation='relu'))
    return profile


def seq_similarity():
    ''' model similarity prediction mlp '''
    prediction = Sequential()
    prediction.add(Dense(64, activation='relu'))
    prediction.add(Dense(32, activation='relu'))
    prediction.add(Dense(8, activation='relu'))
    prediction.add(Dense(1))
    prediction.add(Activation('sigmoid'))
    return prediction


def seq_classification(num_class):
    ''' model classification prediction mlp '''
    prediction = Sequential()
    prediction.add(Dense(64, activation='relu'))
    prediction.add(Dense(num_class))
    prediction.add(Activation('softmax'))
    return prediction


### Model for Siamese LSTM ###
def build_lstm_siamese(num_trajs, padding_length, num_traj_feature, traj_type):
    ''' Build siamese LSTM model '''

    # Define layers
    # trajectory features representation network 
    if traj_type == 'all':
        lstm1 = seq_lstm()  # for seek
        lstm2 = seq_lstm()  # for serve
    else:
        lstm = seq_lstm()  # for seek or serve
    # similarity prediction
    similarity = seq_similarity()

    # From paired input 2*(num_trajs) trajectories
    # to concatenated feature representation of each trajectory
    traj_shape = (num_trajs*2, padding_length, num_traj_feature)
    input = Input(traj_shape)
    # split trajectory
    slices = [
        Slice(
            begin=[0, i, 0, 0], 
            size=[-1, 1, -1, -1], 
            squeeze_dims=[1]
        )(input) 
        for i in range(traj_shape[0])
    ]
    # get each trajectory's representation
    traj_reps = []
    for i, slice in enumerate(slices):
        if traj_type == 'all':
            if i < num_trajs:
                traj_rep = lstm1(slice)  # for seek
            else:
                traj_rep = lstm2(slice)  # for serve
        else:
            traj_rep = lstm(slice)  # for seek or serve
        traj_reps.append(traj_rep)
    # merge trajectory representation
    traj_reps = Concatenate(axis=-1)(traj_reps)
    
    # Similarity MLP output
    prediction = similarity(traj_reps)
    
    # Define model
    model = Model(inputs=input, outputs=prediction)
    model.compile(
        loss="binary_crossentropy", 
        optimizer=optimizers.Adam(0.0001), 
        metrics=['accuracy']
    )
    return model


### Model for Siamese Classification ###
def build_lstm(num_class, num_trajs, padding_length, num_traj_feature, traj_type):
    ''' Build LSTm model for classification '''

    # Define layers
    # trajectory features representation
    lstm = seq_lstm()  # for both seek and serve
    # similarity prediction
    classification = seq_classification(num_class)

    # From input (num_trajs) trajectories
    # to concatenated feature representation of each trajectory
    traj_shape = (num_trajs, padding_length, num_traj_feature)
    input = Input(traj_shape)
    # split trajectory
    slices = [
        Slice(
            begin=[0, i, 0, 0], 
            size=[-1, 1, -1, -1], 
            squeeze_dims=[1]
        )(input) 
        for i in range(traj_shape[0])
    ]
    # get each trajectory's representation
    traj_reps = []
    for i, slice in enumerate(slices):
        traj_rep = lstm(slice)  # for seek and serve
        traj_reps.append(traj_rep)
    # merge trajectory representation
    traj_reps = Concatenate(axis=-1)(traj_reps)
    
    # Classification MLP output
    prediction = classification(traj_reps)
    
    # Define model
    model = Model(inputs=input, outputs=prediction)
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=optimizers.Adam(1.5e-5), 
        metrics=['accuracy']
    )
    return model


### Utils ###
def load_model(path):
    """ Load model that contains Slice layer from path """
    model = keras.models.load_model(path, custom_objects={'Slice': Slice})
    return model


class Slice(keras.layers.Layer):
    """ Slice layer for keras model 
        It is used to split multiple trajectaries input into single trajectory
    """
    def __init__(self, begin, size, squeeze_dims=None, **kwargs):
        super(Slice, self).__init__(**kwargs)
        self.begin = begin
        self.size = size
        self.squeeze_dims = squeeze_dims

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'begin': self.begin,
            'size': self.size,
            'squeeze_dims': self.squeeze_dims,
        })
        return config

    def call(self, inputs):
        # Slice tensor
        slice = tf.slice(inputs, self.begin, self.size)
        # Squeeze specified dimensions
        if self.squeeze_dims is not None:
            slice = tf.squeeze(slice, self.squeeze_dims)
        return slice
