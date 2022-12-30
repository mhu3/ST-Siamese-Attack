from keras.layers import Input, Dense, Bidirectional, LSTM, Lambda, Conv1D, GlobalMaxPooling1D, \
                         Concatenate, Reshape, Dropout, Flatten, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras import optimizers
from keras import backend as K


def seq_lstm():
    '''trajectory features representation lstm'''
    lstm = Sequential()
    lstm.add(LSTM(128, return_sequences=True))
    lstm.add(LSTM(64))
    # lstm.add(Bidirectional(LSTM(32, return_sequences=True)))
    # lstm.add(Bidirectional(LSTM(32)))
    lstm.add(Dense(32, activation='relu'))
    return lstm

def seq_profile():
    '''model profile features representation mlp'''
    profile = Sequential()
    profile.add(Dense(64, activation='relu'))
    profile.add(Dense(32, activation='relu'))
    profile.add(Dense(8, activation='relu'))
    return profile

def seq_similarity():
    '''model similarity prediction mlp'''
    prediction = Sequential()
    prediction.add(Dense(64, activation='relu'))
    # prediction.add(BatchNormalization())
    prediction.add(Dense(32, activation='relu'))
    # prediction.add(BatchNormalization())
    prediction.add(Dense(1))
    prediction.add(Activation('sigmoid'))
    return prediction


def build_lstm_siamese(num_trajs, padding_length, num_traj_feature, traj_type):
    '''Build siamese lstm model'''

    # Define layers
    # trajectory features representation
    if traj_type == 'all':
        lstm1 = seq_lstm() # for seek
        lstm2 = seq_lstm() # for serve
    else:
        lstm = seq_lstm() # for seek or serve
    # similarity prediction
    similarity = seq_similarity()

    # Pair input to feature representation
    traj_shape = (num_trajs*2, padding_length, num_traj_feature)
    input = Input(traj_shape)
    # split trajectory and get each trajectory's representation
    traj_reps = []
    for i in range(traj_shape[0]):
        if traj_type == 'all':
            if i < num_trajs:
                traj_rep = lstm1(input[:, i, :, :]) # for seek
            else:
                traj_rep = lstm2(input[:, i, :, :]) # for serve
        else:
            traj_rep = lstm(input[:, i, :, :]) # for seek or serve
        traj_reps.append(traj_rep)
    # merge trajectory representation
    traj_reps = Concatenate(axis=-1)(traj_reps)
    
    # Similarity MLP output
    prediction = similarity(traj_reps)
    
    # Define model
    model = Model(inputs=input, outputs=prediction)
    model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(0.0001), metrics=['accuracy'])
    return model
