from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras import layers
from keras import Model
import tensorflow as tf
from keras.optimizers import RMSprop


def logloss(y_true, y_pred):     #policy loss
    #return -K.sum( K.log(y_true*y_pred + (1-y_true)*(1-y_pred) + 1e-5), axis=-1)
    return 0.01 * K.sum(y_pred * K.log(y_pred + 1e-5) + (1-y_pred) * K.log(1-y_pred + 1e-5))

#loss function for critic output
def sumofsquares(y_true, y_pred):        #critic loss
    return K.sum(K.square(y_pred - y_true), axis=-1)

def _build_model(graph, state_size, skip_frames, action_size, learning_rate):
    import keras
    ATARI_SHAPE = (state_size[0], state_size[1], skip_frames)  # input image size to model
    ACTION_SIZE = action_size
    # With the functional API we need to define the inputs.
    frames_input = layers.Input(ATARI_SHAPE, name='frames')
    #actions_input = layers.Input((ACTION_SIZE,), name='action_mask')
    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    #normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = layers.convolutional.Conv2D(
        16, (8, 8), strides=(4, 4), activation='relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform'
    )(frames_input)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = layers.convolutional.Conv2D(
        32, (4, 4), strides=(2, 2), activation='relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform'
    )(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = layers.core.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    shared = layers.Dense(256, activation='relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output_actions = layers.Dense(ACTION_SIZE, activation='softmax', name='out1', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(shared)
    output_value = layers.Dense(1, name='out2', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(shared)
    model = Model(inputs=[frames_input], outputs=[output_actions, output_value])
    keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
    rms = RMSprop(lr=learning_rate, rho=0.99, epsilon=0.1)

    model.compile(loss = {'out1': logloss, 'out2': sumofsquares}, loss_weights = {'out1': 1., 'out2' : 0.5}, optimizer = rms)

    return model

def _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate):
    with graph.as_default():
        return _build_model(graph, state_size, skip_frames, action_size, learning_rate)

def get_model_pair(graph, state_size, skip_frames, action_size, learning_rate, threads):
    with graph.as_default():
        model = _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate)
        model._make_predict_function()
        model._make_train_function()
        tmodels = []
        for _ in range(threads):
            pvmodel = _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate)
            pvmodel._make_predict_function()
            pvmodel._make_train_function()
            tmodels.append( pvmodel )
        return (model, tmodels)

