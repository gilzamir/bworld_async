from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras import layers
from keras import Model
import tensorflow as tf
from keras.optimizers import RMSprop

def _build_model(graph, state_size, skip_frames, action_size, learning_rate):
    import keras
    ATARI_SHAPE = (state_size[0], state_size[1], skip_frames)  # input image size to model
    ACTION_SIZE = action_size
    # With the functional API we need to define the inputs.
    frames_input = layers.Input(ATARI_SHAPE, name='frames')
    #actions_input = layers.Input((ACTION_SIZE,), name='action_mask')
    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = layers.convolutional.Conv2D(
        16, (8, 8), strides=(4, 4), activation='relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform'
    )(normalized)
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
    output_value = layers.Dense(1, name='out2', activation='linear', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(shared)
    keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
    pmodel = Model(inputs=[frames_input], outputs=[output_actions, output_value])

    rms = RMSprop(lr=learning_rate, rho=0.99, epsilon=0.1, clipnorm=40.0)
    
    action_pl = K.placeholder(shape=(None, action_size))
    advantages_pl = K.placeholder(shape=(None,))
    discounted_r = K.placeholder(shape=(None,))

    weighted_actions = K.sum(action_pl * output_actions, axis=1)
    
    eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(advantages_pl)

    entropy = K.sum(output_actions * K.log(output_actions + 1e-10), axis=1)
    ploss = 0.001 * entropy - K.sum(eligibility)
    
    closs = K.mean(K.square(discounted_r - output_value))
        
    total_loss = ploss + 0.5 * closs

    updates = rms.get_updates(pmodel.trainable_weights, [], total_loss)
    optimizer = K.function([pmodel.input, action_pl, advantages_pl, discounted_r], [], updates=updates)

    return (pmodel, optimizer)

def _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate):
    with graph.as_default():
        return _build_model(graph, state_size, skip_frames, action_size, learning_rate)

def get_model_pair(graph, state_size, skip_frames, action_size, learning_rate, threads):
    with graph.as_default():
        pmodel, opt = _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate)
        
        pmodel._make_predict_function()
      
        tmodels = []
        for _ in range(threads):
            pmodel, _ = _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate)
            pmodel._make_predict_function()
            tmodels.append(pmodel)
        return (pmodel, tmodels, opt)

