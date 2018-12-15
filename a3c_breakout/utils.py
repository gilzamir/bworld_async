from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras import layers
from keras import Model
import tensorflow as tf
from keras.optimizers import RMSprop

def __keras_imports():
    import keras
    import tensorflow

def _build_model(graph, state_size, skip_frames, action_size, learning_rate):
    __keras_imports()
    ATARI_SHAPE = (state_size[0], state_size[1], skip_frames)  # input image size to model
    ACTION_SIZE = action_size
    # With the functional API we need to define the inputs.
    frames_input = layers.Input(ATARI_SHAPE, name='frames')
    #actions_input = layers.Input((ACTION_SIZE,), name='action_mask')
    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = layers.convolutional.Conv2D(
        16, (8, 8), strides=(4, 4), activation='relu'
    )(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = layers.convolutional.Conv2D(
        32, (4, 4), strides=(2, 2), activation='relu'
    )(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = layers.core.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    x1 = layers.Dense(256, activation='relu')(conv_flattened)
    shared = layers.Dense(256, activation='relu')(x1)
    
    rms_opt = RMSprop(lr=learning_rate, rho=0.99, epsilon=0.1)

    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output_actions = layers.Dense(ACTION_SIZE, activation='softmax', name='out1')(shared)
    output_value = layers.Dense(1, activation='linear', name='out2')(shared)
    pmodel = Model(inputs=[frames_input], outputs=[output_actions])
    vmodel = Model(inputs=[frames_input], outputs=[output_value])
    
    action_pl = K.placeholder(shape=(None, action_size))
    advantages_pl = K.placeholder(shape=(None,))
    discounted_r = K.placeholder(shape=(None,))
    
    weighted_actions = K.sum(action_pl * pmodel.output, axis=1)
    eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(advantages_pl)
    entropy = K.sum(pmodel.output * K.log(pmodel.output + 1e-10), axis=1)
    ploss = 0.001 * entropy - K.sum(eligibility)
    updates = rms_opt.get_updates(pmodel.trainable_weights, [], ploss)
    optimizer = K.function([pmodel.input, action_pl, advantages_pl], [], updates=updates)

    closs = K.mean(K.square(discounted_r - vmodel.output))
    updates2 = rms_opt.get_updates(vmodel.trainable_weights, [], closs)
    optimizer2 = K.function([vmodel.input, discounted_r], [], updates=updates2)

    return (pmodel, vmodel, optimizer, optimizer2)

def _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate):
    with graph.as_default():
        return _build_model(graph, state_size, skip_frames, action_size, learning_rate)

def get_model_pair(graph, state_size, skip_frames, action_size, learning_rate, threads):
    __keras_imports()
    with graph.as_default():
        pmodel, vmodel, opt1, opt2 = _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate)
        pmodel._make_predict_function()
        vmodel._make_predict_function()
        tmodels = []
        for _ in range(threads):
            back_pmodel, back_vmodel, _, _ = _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate)
            back_pmodel._make_predict_function()
            back_vmodel._make_predict_function()
            tmodels.append( (back_pmodel, back_vmodel) )
        return (pmodel, vmodel, tmodels, opt1, opt2)

