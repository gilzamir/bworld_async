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
    INPUT_SHAPE = (state_size,)  # input image size to model
    ACTION_SIZE = action_size
    # With the functional API we need to define the inputs.
    LInput = layers.Input(INPUT_SHAPE, name='inputs')

    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    h1 = layers.Dense(64, activation='relu')(LInput)
    h2 = layers.Dense(128, activation='relu')(h1)
    rms_opt = RMSprop(lr=learning_rate, epsilon=0.1, rho=0.99)

    x1 = layers.Dense(128, activation='relu')(h2)
    x2 = layers.Dense(128, activation='relu')(h2)

    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output_actions = layers.Dense(ACTION_SIZE, activation='softmax', name='out1')(x1)
    
    output_value = layers.Dense(1, activation='linear', name='out2')(x2)
    
    pmodel = Model(inputs=[LInput], outputs=[output_actions])
    vmodel = Model(inputs=[LInput], outputs=[output_value])
    
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
