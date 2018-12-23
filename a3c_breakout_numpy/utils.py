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

    rms = RMSprop(lr=learning_rate, rho=0.99, epsilon=0.1)
    
    #pmodel.compile(rms, loss={'out1':'categorical_crossentropy', 'out2':'mse'})

    action_pl = K.placeholder(shape=(None, action_size))
    advantages_pl = K.placeholder(shape=(None,))
    discounted_r = K.placeholder(shape=(None,))
    
    weighted_actions = K.max(action_pl * output_actions, axis=1, keepdims=True)
    
    eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(advantages_pl)

    entropy = K.sum(output_actions * K.log(output_actions + 1e-10), axis=1, keepdims=True)
    ploss = 0.001 * entropy + eligibility
    
    closs = K.square(discounted_r + output_value)
        
    total_loss = K.mean(ploss + 0.5 * closs)

    #pupdates = rms.get_updates(pmodel.trainable_weights, [], total_loss)
    #optimizer = K.function([pmodel.input, action_pl, advantages_pl, discounted_r], [ploss, closs], updates=pupdates)
    
    input_tensors = pmodel.inputs + [action_pl] + [advantages_pl] + [discounted_r] + [K.learning_phase()]
    #input_loss = pmodel.inputs + [discounted_r]
    gradients = rms.get_gradients(total_loss, pmodel.trainable_weights)
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    get_loss = K.function(inputs=input_tensors, outputs=[total_loss])
    return (pmodel, get_gradients, get_loss)

def _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate):
    with graph.as_default():
        return _build_model(graph, state_size, skip_frames, action_size, learning_rate)

def get_model_pair(graph, state_size, skip_frames, action_size, learning_rate, threads):
    with graph.as_default():
        pmodel, opt, get_loss = _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate)
        
        pmodel._make_predict_function()
        #pmodel._make_train_function()
      
        tmodels = []
        for _ in range(threads):
            tpmodel, topt, get_loss = _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate)
            tpmodel._make_predict_function()
            tmodels.append( (pmodel, topt, get_loss) )
        return (pmodel, tmodels, opt, get_loss)

