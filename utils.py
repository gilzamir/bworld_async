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

def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def front2back(graph, model, back_model):
    with graph.as_default():
        back_model.set_weights(model.get_weights())

def back2front(graph, model, back_model):
    with graph.as_default():
        model.set_weights(back_model.get_weights())

def load(graph, model, back_model):
    with graph.as_default():
        model.load_weights(name)

def save(graph, name, model):
    with graph.as_default():
        model.save_weights(name)

def load_back(graph, name, model):
    with graph.as_default():
        back_model.load_weights(name)

def save_back(graph, name, model):
    with graph.as_default():
        back_model.save_weights(name)


def _build_model(graph, state_size, skip_frames, action_size, learning_rate):
    __keras_imports()
    ATARI_SHAPE = (state_size[0], state_size[1], skip_frames)  # input image size to model
    ACTION_SIZE = action_size
    # With the functional API we need to define the inputs.
    frames_input = layers.Input(ATARI_SHAPE, name='frames')
    actions_input = layers.Input((ACTION_SIZE,), name='action_mask')

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
    hidden = layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = layers.Dense(ACTION_SIZE)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = layers.Multiply(name='QValue')([output, actions_input])

    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
   # model.summary()
    optimizer = RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01)
    # model.compile(optimizer, loss='mse')
    # to changed model weights more slowly, uses MSE for low values and MAE(Mean Absolute Error) for large values
    model.compile(optimizer, loss=huber_loss)
    return model

def _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate):
    with graph.as_default():
        return _build_model(graph, state_size, skip_frames, action_size, learning_rate)

def get_model_pair(graph, state_size, skip_frames, action_size, learning_rate):
    __keras_imports()
    with graph.as_default():
        model = _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate)
        model._make_predict_function()
        #model._make_test_function()
        model._make_train_function()
        
        back_model = _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate)
        back_model._make_predict_function()
        #back_model._make_test_function()
        back_model._make_train_function()
        
        backup_model = _build_model_from_graph(graph, state_size, skip_frames, action_size, learning_rate)
        backup_model._make_predict_function()
        #backup_model._make_test_function()
        backup_model._make_train_function()
        return (model, back_model, backup_model)

def make_update_function(graph, model):
    with graph.as_default():
        outputTensor = model.output

        weights = [] 
        layer_index = {}
        idx = 0

        for tensor in model.trainable_weights: 
            name = tensor.name.split('/')[0]
            if model.get_layer(name).trainable:
                weights.append(tensor)
                layer_index[name] = idx
                idx += 1

        optimizer = model.optimizer

        gradients = optimizer.get_gradients(model.total_loss, weights)

   
        input_tensors = [model.inputs[0], # input data
                model.inputs[1],
                model.sample_weights[0], # how much to weight each sample by
                model.targets[0], # labels
                K.learning_phase(), # train or test mode
        ]

        gradient_update = K.function(inputs=input_tensors, outputs=gradients)
        return gradient_update
