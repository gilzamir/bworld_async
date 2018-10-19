# -*- coding: utf-8 -*-
import random
import numpy as np
from net import act, percept
import net
import threading as td
import numpy as np
import actions
from PIL import Image
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras import layers
from keras import Model
import tensorflow as tf
import io
import os
import sys
import math
import logging
from keras.optimizers import RMSprop
from collections import deque
import gym
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.transform import rotate
from keras.utils.np_utils import to_categorical
from threading import Thread

def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

logger_debug = logging.getLogger(__name__)
logger_debug.setLevel(logging.DEBUG)

handler = logging.FileHandler('agent_debug.log')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger_debug.addHandler(handler)
logger_debug.propagate = False


'''
def apply_updates(shared, tensors, gradients, iterations=K.variable(0, dtype='int64', name='iterations')):
    grads =  gradients
    params = tensors
    accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
    weights = accumulators
    updates = [K.update_add(iterations, 1)]

    lr = shared.model.optimizer.lr
    if shared.initial_decay > 0:
        lr = lr * (1. / (1. + shared.decay * K.cast(shared.iterations,
                                                  K.dtype(shared.decay))))

    for p, g, a in zip(params, grads, accumulators):
        # update accumulator
        new_a = shared.rho * a + (1. - shared.rho) * K.square(g)
        #shared.updates.append(K.update(a, new_a))
        new_p = p - lr * g / (K.sqrt(new_a) + K.epsilon())

        # Apply constraints.
        if getattr(p, 'constraint', None) is not None:
            new_p = p.constraint(new_p)

        K.update(p, new_p)
'''

def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def sample(buffer, size):
    indices = random.sample(range(len(buffer)), size)
    result = []
    for i in indices:
        result.append(buffer[i])
    return result


class SharedModel:
    def __init__(self, state_size, action_size):
        self.model = None
        self.shared_time = 0
        self.gradients = deque(maxlen=1000)
        self.back_model = None
        self.skip_frames = 4
        self.learning_rate = 0.0025
        self.graph = tf.get_default_graph()
        if type(state_size) == tuple:
            self.state_size = state_size
        else:
            self.state_size = (state_size, state_size)
        self.action_size = action_size
        self.mask_actions = np.ones(self.action_size).reshape(1, self.action_size)
        self.get_model_pair()
        with self.graph.as_default():
            outputTensor = self.model.output

            #listOfVariableOfTensors = self.shared.model.trainable_weights

            #gradients = K.gradients(outputTensor, listOfVariableOfTensors)

            self.weights = [] 
            self.layer_index = {}
            idx = 0
            for tensor in self.model.trainable_weights: 
                name = tensor.name.split('/')[0]
                if self.model.get_layer(name).trainable:
                    self.weights.append(tensor)
                    self.layer_index[name] = idx
                    idx += 1

            self.optimizer = self.model.optimizer

            #self.back_weights = [tensor for tensor in self.model.trainable_weights if self.model.get_layer(tensor.name[:-2]).trainable]
            #self.back_optimizer = self.model.optimizer


            gradients = self.optimizer.get_gradients(self.model.total_loss, self.weights)

       
            input_tensors = [self.model.inputs[0], # input data
                    self.model.inputs[1],
                    self.model.sample_weights[0], # how much to weight each sample by
                     self.model.targets[0], # labels
                    K.learning_phase(), # train or test mode
            ]

            self.gradient_update = K.function(inputs=input_tensors, outputs=gradients)

    #https://github.com/keras-team/keras/issues/2226
    def get_gradient(self, x, mask, label):
        return [ x, mask, # X
                   [1], # sample weights
                   label, # y
                   0 # learning phase in TEST mode
            ]

    def get_model_pair(self):
        with self.graph.as_default():
            if not self.model:
                self.model = self._build_model()
                self.model._make_predict_function()
                self.model._make_test_function()
                self.model._make_train_function()
                self.back_model = self._build_model()
                self.back_model._make_predict_function()
                self.back_model._make_test_function()
                self.back_model._make_train_function()
            return (self.model, self.back_model)

    def _build_model(self):
        with self.graph.as_default():
            ATARI_SHAPE = (self.state_size[0], self.state_size[1], self.skip_frames)  # input image size to model
            ACTION_SIZE = self.action_size
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
            model.summary()
            optimizer = RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01)
            # model.compile(optimizer, loss='mse')
            # to changed model weights more slowly, uses MSE for low values and MAE(Mean Absolute Error) for large values
            model.compile(optimizer, loss=huber_loss)
            return model

    def front2back(self):
        with self.graph.as_default():
            self.back_model.set_weights(self.model.get_weights())

    def back2front(self):
        with self.graph.as_default():
            self.model.set_weights(self.back_model.get_weights())

    def load(self, name):
        with self.graph.as_default():
            self.model.load_weights(name)

    def save(self, name):
        with self.graph.as_default():
            self.model.save_weights(name)

    def load_back(self, name):
        with self.graph.as_default():
            self.back_model.load_weights(name)

    def save_back(self, name):
        with self.graph.as_default():
            self.back_model.save_weights(name)

class AsyncAgent:
    def __init__(self, ID, shared_model):
        self.thread_id = ID
        self.shared = shared_model
        self.gradient = None
        self.thread = Thread(target=self.run)
        self.thread_time = 0
        self.update_schedule = 100
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.gamma = 0.99
        self.contextual_actions = [0, 1, 2]
        self.locked = False
        self.RENDER = False
        self.REFRESH_MODEL_NUM = 10000
        self.N_RANDOM_STEPS = 50000
        self.NO_OP_STEPS = 30
        self.ASYNC_UPDATE = 100

    def update_epsilon(self, is_randomic=False):
        if not is_randomic:
            #print("EPS %f  MIN %f DEC %f"%(self.epsilon, self.epsilon_min, self.epsilon_min))
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min

    def act(self, state, is_randomic = False):
        action = 0
        p = np.random.rand()
        if is_randomic or p <= self.epsilon:
            self.update_epsilon(is_randomic)
            return np.random.choice(self.contextual_actions)
        else:
            act_values = self.shared.model.predict([state, self.shared.mask_actions])
            logger_debug.debug("ACTION VALUES %s" % (act_values))
            action = np.argmax(act_values[0])
            #print("MODEL SELECTED ACTION ::::::: %s" % (action))
            self.update_epsilon(is_randomic)
            return action

    def reset(self):
        pass


    def gradient_update(self, state, action, reward, next_state, is_done):
        with self.shared.graph.as_default():
            next_Q_value = self.shared.back_model.predict([next_state, self.shared.mask_actions])[0]        
            if is_done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(next_Q_value)
            targets = np.zeros(1)
            targets[0] = target

            action_one_hot = get_one_hot([action], self.shared.action_size) 
            target_one_hot = action_one_hot * targets[:, None]
            
            gradient = (state, action_one_hot, target_one_hot)
            
            self.shared.gradients.append(gradient)

    def make_environment(self):
        self.env =  gym.make('BreakoutDeterministic-v4')        

    def run(self):
        with self.shared.graph.as_default():
            self.locked = True
            self.env.reset()
            frame = self.env.reset()  

            if self.RENDER:
                self.env.render()

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            for _ in range(random.randint(1, self.NO_OP_STEPS)):
                frame, _, _, _ = self.env.step(1)

            frame = pre_processing(frame)
            stack_frame = tuple([frame]*self.shared.skip_frames)
            initial_state = np.stack(stack_frame, axis=2)
            initial_state = np.reshape([initial_state], (1, 84, 84, self.shared.skip_frames))

            self.reset()

            LOSS = 0
            score, start_life = 0, 5
            is_done = False

            dead = False

            action = 0
            next_state = None
            step  = 0   
            #print("------------------------------------------------------------------------------%s"%(is_done))
            while not is_done:
                if self.shared.shared_time >= self.N_RANDOM_STEPS:
                    action = self.act(initial_state)
                else:
                    action = self.act(initial_state, True)

                frame, reward, is_done, info = self.env.step(action+1)
                #print(is_done)
                next_frame = pre_processing(frame)
                next_state = np.reshape([next_frame], (1, 84, 84, 1))
                next_state = np.append(next_state, initial_state[:, :, :, :3], axis=3)

                if start_life > info['ale.lives']:
                    reward = -1
                    dead = True
                    start_life = info['ale.lives']
                
                reward = np.clip(reward, -1.0, 1.0)
                
                score += reward
                
                logger_debug.debug("REWARD TO ACTION %d is %d" % (action, reward))

                opt = self.shared.model.optimizer

                if self.shared.shared_time >= self.N_RANDOM_STEPS:
                    self.gradient_update(initial_state, action, reward, next_state, dead)
                    if self.shared.shared_time % self.REFRESH_MODEL_NUM == 0:
                        self.shared.back2front()

                    if (self.thread_time % self.ASYNC_UPDATE) == 0:
                        avg_loss = 0.0
                        tensors = self.shared.model.trainable_weights
                        for gradient in list(self.shared.gradients):
                            t_tensors = []
                            loss = self.shared.model.train_on_batch([gradient[0], gradient[1]], gradient[2])
                            avg_loss += loss
                        LOSS = avg_loss/len(self.shared.gradients)
                        print("CURRENT LOSS ON THREAD %d === %f"%(self.thread_id, LOSS))
                        self.shared.gradients.clear()
                if dead:
                    dead = False
                else:
                    initial_state = next_state
                if self.RENDER:
                    self.env.render()
            
                self.shared.shared_time += 1
                self.thread_time += 1
                step += 1
            logger_debug.debug("SCORE ON EPISODE %d IS %d. EPSILON IS %f. STEPS IS %d. GSTEPS is %d." % (
                self.thread_time, score, self.epsilon, step, self.shared.shared_time))
            print("SCORE ON EPISODE %d IS %d. EPSILON IS %f. STEPS IS %d. GSTEPS IS %d." % (
                self.thread_time, score, self.epsilon, step, self.shared.shared_time))
            
            self.locked = False

