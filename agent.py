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
import io
import os
import sys
import math
import logging
from collections import deque
import gym
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.transform import rotate
from keras.utils.np_utils import to_categorical
from multiprocessing import Queue, Process, Pipe
import numpy as np
import time

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


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

REFRESH_MODEL_NUM = 500

def sample(buffer, size):
    indices = random.sample(range(len(buffer)), size)
    result = []
    for i in indices:
        result.append(buffer[i])
    return result

class AsyncAgent:
    def __init__(self, ID, state_size, action_size, learning_rate=0.0025):
        self.ID = ID
        self.gradients = deque(maxlen=1000)
        self.skip_frames = 4
        self.learning_rate = learning_rate
        self.initial_decay = 0.0
        self.decay = 0.0
        self.rho = 0.9
        if type(state_size) == tuple:
            self.state_size = state_size
        else:
            self.state_size = (state_size, state_size)
        self.action_size = action_size
        self.mask_actions = np.ones(self.action_size).reshape(1, self.action_size)
        self.thread_id = ID
        self.gradient = None
        self.thread = None
        self.thread_time = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.gamma = 0.99
        self.batch_size = 32
        self.contextual_actions = [0, 1, 2]
        self.RENDER = False
        self.N_RANDOM_STEPS = 12000
        self.NO_OP_STEPS = 30
        self.ASYNC_UPDATE = 100
        self.env = None

    def update_epsilon(self, is_randomic=False):
        if not is_randomic:
            #print("EPS %f  MIN %f DEC %f"%(self.epsilon, self.epsilon_min, self.epsilon_min))
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min

    def act(self, graph, model, state, is_randomic = False):
        action = 0
        p = np.random.rand()
        if is_randomic or p <= self.epsilon:
            self.update_epsilon(is_randomic)
            return np.random.choice(self.contextual_actions)
        else:
            act_values = model.predict([state, self.mask_actions])
            logger_debug.debug("ACTION VALUES %s" % (act_values))
            action = np.argmax(act_values[0])
            #print("MODEL SELECTED ACTION ::::::: %s" % (action))
            self.update_epsilon(is_randomic)
            return action

    def reset(self):
        pass

    #https://github.com/keras-team/keras/issues/2226
    def get_gradient(self, x, mask, label):
        return [ x, mask, # X
               [1], # sample weights
               label, # y
               0 # learning phase in TEST mode
        ]

    def gradient_update(self, graph, model, back_model, state, action, reward, next_state, is_done):
        next_Q_value = back_model.predict([next_state, self.mask_actions])[0]  
        if is_done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(next_Q_value)
        targets = np.zeros(1)
        targets[0] = target
        #---
        
        action_one_hot = get_one_hot([action], self.action_size) 
        target_one_hot = action_one_hot * targets[:, None]
        

        gradient = self.get_gradient(state, action_one_hot, target_one_hot)

        self.gradients.append(gradient)
     

def run(ID, in_queue, out_queue):
    import tensorflow as tf
    import utils
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.15
    #config.gpu_options.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    agent = AsyncAgent(ID, (84, 84), 3)
    agent.epsilon_decay = ((agent.epsilon - agent.epsilon_min)/1000000)
    #print(agent.ID)
    graph = tf.get_default_graph()
    model, back_model = utils.get_model_pair(graph, agent.state_size, agent.skip_frames, agent.action_size, agent.learning_rate)
    MAX_T = 1000000
    T = 0
    while True:
        try:
            params, back_params = in_queue.get()

            if T >= MAX_T:
                break

            with graph.as_default():
                #>print(params)
                model.set_weights(params)
                back_model.set_weights(back_params)
            
            if agent.env == None:
                agent.env =  gym.make('BreakoutDeterministic-v4')

            agent.env.reset()
            frame = agent.env.reset()
            if agent.RENDER:
                agent.env.render()

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            for _ in range(random.randint(1, agent.NO_OP_STEPS)):
                frame, _, _, _ = agent.env.step(1)

            frame = pre_processing(frame)
            stack_frame = tuple([frame]*agent.skip_frames)
            initial_state = np.stack(stack_frame, axis=2)
            initial_state = np.reshape([initial_state], (1, 84, 84, agent.skip_frames))

            agent.reset()

            LOSS = 0
            score, start_life = 0, 5
            is_done = False

            dead = False

            action = 0
            next_state = None
            step  = 0   
            update_counter = 0
            while not is_done:
                UPDATED = False
                if agent.thread_time >= agent.N_RANDOM_STEPS:
                    action = agent.act(graph, model, initial_state)
                else:
                    action = agent.act(graph, model, initial_state, True)

                frame, reward, is_done, info = agent.env.step(action+1)
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
 
                if agent.thread_time >= agent.N_RANDOM_STEPS:
                    agent.gradient_update(graph, model, back_model, initial_state, action, reward, next_state, dead)
                    if len(agent.gradients) >= agent.batch_size:
                        avg_loss = 0.0
                        gradients = random.sample(list(agent.gradients), agent.batch_size)
                        for gradient in gradients:
                            loss = model.train_on_batch([gradient[0], gradient[1]], gradient[3])
                            avg_loss += loss
                        LOSS += avg_loss/agent.batch_size
                        update_counter += 1
                        if (agent.thread_time > 0 and agent.thread_time % agent.ASYNC_UPDATE == 0):
                            agent.gradients.clear()
                            UPDATED = True
                if dead:
                    dead = False
                else:
                    initial_state = next_state
                if agent.RENDER:
                    agent.env.render()

                agent.thread_time += 1
                step += 1
                if update_counter > 0:
                    LOSS = LOSS/update_counter
                    
                    logger_debug.debug("SCORE ON EPISODE %d IS %d. EPSILON IS %f. STEPS IS %d. GSTEPS is %d. AVG_LOSS %f" % (
                        T, score, agent.epsilon, step, agent.thread_time, LOSS))
                    
                    #print("SCORE ON EPISODE %d IS %d. EPSILON IS %f. STEPS IS %d. GSTEPS IS %d. AVG_LOSS %f" % (
                    #    T, score, agent.epsilon, step, agent.thread_time, LOSS))
                #else:
                    #print("STEPS IS %d. GSTEPS IS %d."%(step, agent.thread_time))

            if update_counter == 0:
                update_counter = 1

            #print("FIM %d"%(agent.ID))
            out_queue.put( (model.get_weights(), back_model.get_weights(), UPDATED, score, LOSS, step, agent.epsilon) )
            in_queue.task_done()
            T += 1
        except Exception as e:
            print("error %"%(s))
            pass
    out_queue.put(None, None, UPDATED, 0.0)
    in_queue.task_done()

 

