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

REFRESH_MODEL_NUM = 10

def sample(buffer, size):
    indices = random.sample(range(len(buffer)), size)
    result = []
    for i in indices:
        result.append(buffer[i])
    return result

class AsyncAgent:
    def __init__(self, ID, state_size, action_size, learning_rate=0.0025):
        self.ID = ID
        self.samples = deque(maxlen=1000)
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
        self.thread = None
        self.thread_time = 0
        self.epsilon = 1.0
        self.epsilon_min = np.random.normal(0.1, 0.05)
        self.epsilon_decay = np.random.normal(0.9, 0.09)
        self.gamma = np.random.normal(0.9, 0.09)
        self.batch_size = 32
        self.contextual_actions = [0, 1, 2]
        self.RENDER = False
        self.N_RANDOM_STEPS = 12500
        self.NO_OP_STEPS = 30
        self.env = None

    def update_epsilon(self, is_randomic=False):
        if not is_randomic:
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
            action = np.argmax(act_values[0])
            self.update_epsilon(is_randomic)
            return action

    def reset(self):
        pass

    #https://github.com/keras-team/keras/issues/2226
    def get_sample(self, x, mask, label):
        return [ x, mask, # X
               [1], # sample weights
               label, # y
               0 # learning phase in TEST mode
        ]

    def memory_update(self, graph, model, back_model, state, action, reward, next_state, is_done):
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
        

        sample = self.get_sample(state, action_one_hot, target_one_hot)

        self.samples.append(sample)

def run(ID, in_queue, out_queue):
    import tensorflow as tf
    import utils
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.15
    #config.gpu_options.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    agent = AsyncAgent(ID, (84, 84), 3)
    agent.epsilon_decay = ((agent.epsilon - agent.epsilon_min)/250000)
    #print(agent.ID)
    graph = tf.get_default_graph()
    model, back_model = utils.get_model_pair(graph, agent.state_size, agent.skip_frames, agent.action_size, agent.learning_rate)
    MAX_T = 100000000
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

            frame = agent.env.reset()
            if agent.RENDER:
                agent.env.render()

            is_done = False

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            #for _ in range(random.randint(1, agent.NO_OP_STEPS)):
            frame, _, _, _ = agent.env.step(1)

            frame = pre_processing(frame)
            stack_frame = tuple([frame]*agent.skip_frames)
            initial_state = np.stack(stack_frame, axis=2)
            initial_state = np.reshape([initial_state], (1, 84, 84, agent.skip_frames))

            agent.reset()

            LOSS = 0
            score, start_life = 0, 5

            dead = False

            action = 0
            next_state = None
            step  = 0   
            update_counter = 0
            gradient = []
            #old_w = model.get_weights()
            UPDATED = False
            while not is_done:
                if agent.thread_time >= agent.N_RANDOM_STEPS:
                    action = agent.act(graph, model, initial_state)
                else:
                    action = agent.act(graph, model, initial_state, True)
                if step > 2000:
                    print(action)
                frame, reward, is_done, info = agent.env.step(action+1)
                next_frame = pre_processing(frame)
                next_state = np.reshape([next_frame], (1, 84, 84, 1))
                next_state = np.append(next_state, initial_state[:, :, :, :3], axis=3)

                if start_life > info['ale.lives']:
                    reward = -1
                    dead = True
                    start_life = info['ale.lives']

                reward = np.clip(reward, -1.0, 1.0)
  
                score += reward

                if agent.thread_time >= agent.N_RANDOM_STEPS:
                    agent.memory_update(graph, model, back_model, initial_state, action, reward, next_state, dead)
                    if len(agent.samples) >= agent.batch_size:
                        avg_loss = 0.0
                        samples = random.sample(list(agent.samples), agent.batch_size)
                        old_w = model.get_weights()
                        for sample in samples:
                            loss = model.train_on_batch([sample[0], sample[1]], sample[3])
                            avg_loss += loss
                                                
                        if len(gradient)==0:
                            for nw, ow in list(zip(model.get_weights(), old_w)):
                                gradient.append(nw-ow)
                        else:
                            idx = 0
                            for nw, ow in list(zip(model.get_weights(), old_w)):
                                gradient[idx] += nw - ow
                                idx += 1
                                                
                        LOSS += avg_loss/agent.batch_size
                        update_counter += 1
                        UPDATED = True
                        agent.samples.clear()
                if dead:
                    dead = False
                    agent.env.step(1)
                else:
                    initial_state = next_state
                
                if agent.RENDER:
                    agent.env.render()

                agent.thread_time += 1
                step += 1

            if update_counter > 0:
                LOSS = LOSS/update_counter
                
            if update_counter == 0:
                update_counter = 1

            #print("FIM %d"%(agent.ID))

            #for nw, ow in list(zip(model.get_weights(), old_w)):
            #    gradient.append(nw-ow)

            out_queue.put( (gradient, UPDATED, score, LOSS, step, agent.epsilon, agent.ID) )
            in_queue.task_done()
            T += 1
        except Exception as e:
            print("error %"%(s))
            pass
    out_queue.put(None, None, UPDATED, 0.0)
    in_queue.task_done()

 

