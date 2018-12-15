# -*- coding: utf-8 -*-
import random
from net import act, percept
import net
import threading as td
import numpy as np
import actions
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
#from keras.utils.np_utils import to_categorical
from multiprocessing import Queue, Process, Pipe
import numpy as np
import time

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

def down_sampling(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (8, 8), mode='constant') * 255)
    return processed_observe   

def same(a, b):
    return a[0][0] == b[0][0] and a[1][0] == b[1][0] and a[2][0] == b[2][0] and a[3][0] == b[3][0]
            and a[4][0] == b[5][0] and a[6][0] and a[7][0]
            and a[0][1] == b[0][1] and a[1][1] == b[1][1] and a[2][1] == b[2][1] and a[3][1] == b[3][1]
            and a[4][1] == b[5][1] and a[6][1] and a[7][1]
            and a[0][2] == b[0][2] and a[1][2] == b[1][2] and a[2][2] == b[2][2] and a[3][2] == b[3][2]
            and a[4][2] == b[5][2] and a[6][2] and a[7][2]
            and a[0][3] == b[0][3] and a[1][3] == b[1][3] and a[2][3] == b[2][3] and a[3][3] == b[3][3]
            and a[4][3] == b[5][3] and a[6][3] and a[7][3]
            and a[0][4] == b[0][4] and a[1][4] == b[1][4] and a[2][4] == b[2][4] and a[3][4] == b[3][4]
            and a[4][4] == b[5][4] and a[6][4] and a[7][4]
            and a[0][5] == b[0][5] and a[1][5] == b[1][5] and a[2][5] == b[2][5] and a[3][5] == b[3][5]
            and a[4][5] == b[5][5] and a[6][5] and a[7][5]
            and a[0][6] == b[0][6] and a[1][6] == b[1][6] and a[2][6] == b[2][6] and a[3][6] == b[3][6]
            and a[4][6] == b[5][6] and a[6][6] and a[7][6]
            and a[0][7] == b[0][7] and a[1][7] == b[1][7] and a[2][7] == b[2][7] and a[3][7] == b[3][7]
            and a[4][7] == b[5][7] and a[6][7] and a[7][7]

REFRESH_MODEL_NUM = 40000
LEARNING_RATE = 0.0025
ACTION_SIZE = 3
SKIP_FRAMES = 4
STATE_SIZE = (84, 84)
EPSILON_STEPS = [4000000, 4500000, 4000000, 4500000]
RANDOM_STEPS = [50000, 50000, 50000, 50000]
GRADIENT_BATCH = 32
ESPSILON_MINS = [0.1, 0.1, 0.05, 0.001]

def sample(buffer, size):
    indices = random.sample(range(len(buffer)), size)
    result = []
    for i in indices:
        result.append(buffer[i])
    return result

class AsyncAgent:
    def __init__(self, ID, state_size, action_size):
        global SKIP_FRAMES
        self.skip_frames = SKIP_FRAMES
        self.ID = ID
        self.samples = deque(maxlen=1000)
        if type(state_size) == tuple:
            self.state_size = state_size
        else:
            self.state_size = (state_size, state_size)
        self.action_size = action_size
        self.mask_actions = np.ones(self.action_size).reshape(1, self.action_size)
        self.thread_id = ID
        self.thread_time = 0
        self.epsilon = 1.0
        self.epsilon_min = ESPSILON_MINS[ID]
        self.epsilon_decay = None
        self.epsilon_steps = EPSILON_STEPS[ID]
        self.gamma = 0.99
        self.batch_size = 1
        self.ASYNC_UPDATE = 32
        self.contextual_actions = [0, 1, 2]
        self.RENDER = False
        self.N_RANDOM_STEPS = RANDOM_STEPS[ID]
        self.env = None

    def update_epsilon(self, is_randomic=False):
        if not is_randomic:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min

    def act(self, qin, qout, state, is_randomic = False):
        action = 0
        p = np.random.rand()
        if is_randomic or p <= self.epsilon:
            self.update_epsilon(is_randomic)
            return np.random.choice(self.contextual_actions)
        else:
            ID = None
            TT = None
            while ID != self.ID or TT != self.thread_time:
                qout.put( (state, self.mask_actions, self.ID, self.thread_time) )
                qout.join()
                act_values, ID, TT = qin.get()

            action = np.argmax(act_values[0])
            self.update_epsilon(is_randomic)
            return action

    def reset(self):
        pass

    #https://github.com/keras-team/keras/issues/2226
    def get_sample(self, x, mask, label):
        return [x, mask, # X
               [1], # sample weights
               label, # y
               0 # learning phase in TEST mode
        ]

    def memory_update(self, qin, qout, state, action, reward, next_state, is_done):
        try:

            target = reward
            if not is_done:
                ID = None
                TT = None
                next_Q_value = None
                while ID != self.ID or TT != self.thread_time:
                    qout.put( (next_state, self.mask_actions, self.ID, self.thread_time) )
                    qout.join()
                    next_Q_value, ID, TT = qin.get()
                target = reward + self.gamma * np.amax(next_Q_value[0])
            
            targets = np.zeros(1)
            targets[0] = target
            #---
            
            action_one_hot = get_one_hot([action], self.action_size) 
            target_one_hot = action_one_hot * targets[:, None]
            sample = self.get_sample(state, action_one_hot[0], target_one_hot[0])

            self.samples.append(sample)
        except ValueError as ve:
            print('Error nao esperado em agent.memory_update: %s'%(ve))
        except Exception as e:
            print("Error nao esperado em agent.memory_update: %s"%(e))
        except:
            print("Erro nao esperado em agent.memory_update: %s"%(sys.exc_info()[0]))
            raise


def run(ID, qin, qout, bqin, bqout, out_uqueue):
    agent = AsyncAgent(ID, (84, 84), 3)
    if agent.epsilon_decay == None:
        agent.epsilon_decay = ((agent.epsilon - agent.epsilon_min)/agent.epsilon_steps)
    print(agent.ID)
    MAX_T = 100000000
    T = 0
    while True:
        try:
            if T >= MAX_T:
                break

            if agent.env == None:
                agent.env =  gym.make('BreakoutDeterministic-v4')

            frame = agent.env.reset()
            if agent.RENDER:
                agent.env.render()

            is_done = False

            for _ in range(random.randint(1,30)):
                frame, _, _, _ = agent.env.step(1)

            frame = pre_processing(frame)
            stack_frame = tuple([frame]*agent.skip_frames)
            initial_state = np.stack(stack_frame, axis=2)
            initial_state = np.reshape([initial_state], (1, 84, 84, agent.skip_frames))

            agent.reset()

            score, start_life = 0, 5

            dead = False

            action = 0
            next_state = None
            step  = 0

            while not is_done:
                if agent.thread_time >= agent.N_RANDOM_STEPS:
                    action = agent.act(qin, qout, initial_state)
                else:
                    action = agent.act(qin, qout, initial_state, True)
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
                    agent.memory_update(bqin, bqout, initial_state, action, reward, next_state, dead)
                    if len(agent.samples) >= agent.batch_size: #ASYNC_UPDATE
                        samples = random.sample(agent.samples, agent.batch_size)
                        for sample in samples:
                            out_uqueue.put( ([sample[0], sample[1]], sample[3], agent.ID, False) ) 
                        agent.samples.clear()

                if (agent.thread_time > 0) and ( (agent.thread_time % agent.ASYNC_UPDATE == 0) ):
                    out_uqueue.put( (_, _, agent.ID, True) )
                
                if dead:
                    dead = False
                    agent.env.step(1)
                else:
                    initial_state = next_state
                
                if agent.RENDER:
                    agent.env.render()

                agent.thread_time += 1
                step += 1
            print("THREAD_ID %d T %d SCORE %d STEPS %d TOTAL_STEPS %d  EPSILON %f"%(agent.ID, T, score, step, agent.thread_time, agent.epsilon))
            T += 1
        except ValueError as ve:
            print("Erro nao esperado em agent.run")
            print(ve)
        except Exception as e:
            print("Erro nao esperado em agent.run")
            print("error %s"%(e))
        except:
            print("Erro nao esperado em agent.run: %s"%(sys.exc_info()[0]))
            raise

 

