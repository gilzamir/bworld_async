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
from multiprocessing import Queue, Process, Pipe
import numpy as np
import time

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

REFRESH_MODEL_NUM = 40000
LEARNING_RATE = 0.0025
ACTION_SIZE = 3
SKIP_FRAMES = 4
STATE_SIZE = (84, 84)
EPSILON_STEPS = [4000000, 4500000, 4000000, 4500000]
RANDOM_STEPS = [30000, 40000, 50000, 60000]
#RANDOM_STEPS = [10000, 10000, 10000, 10000] #TESTE
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
        self.ASYNC_UPDATE = [31, 37, 41, 53]
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
                qout.put( (state, self.ID, self.thread_time, 1) )
                qout.join()
                act_values, ID, TT = qin.get()

            action = np.argmax(act_values[0])
            self.update_epsilon(is_randomic)
            return action

    def reset(self):
        pass

    '''
        req_type pode ser 0 (pi e valor), 1 (pi) ou 2 (valor)
    '''
    def predict(self, nstate, qin, qout, req_type=0): 
        ID = None
        TT = None
        response = None
        while ID != self.ID or TT != self.thread_time:
            qout.put( (nstate, self.ID, self.thread_time, req_type) )
            qout.join()
            response, ID, TT = qin.get()
        return response

    def memory_update(self, qin, qout, state, action, reward, next_state, is_done):
        try:
            pi = self.predict(next_state, qin, qout, 1)
            #---
            sample = (state, action, reward, next_state, is_done, pi)
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

            # this is one of DeepMind's idea.
            # just do nothing at the start of episode to avoid sub-optimal
            #for _ in range(random.randint(1, agent.NO_OP_STEPS)):
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
                    action = agent.act(bqin, bqout, initial_state)
                else:
                    action = agent.act(bqin, bqout, initial_state, True)
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
                    if (agent.thread_time > 0) or (agent.thread_time % agent.ASYNC_UPDATE == 0) or dead: #ASYNC_UPDATE
                        v = agent.predict(next_state, bqin, bqout, 2)
                        R = 0
                        if not dead:
                            R = v[0]
                        for i in range(len(agent.samples)-1, -1, -1):
                            sample = agent.samples[i]
                            R = sample[2] + agent.gamma * R
                            out_uqueue.put( (sample[0], R, sample[-1], v[0], agent.ID, False) )
                        agent.samples.clear()
                        out_uqueue.put( (_, _, _, _, agent.ID, True) )

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
           # logger_debug.debug("THREAD_ID %d T %d SCORE %d STEPS %d TOTAL_STEPS %d  EPSILON %f"%(agent.ID, T, score, step, agent.thread_time, agent.epsilon))
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
