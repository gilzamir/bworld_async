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

def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

_env_test = gym.make('CartPole-v0')
LEARNING_RATE = 0.005
ACTION_SIZE = _env_test.action_space.n
SKIP_FRAMES = 1
STATE_SIZE = _env_test.observation_space.shape[0]

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
        self.gamma = 0.99
        self.ASYNC_UPDATE = 32
        self.RENDER = False
        self.env = None
        self.NO_OP_STEPS = 0

    def act(self, qin, qout, state):        
        ID = None
        TT = None
        qout.put( (state, self.ID, self.thread_time, 1) )
        qout.join()
        act_values, ID, TT = qin.get()

        if ID != self.ID or TT != self.thread_time:
            print('INCONSISTENCE DETECTED ON agent.act: predict response returns inconsistent data!')

        return np.random.choice(np.arange(ACTION_SIZE), 1, p=act_values[0])[0]

    def reset(self):
        pass

    '''
        req_type pode ser 0 (pi e valor), 1 (pi) ou 2 (valor)
    '''
    def predict(self, nstate, qin, qout, req_type=0): 
        ID = None
        TT = None
        response = None
        qout.put( (nstate, self.ID, self.thread_time, req_type) )
        qout.join()
        response, ID, TT = qin.get()
        if not self.thread_id == TT and not self.ID == ID:
                print("ERRO : DADOS INCOERENTES EM Agent.predict_____________________________________________________________")
        return response

def run(ID, qin, qout, bqin, bqout, out_uqueue):
    agent = AsyncAgent(ID, STATE_SIZE, ACTION_SIZE)

    print(agent.ID)
    MAX_T = 1000
    T = 0
    start_time = time.time()

    time.sleep(5 * (ID))
    while True:
        try:
            if T >= MAX_T:
                break

            if agent.env == None:
                agent.env =  gym.make('CartPole-v1')

            obs = agent.env.reset()
            if agent.RENDER:
                agent.env.render()

            is_done = False

            initial_state = obs
            initial_state = np.reshape([initial_state], (1, STATE_SIZE))
            agent.reset()
            score = 0
            action = 0
            next_state = None
            step  = 0
            avg_value = 0.0
            count_values = 0
            while not is_done:
                action = agent.act(bqin, bqout, initial_state)
                
                obs, reward, is_done, _ = agent.env.step(action)
                next_state = np.reshape([obs], (1, STATE_SIZE))
                
                score += reward
                
                sample = (initial_state, action, reward, next_state)
                agent.samples.append(sample)
                
                if len(agent.samples) >= agent.ASYNC_UPDATE or is_done: #ASYNC_UPDATE
                    v = agent.predict(next_state, bqin, bqout, 2)[0]
                    avg_value += v[0]
                    count_values += 1
                    R = 0.0
                    if not is_done:
                        R = v[0]
                    for i in reversed(range(0, len(agent.samples))):
                        sstate, saction, sreward, _ = agent.samples[i]
                        R = sreward + agent.gamma * R

                        out_uqueue.put( (sstate, saction, R, v[0], agent.ID, False) )
                    agent.samples.clear()

                    out_uqueue.put( (_, _, _, _, agent.ID, True) )
                
                initial_state = next_state
                
                if agent.RENDER:
                    agent.env.render()

                agent.thread_time += 1
                step += 1
            print("THREAD_ID %d T %d SCORE %d STEPS %d TOTAL_STEPS %d AVG_VALUE %f ELAPSED TIME %d segs"%(agent.ID, T, score, step, agent.thread_time, avg_value/count_values, time.time()-start_time))
           # logger_debug.debug("THREAD_ID %d T %d SCORE %d STEPS %d TOTAL_STEPS %d  EPSILON %f"%(agent.ID, T, score, step, agent.thread_time, agent.epsilon))
            T += 1
        except ValueError as ve:
            print("Erro nao esperado em agent.run")
            print(ve)
            raise
        except Exception as e:
            print("Erro nao esperado em agent.run")
            print("error %s"%(e))
            raise
        except:
            print("Erro nao esperado em agent.run: %s"%(sys.exc_info()[0]))
            raise
