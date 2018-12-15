import gym
import numpy as np
from bagent import DQNAgent as Agent
from collections import deque
import logging
logger_debug = logging.getLogger(__name__)
logger_debug.setLevel(logging.DEBUG)

handler = logging.FileHandler('run_debug.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger_debug.addHandler(handler)

agent = Agent( (105, 80), 17)
agent.epsilon_min = 0.0
agent.epsilon = 0.0
agent._build_model()
agent.load("tmp.h5")

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)
 
def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))

env = gym.make('BreakoutDeterministic-v4')

n = 0
for i in range(1000000):
    frame = env.reset()
    env.render()
    is_done = False
    mem = deque(maxlen=agent.skip_frames)
    batch_size = 12
    batch_size3 = 3*12
    while not is_done:
        mem.append(preprocess(frame))
        action = -1
        if (len(mem)>=agent.skip_frames):
            state = np.reshape(list(mem), [agent.state_size[0], agent.state_size[1], agent.skip_frames])
            state = np.expand_dims(state, axis=0)
            action = agent.act(state)
            frame, reward, is_done, _ = env.step(action)
            mem.append(preprocess(frame))
            next_state = np.reshape(list(mem), [agent.state_size[0], agent.state_size[1], agent.skip_frames])
            next_state = np.expand_dims(next_state, axis=0)
            logger_debug.debug("REWARD TO ACTION %s is %s"%(action, reward))
            logger_debug.debug("MEMORY SIZE (%d, %d, %d)"%(agent.positive_msize(), agent.negative_msize(), agent.neutral_msize()))
        else:
            if action != -1:
                env.step(action)
        logger_debug.debug("SELECTED ACTION %d"%(action))
        env.render()