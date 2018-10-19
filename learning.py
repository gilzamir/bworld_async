import numpy as np
from agent import SharedModel, AsyncAgent
from collections import deque
import logging
from multiprocessing import Queue, Process
import random


logger_debug = logging.getLogger(__name__)
logger_debug.setLevel(logging.DEBUG)

handler = logging.FileHandler('train_debug.log')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger_debug.addHandler(handler)
logger_debug.propagate = False

agents = []
MAX_THREADS = 4
sharedModel = SharedModel((84, 84), 3)
sharedModel.front2back()

queue = Queue()

queue.put(sharedModel)

for i in range(MAX_THREADS):
    agent = AsyncAgent(i)
    agents.append(agent)

agent.epsilon_decay = ((agent.epsilon - agent.epsilon_min)/1000000)

MAX_EPSODES = 100000000

i = 0
while i < MAX_EPSODES:
    rg = range(MAX_THREADS)

    for i in rg:
        agents[i].thread = Process(target=agents[i].run, args=(queue, ))
        agents[i].thread.start()

    for i in rg:
        agents[i].thread.join()

    if (i % 1000 == 0):
        sharedModel.save("model%d" % (i))
    i += 1
