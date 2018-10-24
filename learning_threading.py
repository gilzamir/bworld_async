import numpy as np
from agent_threading import SharedModel, AsyncAgent
from collections import deque
import logging
from threading import Lock, Thread
import random

lock = Lock()

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

for i in range(MAX_THREADS):
    agent = AsyncAgent(i, sharedModel)
    agent.make_environment()
    agents.append(agent)

agent.epsilon_decay = ((agent.epsilon - agent.epsilon_min)/1000000)

MAX_EPSODES = 100000000

i = 0
while i < MAX_EPSODES:
    rg = range(MAX_THREADS)
    for i in rg:
        agents[i].thread.start()
    for i in rg:
        agents[i].thread.join()
        agents[i].thread = Thread(target=agents[i].run)
    if (i % 1000 == 0):
        sharedModel.save("model%d" % (i))
    i += 1



