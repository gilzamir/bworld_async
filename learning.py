import numpy as np
from agent import SharedModel, AsyncAgent
from collections import deque
import logging
import threading as td
import random

logger_debug = logging.getLogger(__name__)
logger_debug.setLevel(logging.DEBUG)

handler = logging.FileHandler('train_debug.log')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger_debug.addHandler(handler)


sharedModel = SharedModel((84, 84), 3)
agent = AsyncAgent(sharedModel)
sharedModel.front2back()

agent.epsilon_decay = ((agent.epsilon - agent.epsilon_min)/1000000)

MAX_EPSODES = 100000000
agent.make_environment()
i = 0
while i < MAX_EPSODES:
    agent.run(i)
    if (i % 1000 == 0):
        sharedModel.save("model%d" % (i))
    i += 1



