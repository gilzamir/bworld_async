import numpy as np
import agent
from collections import deque
import logging
from multiprocessing import Queue, Process,  Manager, Pool
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


state_size = (84, 84)
action_size = 3
learning_rate = 0.0025
skip_frames = 4


def run_learning(input_queue, output_queue, max_threads):
    import tensorflow as tf
    import utils

    state_size = (84, 84)
    action_size = 3
    learning_rate = 0.0025
    skip_frames = 4

    graph = tf.get_default_graph()
    model, back_model = utils.get_model_pair(graph, state_size, skip_frames, action_size, learning_rate)
    utils.front2back(graph, model, back_model)
    weights = model.get_weights()
    back_weights =back_model.get_weights()
    MAX_EPSODES = 10000000
    i = 0
    while True:
        for j in range(max_threads):
            output_queue.put((i, MAX_EPSODES, weights, back_weights))
        
        w, bw, a = input_queue.get()

        for p, new_p  in list(zip(weights, w)):
            p = p + new_p

        i += 1
        print("Epoch: %d"%(i))



def main():
    m = Manager()
    agents = []
    MAX_THREADS = 4


    pool = Pool()
    input_queue = m.Queue()
    output_queue = m.Queue()
    pool.apply_async(run_learning, (input_queue, output_queue, MAX_THREADS))

    for j in range(MAX_THREADS):
        pool.apply_async(agent.run, (j, output_queue, input_queue))
    
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
