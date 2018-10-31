import numpy as np
import agent
from collections import deque
import logging
from multiprocessing import Queue, Process,  Manager, Pool
import random
import queue
import time


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
    try:
        import tensorflow as tf
        import utils
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        #config.gpu_options.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        state_size = (84, 84)
        action_size = 3
        learning_rate = 0.0025
        skip_frames = 4

        graph = tf.get_default_graph()
        model, back_model = utils.get_model_pair(graph, state_size, skip_frames, action_size, learning_rate)
        utils.front2back(graph, model, back_model)
        weights = model.get_weights()
        back_weights =back_model.get_weights()
        T = 0
        while True:
            if T > 0 and T % agent.REFRESH_MODEL_NUM == 0:
                utils.front2back(graph, model, back_model)

            for j in range(max_threads):
                output_queue.put((weights, back_weights))
            output_queue.join()

            avg_loss = 0.0
            avg_score = 0.0
            avg_steps = 0.0
            updates = 0
            epsilon = [0]*max_threads 
            it = 0 
            while not input_queue.empty():
                w, bw, updated, score, loss, steps, eps = input_queue.get()
                avg_score += score
                avg_steps += steps
                if (updated):
                    avg_loss += loss
                    epsilon[c] = eps
                    params = list(zip(weights, w))
                    for p, new_p  in params:
                        p = p + new_p * 1.0/max_threads
                    updates += 1
                it += 1
            if it > 0:
                avg_steps = avg_steps/it
                avg_score = avg_score/it
                if updates > 0:
                    avg_loss = avg_loss/updates
                    print("Global time: %d,  Avg Steps: %f, Avg Score: %f, Avg Loss: %f, Epsilon: %s"%(T, avg_steps, avg_score, avg_loss, epsilon))
                else:
                    print("Global time: %d, Avg Steps: %f, Avg Score: %f"%(T, avg_steps, avg_score))
            T += 1
    except Exception as e:
        print(e)
def main():
    m = Manager()
    agents = []
    MAX_THREADS = 4


    pool = Pool()
    input_queue = m.Queue()
    output_queue = m.JoinableQueue()
    pool.apply_async(run_learning, (input_queue, output_queue, MAX_THREADS))

    for j in range(MAX_THREADS):
        pool.apply_async(agent.run, (j, output_queue, input_queue))
    
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
