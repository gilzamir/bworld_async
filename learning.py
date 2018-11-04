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
        T = 0
        while True:
            if T > 0 and T % agent.REFRESH_MODEL_NUM == 0:
                utils.front2back(graph, model, back_model)

            for _ in range(max_threads):
                output_queue.put((model.get_weights(), back_model.get_weights()))
            output_queue.join()

            weights = model.get_weights()
            avg_loss = 0.0
            avg_score = 0.0
            avg_steps = 0.0
            updates = 0
            epsilon = [0]*max_threads 
            it = 0 
            alfa = 1.0/max_threads
            while not input_queue.empty():
                gradient, updated, score, loss, steps, eps, thread_id = input_queue.get()
                avg_score += score
                avg_steps += steps
                if (updated):
                    avg_loss += loss
                    epsilon[it] = eps
                    params = list(zip(weights, gradient))
                    idx = 0
                    for p, new_p  in params:
                        p = p + new_p * alfa
                        weights[idx] = p
                        idx += 1
                    model.set_weights(weights)
                    updates += 1
                it += 1
            if it > 0:
                avg_steps = avg_steps/it
                avg_score = avg_score/it
                if updates > 0:
                    msg = "TID: %d Global time: %d Avg Steps: %f Avg Score: %f Avg Loss: %f Epsilon: %s"%(thread_id, T, avg_steps, avg_score, avg_loss, epsilon)
                    avg_loss = avg_loss/updates
                    logger_debug.debug(msg)
                    print(msg)
                else:
                    msg = "TID: %d Global time: %d Avg Steps: %f Avg Score: %f"%(thread_id, T, avg_steps, avg_score)
                    logger_debug.debug(msg)
                    print(msg)
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
