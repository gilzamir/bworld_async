import numpy as np
import agent_async as agent
from collections import deque
import logging
from multiprocessing import Queue, Process,  Manager, Pool, Lock
import threading
import random
import queue
import time
import sys


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

def predict_back(bqin, bqout, graph, model):
    try:
        while True:
            state, mask = bqin.get()
            with graph.as_default():
                result = model.predict([state, mask])
                bqout.put(result)
            bqin.task_done()
    except Exception as e:
        print("Erro nao esperado em predict_back")
        print(e)
    except ValueError as ve:
        print("Erro nao esperado em predict_back")
        print(ve)
    except:
        print("Erro nao esperado em predict_back: %s"%(sys.exc_info()[0]))
        raise

def predict(qin, qout, graph, model):
    try:
        while True:
            state, mask = qin.get()
            with graph.as_default():
                result = model.predict([state, mask])
                qout.put(result)
            qin.task_done()
    except Exception as e:
        print("Erro nao esperado em predict")
        print(e)
    except ValueError as ve:
        print("Erro nao esperado em predict")
        print(ve)
    except:
        print("Erro nao esperado em predict: %s"%(sys.exc_info()[0]))
        raise

def update_model(qin, graph, model, back_model, threads):
    try:
        print("UPDATING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        T = 0
        loss = 0
        N = 1000
        gradients = {}
        for id in threads:
            gradients[id] = []
        while True:

            X, Y, TID, apply_gradient = qin.get()
            
            with graph.as_default():
                if not apply_gradient:
                    old_w = model.get_weights()   
                    c_loss = model.train_on_batch(X, Y)
                    loss += c_loss
                    new_w = model.get_weights()
                    gradient = gradients[TID]
                    if len(gradient) == 0:
                         for op, np in list(zip(old_w, new_w)):
                            gradient.append(np - op)
                    else:
                        idx = 0
                        for op, np in list(zip(old_w, new_w)):
                            gradient[idx] += np - op
                            idx += 1
                    if T > 0 and T % N == 0:
                        print("T %d LOSS %f"%(T, loss/N))
                        loss = 0.0
                    logger_debug.debug("T %d LOSS %f"%(T, c_loss))
                else:
                    gradient = gradients[TID]
                    if len(gradient) > 0:
                        weights = model.get_weights()
                        idx = 0
                        for p, g in list(zip(weights, gradient)):
                            weights[idx] = p + g
                            idx += 1
                        model.set_weights(weights) 
                        gradients[TID] = []
                        #print("GRADIENT UPDATING %d >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"%(TID))
                if T > 0 and T % agent.REFRESH_MODEL_NUM == 0:
                    back_model.set_weights(model.get_weights())
            T += 1
    except Exception as e:
        print("Erro nao esperado em update model")
        print(e)
    except ValueError as ve:
        print("Erro nao esperado em update model")
        print(ve)
    except:
        print("Erro nao esperado em update model: %s"%(sys.exc_info()[0]))
        raise
        

def server_work(input_queue, output_queue, qupdate, bqin, bqout, threads):
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
        with graph.as_default():
            model, back_model = utils.get_model_pair(graph, state_size, skip_frames, action_size, learning_rate)
      
        predict_work = threading.Thread(target=predict, args=(input_queue, output_queue, graph, model))  
        predict_bwork = threading.Thread(target=predict_back, args=(bqin, bqout, graph, back_model))
        update_model_work = threading.Thread(target=update_model, args=(qupdate, graph, model, back_model, threads))
        predict_work.start()
        update_model_work.start()        
        predict_bwork.start()

        predict_work.join()
        predict_bwork.join()
        update_model_work.join()
    except Exception as e:
        print(e)
    except ValueError as ve:
        print(ve)
    except:
        print("Erro nao esperado em server_work")
        raise

def main():
    m = Manager()
    agents = []
    MAX_THREADS = 4
    pool = Pool()
    input_queue = m.JoinableQueue()
    output_queue = m.Queue()
    input_uqueue = m.Queue()
    bqin = m.JoinableQueue()
    bqout = m.Queue()
    threads = list(range(MAX_THREADS))
    pool.apply_async(server_work, (input_queue, output_queue, input_uqueue, bqin, bqout, threads))

    for j in range(MAX_THREADS):
        pool.apply_async(agent.run, (j, output_queue, input_queue, bqout, bqin, input_uqueue))
    
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
