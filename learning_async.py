import numpy as np
import agent_async as agent
from collections import deque
import logging
from multiprocessing import Queue, Process,  Manager, Pool, Lock, cpu_count
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


def predict_back(bqin, bqout, graph, model, lock):
    try:
        while True:
            state, mask, ID, REQ = bqin.get()
            with graph.as_default():
                lock.acquire()
                result = model.predict([state, mask])
                lock.release()
                bqin.task_done()
                bqout.put( (result, ID, REQ))
    except Exception as e:
        print("Erro nao esperado em predict_back")
        print(e)
    except ValueError as ve:
        print("Erro nao esperado em predict_back")
        print(ve)
    except:
        print("Erro nao esperado em predict_back: %s"%(sys.exc_info()[0]))
        raise

def predict(qin, qout, graph, model, lock):
    try:
        while True:
            state, mask, ID, REQ = qin.get()
            with graph.as_default():
                lock.acquire()
                result = model.predict([state, mask])
                lock.release()
                qin.task_done()
                qout.put( (result, ID, REQ) )
    except Exception as e:
        print("Erro nao esperado em predict")
        print(e)
    except ValueError as ve:
        print("Erro nao esperado em predict")
        print(ve)
    except:
        print("Erro nao esperado em predict: %s"%(sys.exc_info()[0]))
        raise

def update_model(qin, graph, model, back_model, threads, lock, lock_back):
    try:
        print("UPDATING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        T = 0
        loss = 0
        N = 1000
        inputs = {}
        outputs = {}
        masks = {}
        for id in threads:
            inputs[id] = []
            outputs[id] = []
            masks[id] = []
        count_loss = 0
        while True:
            X, Y, TID, apply_gradient = qin.get()
            
            with graph.as_default():
                if not apply_gradient:
                    inputs[TID].append(X[0][0])
                    masks[TID].append(X[1])
                    outputs[TID].append(Y)
                else:
                    cinput = inputs[TID]
                    cmask = masks[TID]
                    if len(cinput) > 0:
                        coutput = outputs[TID]
                        lock_back.acquire()
                        h = back_model.fit(
                                [np.array(cinput), np.array(cmask)], np.array(coutput), epochs=1, batch_size=len(cinput), verbose=0)
                        lock_back.release()
                        loss +=  h.history['loss'][0]
                        count_loss += 1
                        inputs[TID].clear()
                        outputs[TID].clear()
                        masks[TID].clear()
                if T > 0:
                    if T % N == 0 and count_loss > 0:
                        print("T %d LOSS %f"%(T, loss/count_loss))
                        loss = 0.0
                        count_loss = 0
                        #logger_debug.debug("T %d LOSS %f"%(T, c_loss))
                    if T % agent.REFRESH_MODEL_NUM == 0:
                        lock.acquire()
                        model.set_weights(back_model.get_weights())
                        lock.release()
            T += 1
    except Exception as e:
        print("Erro nao esperado em update model")
        print(e)
    except ValueError as ve:
        print("Erro nao esperado em update model")
        print(ve)
    except:
        print("Erro nao esperado em update model: %s"%(sys.exc_info()))
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
        state_size = agent.STATE_SIZE
        action_size = agent.ACTION_SIZE
        learning_rate = agent.LEARNING_RATE
        skip_frames = agent.SKIP_FRAMES
        graph = tf.get_default_graph()

        lock = threading.Lock()
        lock_back = threading.Lock()

        with graph.as_default():
            model, back_model = utils.get_model_pair(graph, state_size, skip_frames, action_size, learning_rate)
            back_model.set_weights(model.get_weights())
        predict_work = threading.Thread(target=predict, args=(input_queue, output_queue, graph, model, lock))  
        predict_bwork = threading.Thread(target=predict_back, args=(bqin, bqout, graph, back_model, lock_back))
        update_model_work = threading.Thread(target=update_model, args=(qupdate, graph, model, back_model, threads, lock, lock_back))
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
    MAX_THREADS = cpu_count()
    pool = Pool(MAX_THREADS+1)
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
