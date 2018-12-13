import numpy as np
import agent_async as agent
from collections import deque
import logging
from multiprocessing import Queue, Process,  Manager, Pool, Lock, cpu_count
import threading
import random
from collections import deque
import time
from keras.models import clone_model
import sys
from keras.utils import to_categorical


logger_debug = logging.getLogger(__name__)
logger_debug.setLevel(logging.DEBUG)

handler = logging.FileHandler('train_debug.log')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger_debug.addHandler(handler)
logger_debug.propagate = False


def predict_back(bqin, bqout, graph, pmodel, vmodel, lock):
    mask = np.array([np.ones(agent.ACTION_SIZE)])
    try:
        while True:
            state, ID, REQ, get_policy = bqin.get()
            with graph.as_default():
                lock.acquire()
                result = None
                if get_policy==1:
                   result = pmodel.predict([state])
                elif get_policy==2:
                    result = vmodel.predict([state])
                else:
                    result = (pmodel.predict([state]), vmodel.predict([state]))
                lock.release()
                bqin.task_done()
                bqout.put( (result, ID, REQ))
    except ValueError as ve:
        print("Erro nao esperado em predict_back")
        print(ve)
        raise
    except Exception as e:
        print("Erro nao esperado em predict_back")
        print(e)
        raise
    except:
        print("Erro nao esperado em predict_back: %s"%(sys.exc_info()[0]))
        raise

def predict(qin, qout, graph, pmodel, vmodel, lock):
    try:
        while True:
            state, ID, REQ, is_policy = qin.get()
            with graph.as_default():
                lock.acquire()
                result = None
                if is_policy==1:
                    result = pmodel.predict([state])
                elif is_policy==2:
                    result = vmodel.predict([state])
                else:
                    result = (pmodel.predict([state]), vmodel.predict([state]))
                lock.release()
                qin.task_done()
                qout.put( (result, ID, REQ) )
                print('------------------------------------------------')
                print(result)
    except ValueError as ve:
        print("Erro nao esperado em predict")
        print(ve)
        raise
    except Exception as e:
        print("Erro nao esperado em predict")
        print(e)
        raise
    except:
        print("Erro nao esperado em predict: %s"%(sys.exc_info()[0]))
        raise

def update_model(qin, graph, pmodel, vmodel, back_pmodel, back_vmodel, popt, vopt, threads, lock, lock_back):
    try:
        print("UPDATING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        T = 0
        ploss = 0
        closs = 0
        N = 200
        inputs = {}
        pactions = {}
        advantages = {}
        discounts_r = {}
        for id in threads:
            inputs[id] = deque(maxlen=50000)
            pactions[id] = deque(maxlen=50000)
            advantages[id] = deque(maxlen=50000)
            discounts_r[id] = deque(maxlen=50000)
        count_loss = 0
        while True:
            state, action, R, _, svalue, TID, apply_gradient = qin.get()
            with graph.as_default():
                if not apply_gradient:
                    inputs[TID].append(state[0])
                    cat_action = to_categorical(action, agent.ACTION_SIZE)
                    pactions[TID].append(cat_action)
                    advantages[TID].append(R-svalue)
                    discounts_r[TID].append(R)
                else:
                    inputs_c = inputs[TID]
                    if len(inputs_c) > 0:
                        pactions_c = pactions[TID]
                        advantages_c = advantages[TID]
                        discounts_r_c = discounts_r[TID]

                        lock.acquire()
                        h = popt([np.array(inputs_c), np.array(pactions_c), np.array(advantages_c)])
                        #print(h)
                        ploss +=  np.mean(h)
                        h = vopt([np.array(inputs_c), np.array(discounts_r_c)])
                        #print(h)
                        closs +=  np.mean(h)
                        lock.release()
                        
                        count_loss += 1
                        inputs[TID].clear()
                        pactions[TID].clear()
                        advantages[TID].clear()
                        discounts_r[TID].clear()
                if T > 0:
                    if T % N == 0 and count_loss > 0:
                        print("T %d PLOSS %f  CLOSS %f"%(T, ploss/count_loss, closs/count_loss))
                        ploss = 0.0
                        closs = 0.0
                        count_loss = 0
                        #logger_debug.debug("T %d LOSS %f"%(T, c_loss))
                    #if T % agent.REFRESH_MODEL_NUM == 0:
                    lock_back.acquire()
                    back_pmodel.set_weights(pmodel.get_weights())
                    back_vmodel.set_weights(vmodel.get_weights())
                    lock_back.release()
                    if T % 1000000 == 0:
                        pmodel.save_weights("pmodel_%d.wght"%(T))
                        vmodel.save_weights("vmodel_%d.wght"%(T))
            T += 1
    except ValueError as ve:
        print("Erro (ValueError) nao esperado em update model")
        print(ve)
        raise
    except Exception as e:
        print("Erro nao esperado em update model")
        print(e)
        raise
    except:
        print("Erro nao esperado em update model: %s"%(sys.exc_info()))
        raise


def server_work(input_queue, output_queue, qupdate, bqin, bqout, threads):
    try:
        import tensorflow as tf
        import utils
        #from keras.backend.tensorflow_backend import set_session
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.3
        #config.gpu_options.gpu_options.allow_growth = True
        #set_session(tf.Session(config=config))

        START_WITH_PWEIGHTS = None #nome do arquivo de pesos jah treinados. Se None, inicia do zero
        START_WITH_VWEIGHTS = None

        state_size = agent.STATE_SIZE
        action_size = agent.ACTION_SIZE
        learning_rate = agent.LEARNING_RATE
        skip_frames = agent.SKIP_FRAMES
        graph = tf.get_default_graph()

        lock = threading.Lock()
        lock_back = threading.Lock()

        with graph.as_default():
            pmodel, vmodel, back_pmodel, back_vmodel, popt, vopt = utils.get_model_pair(graph, state_size, skip_frames, action_size, learning_rate)
            
            if START_WITH_PWEIGHTS != None:
                pmodel.load_weights(START_WITH_PWEIGHTS)
            
            if START_WITH_VWEIGHTS != None:
                vmodel.load_weights(START_WITH_VWEIGHTS)

            back_pmodel.set_weights(pmodel.get_weights())
            back_vmodel.set_weights(vmodel.get_weights())

            predict_work = threading.Thread(target=predict, args=(input_queue, output_queue, graph, pmodel, vmodel, lock))  
            predict_bwork = threading.Thread(target=predict_back, args=(bqin, bqout, graph, back_pmodel, back_vmodel, lock_back))
            update_model_work = threading.Thread(target=update_model, args=(qupdate, graph, pmodel, vmodel, back_pmodel, back_vmodel, popt, vopt, threads, lock, lock_back))
            predict_work.start()
            update_model_work.start()        
            predict_bwork.start()

        predict_work.join()
        predict_bwork.join()
        update_model_work.join()
    except ValueError as ve:
        print("Erro nao esperado em server_work")
        print(ve)
        raise
    except Exception as e:
        print("Erro nao esperado em server_work")
        print(e)
        raise
    except:
        print("Erro nao esperado em server_work")
        raise

def main():
    m = Manager()
    MAX_THREADS = 4
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
