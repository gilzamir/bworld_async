import numpy as np
import agent_async as agent
from collections import deque
from multiprocessing import Queue, Process,  Manager, Pool, Lock, cpu_count
import threading
import random
from collections import deque
import time
from keras.models import clone_model
import sys
from keras.utils import to_categorical


def predict_back(bqin, bqout, graph, tmodels):
    try:
        while True:
            state, ID, REQ, get_policy = bqin.get()
            with graph.as_default():
                result = None
                if get_policy==1:
                   result = tmodels[ID][0].predict([state])
                elif get_policy==2:
                    result = tmodels[ID][1].predict([state])
                else:
                    result = (tmodels[ID][0].predict([state]), tmodels[ID][1].predict([state]))
                bqin.task_done()
                bqout.put( (result, ID, REQ) )
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

def update_model(qin, graph, pmodel, vmodel, tmodels, opt, threads, lock):
    try:
        print("UPDATING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        T = 0
        loss = 0
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
            state, action, R, svalue, TID, apply_gradient = qin.get()
            #lock.acquire()
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

                        
                        h = opt([np.array(inputs_c), np.array(pactions_c), np.array(advantages_c), np.array(discounts_r_c)])
                        
                        loss +=  np.mean(h)

                        tmodels[TID][0].set_weights(pmodel.get_weights())
                        tmodels[TID][1].set_weights(vmodel.get_weights())

                        count_loss += 1
                        inputs[TID].clear()
                        pactions[TID].clear()
                        advantages[TID].clear()
                        discounts_r[TID].clear()    
                if T > 0:
                    if T % N == 0 and count_loss > 0:
                        print("T %d TOTAL LOSS %f"%(T, loss/count_loss))
                        loss = 0.0
                        count_loss = 0
                    if T % 5000000 == 0:
                        pmodel.save_weights("pmodel_%d.wght"%(T))
                        vmodel.save_weights("vmodel_%d.wght"%(T))
            T += 1
            #lock.release()
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


def server_work(input_queue, output_queue, qupdate, com, threads):
    try:
        import tensorflow as tf
        import utils
        #from keras.backend.tensorflow_backend import set_session
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.3
        #config.gpu_options.gpu_options.allow_growth = True
        #set_session(tf.Session(config=config))
        num_threads = len(threads)
        START_WITH_PWEIGHTS = None #nome do arquivo de pesos jah treinados. Se None, inicia do zero
        START_WITH_VWEIGHTS = None

        state_size = agent.STATE_SIZE
        action_size = agent.ACTION_SIZE
        learning_rate = agent.LEARNING_RATE
        skip_frames = agent.SKIP_FRAMES
        graph = tf.get_default_graph()

        lock = threading.Lock()

        with graph.as_default():
            pmodel, vmodel, tmodels, opt = utils.get_model_pair(graph, state_size, skip_frames, action_size, learning_rate, num_threads)

            if START_WITH_PWEIGHTS != None:
                pmodel.load_weights(START_WITH_PWEIGHTS)
            
            if START_WITH_VWEIGHTS != None:
                vmodel.load_weights(START_WITH_VWEIGHTS)

            predicts = []

            for i in range(num_threads):
                tmodels[i][0].set_weights(pmodel.get_weights())
                tmodels[i][1].set_weights(vmodel.get_weights())
                t = threading.Thread(target=predict_back, args=(com[i][0], com[i][1], graph, tmodels))
                predicts.append(t)
                t.start()

            update_model_work = threading.Thread(target=update_model, args=(qupdate, graph, pmodel, vmodel, tmodels, opt, threads, lock))
            update_model_work.start()

            for i in range(num_threads):
                predicts[i].join()
            
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
    MAX_THREADS = 8
    TIME_TO_CLIENTS = 0.5
    pool = Pool(MAX_THREADS+1)
    input_queue = m.JoinableQueue()
    output_queue = m.Queue()
    input_uqueue = m.Queue()

    com = []
    for _ in range(MAX_THREADS):
        bqin = m.JoinableQueue()
        bqout = m.Queue()
        com.append((bqin, bqout))
    
    threads = list(range(MAX_THREADS))
    pool.apply_async(server_work, (input_queue, output_queue, input_uqueue, com, threads))
    time.sleep(TIME_TO_CLIENTS)
    for j in range(MAX_THREADS):
        pool.apply_async(agent.run, (j, output_queue, input_queue, com[j][1], com[j][0], input_uqueue))

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
