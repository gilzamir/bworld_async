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


logger_debug = logging.getLogger(__name__)
logger_debug.setLevel(logging.DEBUG)

handler = logging.FileHandler('train_debug.log')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
logger_debug.addHandler(handler)
logger_debug.propagate = False

def predict_back(bqin, bqout, graph, pi, V, lock):
    try:
        while True:
            state, ID, REQ, get_policy = bqin.get()
            with graph.as_default():
                lock.acquire()
                result = None
                if get_policy==1:
                   result = pi.predict([state])
                elif get_policy==2:
                    result = V.predict([state])
                else:
                    result = (pi.predict([state]), V.predict([state]))
                lock.release()
                bqin.task_done()
                bqout.put( (result, ID, REQ))
    except ValueError as ve:
        print("Erro nao esperado em predict_back")
        print(ve)
    except Exception as e:
        print("Erro nao esperado em predict_back")
        print(e)
    except:
        print("Erro nao esperado em predict_back: %s"%(sys.exc_info()[0]))
        raise

def predict(qin, qout, graph, pi, V, lock):
    try:
        while True:
            state, ID, REQ, is_policy = qin.get()
            with graph.as_default():
                lock.acquire()
                result = None
                if is_policy==1:
                    result = pi.predict([state])
                elif is_policy==2:
                    result = V.predict([state])
                else:
                    result = (pi.predict([state]), V.predict([state]))
                lock.release()
                qin.task_done()
                qout.put( (result, ID, REQ) )
    except ValueError as ve:
        print("Erro nao esperado em predict")
        print(ve)
    except Exception as e:
        print("Erro nao esperado em predict")
        print(e)
    except:
        print("Erro nao esperado em predict: %s"%(sys.exc_info()[0]))
        raise

def update_model(qin, graph, pi, pi2, V, V2, threads, lock, lock_back):
    try:
        print("UPDATING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        T = 0
        loss = 0
        N = 1000
        inputs = {}
        poutputs = {}
        voutputs = {}
        for id in threads:
            inputs[id] = deque(maxlen=50000)
            poutputs[id] = deque(maxlen=50000)
            voutputs[id] = deque(maxlen=50000)
        count_loss = 0
        while True:
            state, R, pvalue, svalue, TID, apply_gradient = qin.get()
            with graph.as_default():
                if not apply_gradient:
                    inputs[TID].append(state[0])
                    poutputs[TID].append( (R-svalue[0]) * pvalue[0] )
                    voutputs[TID].append(R)
                else:
                    cinput = inputs[TID]
                    if len(cinput) > 0:
                        coutput = poutputs[TID]
                        lock.acquire()
                        h = pi.fit(
                                [np.array(cinput)], np.array(coutput), epochs=1, batch_size=agent.GRADIENT_BATCH, verbose=0)
                        loss +=  0.5 * h.history['loss'][0]
                        coutput = voutputs[TID]

                        h2 = V.fit(
                                [np.array(cinput)], np.array(coutput), epochs=1, batch_size=agent.GRADIENT_BATCH, verbose=0)
                        loss +=  0.5 * h2.history['loss'][0]

                        lock.release()
                        count_loss += 1
                        inputs[TID].clear()
                        poutputs[TID].clear()
                        voutputs[TID].clear()
                if T > 0:
                    if T % N == 0 and count_loss > 0:
                        print("T %d LOSS %f"%(T, loss/count_loss))
                        loss = 0.0
                        count_loss = 0
                        #logger_debug.debug("T %d LOSS %f"%(T, c_loss))
                    #if T % agent.REFRESH_MODEL_NUM == 0:
                    lock_back.acquire()
                    pi2.set_weights(pi.get_weights())
                    V2.set_weights(V.get_weights())
                    lock_back.release()
                    if T % 1000000 == 0:
                        pi.save_weights("pimodel_%d.wght"%(T))
                        V.save_weights("vmodel_%d.wght"%(T))
            T += 1
    except ValueError as ve:
        print("Erro nao esperado em update model")
        print(ve)
    except Exception as e:
        print("Erro nao esperado em update model")
        print(e)
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

        START_WITH_WEIGHTS = None #nome do arquivo de pesos jah treinados. Se None, inicia do zero

        state_size = agent.STATE_SIZE
        action_size = agent.ACTION_SIZE
        learning_rate = agent.LEARNING_RATE
        skip_frames = agent.SKIP_FRAMES
        graph = tf.get_default_graph()

        lock = threading.Lock()
        lock_back = threading.Lock()

        with graph.as_default():
            pi, pi2, V, V2 = utils.get_model_pair(graph, state_size, skip_frames, action_size, learning_rate)
            
            if START_WITH_WEIGHTS != None:
                pi.load_weights(START_WITH_WEIGHTS[0])
                V.load_weights(START_WITH_WEIGHTS[1])
                
            pi2.set_weights(pi.get_weights())
            V2.set_weights(V.get_weights())
            predict_work = threading.Thread(target=predict, args=(input_queue, output_queue, graph, pi, V, lock))  
            predict_bwork = threading.Thread(target=predict_back, args=(bqin, bqout, graph, pi2, V2, lock_back))
            update_model_work = threading.Thread(target=update_model, args=(qupdate, graph, pi, pi2, V, V2, threads, lock, lock_back))
            predict_work.start()
            update_model_work.start()        
            predict_bwork.start()

        predict_work.join()
        predict_bwork.join()
        update_model_work.join()
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(e)
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
