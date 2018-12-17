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
        with graph.as_default():
            while True:
                state, ID, REQ, get_policy = bqin.get()
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

def update_model(qin, graph, pmodel, vmodel, tmodels, opt1, opt2, threads):
    import gc
    try:
        print("UPDATING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        T = 0
        N = 0
        memory = {}
        num_threads = len(threads)

        for t in range(num_threads):
            memory[t] = []

        with graph.as_default():
            while True:
                data, TID, sync_net = qin.get()
                if sync_net:
                    tmodels[TID][0].set_weights(pmodel.get_weights())
                    tmodels[TID][1].set_weights(vmodel.get_weights())
                    if T > 0 and T % 500000 == 0:
                        print("SAVING MODELS ON STEP %d........................"%(T))
                        pmodel.save_weights("modelp.wght")
                        vmodel.save_weights("modelv.wght")
                        T = 1
                        N += 1
                    T += 1
                else:
                    n = 0
                    for state, action, R, svalue in data:
                        n += 1
                        memory[TID].append( (state[0], to_categorical(action, agent.ACTION_SIZE), R-svalue, R) )
                    if n >= 0:
                        inputs_c = []
                        pactions_c = []
                        advantages_c = []
                        discounts_r_c = []
                        c = 0
                        while c < n:
                            sstate, saction, sadv, sdisc = memory[TID][c]
                            inputs_c.append(sstate)
                            pactions_c.append(saction)
                            advantages_c.append(sadv)
                            discounts_r_c.append(sdisc)
                            c += 1

                        opt1([np.array(inputs_c), np.array(pactions_c), np.array(advantages_c)])
                        opt2([np.array(inputs_c), np.array(discounts_r_c)])
                        
                        memory[TID] = []
 

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
        from keras.backend.tensorflow_backend import set_session
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.3
        #config.gpu_options.gpu_options.allow_growth = True
        #config.inter_op_parallelism_threads = 4
        #set_session(tf.Session(config=config))
        START_WITH_PWEIGHTS = None #nome do arquivo de pesos jah treinados. Se None, inicia do zero
        START_WITH_VWEIGHTS = None

        state_size = agent.STATE_SIZE
        action_size = agent.ACTION_SIZE
        learning_rate = agent.LEARNING_RATE
        skip_frames = agent.SKIP_FRAMES
        graph = tf.get_default_graph()

        with graph.as_default():
            pmodel, vmodel, tmodels, opt1, opt2 = utils.get_model_pair(graph, state_size, skip_frames, action_size, learning_rate, len(threads))

            if START_WITH_PWEIGHTS != None:
                pmodel.load_weights(START_WITH_PWEIGHTS)
            
            if START_WITH_VWEIGHTS != None:
                vmodel.load_weights(START_WITH_VWEIGHTS)

            predicts = []

            for i in threads:
                tmodels[i][0].set_weights(pmodel.get_weights())
                tmodels[i][1].set_weights(vmodel.get_weights())
                t = threading.Thread(target=predict_back, args=(com[i][0], com[i][1], graph, tmodels))
                predicts.append(t)
                t.start()

            update_model_work = threading.Thread(target=update_model, args=(qupdate, graph, pmodel, vmodel, tmodels, opt1, opt2, threads))
            update_model_work.start()

            for i in threads:
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
    QUEUE_BUFFER_SIZE = 10
    m = Manager()
    MAX_THREADS = 4
    TIME_TO_CLIENTS = 0.5
    pool = Pool(MAX_THREADS+1)
    input_queue = m.JoinableQueue(maxsize=QUEUE_BUFFER_SIZE)
    output_queue = m.Queue(maxsize=QUEUE_BUFFER_SIZE)
    input_uqueue = m.Queue(maxsize=QUEUE_BUFFER_SIZE)

    com = []
    for _ in range(MAX_THREADS):
        bqin = m.JoinableQueue(maxsize=QUEUE_BUFFER_SIZE)
        bqout = m.Queue(maxsize=QUEUE_BUFFER_SIZE)
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
