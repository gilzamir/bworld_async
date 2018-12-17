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
                result = tmodels[ID].predict([state])
                resp = None
                if get_policy==1:
                    resp = result[0][0]
                elif get_policy==2:
                    resp = result[1][0]
                else:
                    resp = result
                bqin.task_done()
                bqout.put( (resp, ID, REQ) )
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

def update_model(qin, graph, pmodel, tmodels, threads):
    import gc
    try:
        print("UPDATING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        T = 0
        N = 0
        memory = {}
        num_threads = len(threads)

        for t in range(num_threads):
            memory[t] = []
        loss = 0
        count_loss = 0
        with graph.as_default():
            while True:
                data, TID, sync_net = qin.get()
                if sync_net:
                    tmodels[TID].set_weights(pmodel.get_weights())
                    if T > 0 and T % 50 == 0 and count_loss > 0:
                        print("AVG LOSS %f "%(loss/count_loss))
                        count_loss = 0
                        loss = 0.0
                    if T > 0 and T % 100 == 0:
                        print("SAVING MODELS ON STEP %d........................"%(T))
                        pmodel.save_weights("modelp.wght")
                        T = 1
                        N += 1
                    T += 1
                else:
                    n = 0
                    for state, action, R, svalue, pi in data:
                        n += 1
                        caction = to_categorical(action, agent.ACTION_SIZE)
                        caction[action] = pi[action]
                        adv = R - svalue
                        memory[TID].append( (state, adv, caction, R) )
                    if n >= 0:
                        inputs_c = []
                        advantages_c = []
                        discounts_r_c = []
                        pactions_c = []
                        c = 0
                        while c < n:
                            sstate, adv, action_c, sdisc = memory[TID][c]
                            inputs_c.append(sstate[0])
                            advantages_c.append(adv)
                            pactions_c.append(action_c)
                            discounts_r_c.append(sdisc)
                            c += 1
                        weights = {'out1':np.array(advantages_c), 'out2':np.ones(c)}
                        history = pmodel.fit([np.array(inputs_c)], [np.array(pactions_c), np.array(discounts_r_c)], epochs = T + 1, batch_size = n, sample_weight = weights, initial_epoch = T, verbose=0)
                        loss += history.history['loss'][0]
                        count_loss += 1
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
            pmodel, tmodels = utils.get_model_pair(graph, state_size, skip_frames, action_size, learning_rate, len(threads))

            if START_WITH_PWEIGHTS != None:
                pmodel.load_weights(START_WITH_PWEIGHTS)

            predicts = []

            for i in threads:
                tmodels[i].set_weights(pmodel.get_weights())
                t = threading.Thread(target=predict_back, args=(com[i][0], com[i][1], graph, tmodels))
                predicts.append(t)
                t.start()

            update_model_work = threading.Thread(target=update_model, args=(qupdate, graph, pmodel, tmodels, threads))
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
