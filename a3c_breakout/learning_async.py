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


def predict_back(bqin, bqout, graph, tmodels, pmodel):
    try:
        with graph.as_default():
            while True:
                dt, ID, REQ, op = bqin.get()
                result = None
                if op==agent.OP_GET_PI:
                    result = tmodels[ID][0].predict([dt])[0]
                elif op==agent.OP_GET_VALUE:
                    result = tmodels[ID][0].predict([dt])[1]
                elif op==agent.OP_GET_PI_AND_VALUE:
                    result = tmodels[ID][0].predict([dt])
                elif op == agent.OP_GET_GRADIENT:
                    topt = tmodels[ID][1]
                    result = topt(dt)
                elif op == agent.OP_GET_SHARED_PARAMS:
                    result = pmodel.get_weights()
                elif op == agent.OP_SET_SHARED_PARAMS:
                    pmodel.set_weights(dt)
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

def update_model(qin, graph, pmodel, tmodels, opt, threads, locker):
    try:
        T = 0

        num_threads = len(threads)
        rho = 0.90
        glr = 0.00007
        eps = 1e-1
        decay = 0.99
        batch_size = 32


        with graph.as_default():
            samples = []
            while True:
                data, TID, operation = qin.get()
                #-------------------------------------------------------------------
                #BEGIN UPDATE THREAD NETWORK
                if len(samples) >= batch_size:
                    t = 0
                    loss = 0
                    for sample in samples:
                        #print(grad)
                        loss = opt(sample)[1]
                        t += 1
                        if random.random() < 0.001:
                            print("CURRENT LOSS: %s"%(np.mean(loss)))
                    samples = []
                    tmodels[TID][0].set_weights(pmodel.get_weights())
                    if T > 0 and T % 100000 == 0:
                        print("Saving model in time %d"%(T))
                        pmodel.save_weights("modelp.wght")
                    T += 1
                if operation==agent.OP_ACM_GRADS:
                    samples.append(data)   

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


def server_work(qupdate, prediction_queue, threads):
    try:
        import tensorflow as tf
        import utils
        from keras.backend.tensorflow_backend import set_session
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.3
        #config.gpu_options.gpu_options.allow_growth = True
        #config.inter_op_parallelism_threads = 4
        #set_session(tf.Session(config=config))

        state_size = agent.STATE_SIZE
        action_size = agent.ACTION_SIZE
        learning_rate = agent.LEARNING_RATE
        skip_frames = agent.SKIP_FRAMES
        graph = tf.get_default_graph()
        lock = Lock()
        with graph.as_default():
            pmodel, tmodels, opt = utils.get_model_pair(graph, state_size, skip_frames, action_size, learning_rate, len(threads))

            predicts = []

            for i in threads:
                tmodels[i][0].set_weights(pmodel.get_weights())
                t = threading.Thread(target=predict_back, args=(prediction_queue[i][0], prediction_queue[i][1], graph, tmodels, lock))
                predicts.append(t)
                t.start()

            update_model_work = threading.Thread(target=update_model, args=(qupdate, graph, pmodel, tmodels, opt, threads, lock))
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
    input_uqueue = m.Queue(maxsize=QUEUE_BUFFER_SIZE)

    prediction_queue = []
    for _ in range(MAX_THREADS):
        bqin = m.JoinableQueue(maxsize=QUEUE_BUFFER_SIZE)
        bqout = m.Queue(maxsize=QUEUE_BUFFER_SIZE)
        prediction_queue.append((bqin, bqout))
    
    threads = list(range(MAX_THREADS))
    pool.apply_async(server_work, (input_uqueue, prediction_queue, threads))
    time.sleep(TIME_TO_CLIENTS)
    for j in range(MAX_THREADS):
        pool.apply_async(agent.run, (j, prediction_queue[j][1], prediction_queue[j][0], input_uqueue))

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
