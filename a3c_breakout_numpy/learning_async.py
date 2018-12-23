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


def predict_back(bqin, bqout, graph, tmodels, locker):
    try:
        with graph.as_default():
            while True:
                state, ID, REQ, get_policy = bqin.get()
                result = None
                locker.acquire()
                if get_policy==1:
                    result = tmodels[ID][0].predict([state])[0]
                elif get_policy==2:
                    result = tmodels[ID][0].predict([state])[1]
                else:
                    result = tmodels[ID][0].predict([state])
                locker.release()
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

def update_model(qin, graph, pmodel, tmodels, opt, get_loss, threads, locker):
    try:
        print("UPDATING>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        T = 0

        memory = {}
        num_threads = len(threads)
        rho = 0.90
        glr = 0.00007
        eps = 1e-1
        decay = 0.99

        for t in range(num_threads):
            memory[t] = []
        N = 0
        with graph.as_default():
            shared_params = pmodel.get_weights()
            idx = np.arange(0, len(shared_params))  
            grad_acm = []
            avg_loss = 0.0
            while True:
                data, TID, sync_net = qin.get()
                if sync_net:
                    if N > 0:
                        t = 0
                        acm = [np.zeros(w.shape) for w in shared_params]
                        for grads in grad_acm:
                            lr = glr
                            if decay > 0:
                                lr = lr * (1.0/(1.0 + decay * t))
                            t += 1
                            for p, gr, i in zip(shared_params, grads, idx):
                                acm[i] = rho * acm[i] + (1-rho) * np.square(grad[i])
                                shared_params[i] = p - lr * gr/(np.sqrt(acm[i]+eps))

                        if N >= 1000:
                            print("LOSS %f"%(avg_loss/N))
                            avg_loss = 0.0
                            N = 0
                        
                        grad_acm = []
                    tmodels[TID][0].set_weights(shared_params)
                    if T > 0 and T % 100000 == 0:
                        print("Saving model in time %d"%(T))
                        pmodel.set_weights(shared_params)
                        pmodel.save_weights("modelp.wght")
                    T += 1
                    continue
                n = 0
                for state, action, R, svalue, _ in data:
                    n += 1
                    caction = to_categorical(action, agent.ACTION_SIZE)
                    advantage = R - svalue
                    memory[TID].append( (state, caction, advantage, R) )
                
                if n >= 0:
                    inputs_c = []
                    advantages_c = []
                    discounts_r_c = []
                    pactions_c = []
                    c = 0
                    while c < n:
                        sstate, action_c, adv, R = memory[TID][c]
                        #print('-------------------------------------------------------')
                        #print(sstate.shape)
                        inputs_c.append(sstate[0])
                        advantages_c.append([adv])
                        pactions_c.append(action_c)
                        discounts_r_c.append([R])
                        c += 1

                    input_dt = [np.array(inputs_c), np.array(pactions_c), np.array(advantages_c), np.array(discounts_r_c)]
                
                    avg_loss += tmodels[0][2](input_dt)[0]

                    topt = tmodels[TID][1]
                    grad = topt(input_dt)
                    #print("NAO CHEGOU ______________________________________________-")               
                    grad_acm.append(grad)
                             
                    N += 1
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

        state_size = agent.STATE_SIZE
        action_size = agent.ACTION_SIZE
        learning_rate = agent.LEARNING_RATE
        skip_frames = agent.SKIP_FRAMES
        graph = tf.get_default_graph()
        lock = Lock()
        with graph.as_default():
            pmodel, tmodels, opt, get_loss = utils.get_model_pair(graph, state_size, skip_frames, action_size, learning_rate, len(threads))

            predicts = []

            for i in threads:
                tmodels[i][0].set_weights(pmodel.get_weights())
                t = threading.Thread(target=predict_back, args=(com[i][0], com[i][1], graph, tmodels, lock))
                predicts.append(t)
                t.start()

            update_model_work = threading.Thread(target=update_model, args=(qupdate, graph, pmodel, tmodels, opt, get_loss, threads, lock))
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
