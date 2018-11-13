
from multiprocessing import Queue, Process
import multiprocessing as mp
import time

def work1(qin, qout):
	x = 0
	while True:
		qout.put(x)
		print("work 1 send %d"%(x))
		try:
			bx = qin.get()
			print("work 1 receive %d"%(bx))
			x = bx
		except ValueError:
			print("erro")
			pass
def work2(qin, qout):
	x = 0
	while True:
		try:
			bx = qin.get()
			print("work 2 receive %d"%(bx))			
			x = bx + 1
			qout.put(x)
			print("work 2 send %d"%(x))
		except ValueError:
			print("ERRO")
			pass

def main():
	m = mp.Manager()

	qin = m.Queue()
	qout = m.Queue()
	pool = mp.Pool()

	pool.apply_async(work1, (qin, qout))
	pool.apply_async(work2, (qout, qin))

	pool.close()
	pool.join()

if __name__=="__main__":
	main()



