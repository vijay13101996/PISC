import numpy as np
import multiprocessing as mp

def chunks(L, n): 
	return [L[x: x+n] for x in range(0, len(L), n)]

def batching(func,inst_split,max_time=1e7):
	for inst_batch in inst_split:
		procs = []
		for i in inst_batch:
			p = mp.Process(target=func, args=(i,))
			procs.append(p) 
			p.start()
			#print('start', p.name)
		for p in procs:
			p.join(max_time)
			if p.is_alive():
				p.terminate()
				print('end', p.name)


