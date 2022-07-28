import numpy as np
import multiprocessing as mp
#from multiprocessing import get_context
#mp.set_start_method('fork')
#mp.freeze_support()
#import os
from functools import partial

def chunks(L, n): 
	return [L[x: x+n] for x in range(0, len(L), n)]

def batching(func,inst_split,max_time=1e7):
	try:
		ctx = mp.get_context('fork')	
	except RuntimeError:
		pass
	for i,inst_batch in enumerate(inst_split):
			print('\nBatch #{}\n'.format(i))
			procs = []
			for i in inst_batch:
				p = mp.Process(target=func, args=(i,),daemon=True)
				procs.append(p)
				p.start()
				#print('p', mp.active_children())
				#print('start', p.name)
			for p in procs:
				#p.close()
				p.join(max_time)
				#print('p', p.name)
				if p.is_alive():
					p.terminate()
					print('end', p.name)


