import numpy as np

def reorder_time(array,ndim,mode=1):
    """Reorder inputs in array to have a specific time ordering"""
    if mode ==0:
        return array
    elif mode ==1:
       #input  [0,1,...,tlen,-1,...,-tlen]
       #output [-tlen,...,-1,0,1,...,tlen]
       if len(array.shape)==1:
           aux_array=array[:, np.newaxis]
       else:
           aux_array=array
       tlen=ndim//2
       new_array=np.roll(aux_array,tlen,axis=0)
       new_array[0:tlen,:]=np.flip(aux_array[tlen+1:,:],axis=0)
       return new_array
    else:
        raise NotImplementedError

