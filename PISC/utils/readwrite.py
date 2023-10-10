import numpy as np
import pickle

def store_1D_plotdata(x,y,fname,fpath=None,ebar=None):
        if(ebar is None):
                data = np.column_stack([x,y])
        else:
                data = np.column_stack([x,y,ebar])

        if(fpath is None):
                datafile_path = "/home/vgs23/Pickle_files/{}.txt".format(fname)
        else:
                datafile_path = "{}/{}.txt".format(fpath,fname)
        np.savetxt(datafile_path , data)#,fmt = ['%f','%f'])

def read_1D_plotdata(fname):
        data = np.loadtxt("{}".format(fname),dtype=complex)
        return data

def store_2D_imagedata_column(X,Y,F,fname,fpath=None,extcol=None):
        if extcol is None:
                data = np.column_stack([X.flatten(), Y.flatten(), F.flatten()])
        else:
                data = np.column_stack([X.flatten(), Y.flatten(), F.flatten(), extcol.flatten()])

        if(fpath is None):
                datafile_path = "/home/vgs23/Pickle_files/{}.txt".format(fname)
        else:
                datafile_path = "{}/{}.txt".format(fpath,fname)

        np.savetxt(datafile_path,data)

def store_2D_imagedata(X,Y,F,fname,fpath=None):
        """ Saves data assuming two independendent variables (normally time) and one observable (normally a correlation function)
        Format:
            write('#x')
            np.savedata(X)
            write('#y')
            np.savedata(X)
            write('#f')
            np.savedata(F)
             """""
        if(len(np.array(X).shape) == 1 and len(np.array(Y).shape) ==1):
                x,y = np.meshgrid(X,Y)
        else:
                x,y = X,Y

        if(fpath is None):
                datafile_path = "/home/vgs23/Pickle_files/{}.txt".format(fname)
        else:
                datafile_path = "{}/{}.txt".format(fpath,fname)

        with open(datafile_path, 'w') as outfile:
                for data,qual in zip([x,y,F],['x','y','f']):
                        outfile.write('#{}\n'.format(qual))
                        np.savetxt(outfile, data)

def read_2D_imagedata(fname):
    """ Reads data saved with the function store_2D_imagedata """
    data = np.loadtxt(fname)
    ndim=len(data)//3
    X = data[:ndim]
    Y = data[ndim:2*ndim]
    F = data[2*ndim:]
    return X,Y,F

def chunks(L, n):
        return [L[x: x+n] for x in range(0, len(L), n)]

def store_arr(arr,fname,fpath=None):
        if(fpath is None):
                f = open('/home/vgs23/Pickle_files/{}.dat'.format(fname),'wb')
        else:
                f = open('{}/{}.dat'.format(fpath,fname),'wb')
        pickle.dump(arr,f)

def read_arr(fname,fpath=None):
        if fpath is None:
                f = open('/home/vgs23/Pickle_files/{}.dat'.format(fname),'rb')
                arr = pickle.load(f)
        else:
                f = open('{}/{}.dat'.format(fpath,fname),'rb')
                arr = pickle.load(f)
        return arr

