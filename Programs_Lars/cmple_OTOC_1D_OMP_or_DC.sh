#!/bin/bash
cd mylib/oneD
###non OMP, same as below
#python -m numpy.f2py OTOC_module.f90 -m OTOC1D -h --overwrite-signature OTOC1D.pyf' 
#python -m numpy.f2py -c OTOC1D.pyf  OTOC_module.f90' 

###non omp, same as above
#f2py OTOC_module.f90 -m OTOC1D -h --overwrite-signature OTOC1D.pyf 
#f2py -c OTOC1D.pyf  OTOC_module.f90

#OMP, both work/do the same
#f2py -m OTOC1D --fcompiler=gfortran --f90flags='-fopenmp' -lgomp -c OTOC_module.f90


f2py -c --f90flags='-ftree-parallelize-loops=24' -lgomp -m OTOC1D OTOC_module.f90

cd ../..
echo ' '
echo '##### Warning can be ignored! #####'
echo ' '
python OTOC_1D_Fortran.py
