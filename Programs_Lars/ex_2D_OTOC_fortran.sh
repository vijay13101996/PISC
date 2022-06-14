#!/bin/bash
cd mylib/twoD

f2py -c --f90flags='-ftree-parallelize-loops=24' -lgomp -m OTOC2D 2D_OTOC_module.f90

cd ../..
echo ' '
echo '##### Warning can be ignored! #####'
echo ' '
python OTOC_2D_Fortran.py
