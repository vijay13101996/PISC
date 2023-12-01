f2py3 -c --f90flags="-fopenmp" -m misc_f misc_f.f90  -lgomp
f2py3 -c --f90flags="-fopenmp" -m nmtrans_f nmtrans_f.f90  -lgomp

