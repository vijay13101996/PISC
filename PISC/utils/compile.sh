
#Compile tcf_fort_tools library

#f2py3 -c --f90flags="-fopenmp" -m tcf_fort_tools tcf_fort_tools.f90 -lgomp
f2py3 -c --f90flags="-fopenmp" -m tcf_fort_tools_omp tcf_fort_tools.f90 -lgomp

