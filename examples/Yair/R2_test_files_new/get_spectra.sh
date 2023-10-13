folder=classical_R2
filename=average.out
script=/scratch2/yl899/gitlab/cuartic-potential/R2_full_parallel/FFT_2D.py

python ${script} -beta 8 -tau 13 -dt 0.1 -n1 401 -n2 401 -t Quantum -file1 ${folder}/${filename}
