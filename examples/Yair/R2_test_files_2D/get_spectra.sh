folder=${1}
filename=${2}
ndim=${3}
script=/scratch2/yl899/gitlab/cuartic-potential/R2_full_parallel/FFT_2D.py
script=/scratch2/yl899/gitlab/PISC/PISC/tools/FFT_2D.py
#python ${script} -beta 8 -tau 13 -dt 0.1 -n1 ${ndim} -n2 ${ndim} -t Quantum -file1 ${folder}/${filename}
python ${script} -beta 8 -tau 13 -dt 0.1 -n1 ${ndim} -n2 ${ndim} -t classical -file1 ${folder}/${filename} --what_to_plot FT &
python ${script} -beta 8 -tau 13 -dt 0.1 -n1 ${ndim} -n2 ${ndim} -t classical -file1 ${folder}/${filename} --what_to_plot coscos


