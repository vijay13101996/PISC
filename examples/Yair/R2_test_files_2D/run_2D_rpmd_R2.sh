
sys_name='Tosca2'
method='rpmd'
corr_func='R2eq'

pes='mildly_anharmonic_2D'
dim=2
mass=1.0

ens='thermal'
temp=0.125
temp_tau=1.0
pile_lambda=1000.0
dt_therma=0.05
time_therma=50.0

dt_tcf=0.1
dt=0.01
time_total=50.0
ndim=1001

n_traj=1000
nseeds=8
chunk_size=1

#J. Chem. Phys. 156, 131102 (2022)
folder_name='RPMD_R2_tomislav'
param1=1.00 #m
param2=0.5 #w1
param3=0.025 #a*omega**3
param4=0.0025 #(a**2)*omega**4

param5=2.00 #w2
param6=1.6 #a*omega**3
param7=0.64 #b2*omega**4

param8=0.10 #c

icoord_1=0
icoord_2=1
icoord_3=1

nbeads=8


label='rpmd'
dir=/scratch2/yl899/gitlab/PISC

###############################

 rm -f ${folder_name}/*
 python  ${dir}/examples/examples_main.py  -dim ${dim} -pes ${pes} -m ${mass} -sys_name ${sys_name} -method ${method} -ens ${ens} -temp ${temp} -temp_tau ${temp_tau} -dt ${dt} -dt_therma ${dt_therma} -time_therma ${time_therma} -time_total ${time_total} -n_traj ${n_traj} -corr_func ${corr_func} -nbeads ${nbeads} -nseeds ${nseeds} -chunk ${chunk_size} -label ${label} -folder ${folder_name} -pes_param     ${param1} ${param2} ${param3} ${param4} ${param5} ${param6}  ${param7} ${param8} -pile_lambda  ${pile_lambda} --coordinate_list ${icoord_1} ${icoord_2} ${icoord_3}


  python ${dir}/PISC/tools/average_tcf.py -pre ${method}  -folder ${folder_name}
 mv average.out ${folder_name}
 python ${dir}/PISC/tools/rotate_time.py ${folder_name}/average.out 2 ${dt_tcf} aux.dat
 mv aux.dat ${folder_name}/rot_average.out
 ./get_spectra.sh ${folder_name} rot_average.out ${ndim} &
 sleep 10
 ./get_spectra.sh ${folder_name} average.out ${ndim} &

