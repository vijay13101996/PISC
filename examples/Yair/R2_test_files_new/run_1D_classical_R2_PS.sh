
sys_name='Tosca2'
method='Classical'
corr_func='R2eq'

pes='mildly_anharmonic'
dim=1
param1=1.0 #m
param2=-0.605 #a
param3=+0.427 #b
mass=1.0
nbeads='1'


ens='thermal'
temp=0.125 #beta 8
temp_tau=1.0
pile_lambda=1000.0
dt_therma=0.05
time_therma=50.0

dt=0.01
time_total=50.0
ndim=1001

n_traj=1000
nseeds=20
chunk_size=10
label='RPMD'
folder_name='classical_R2_PS'

dir=/scratch2/yl899/gitlab/PISC
rm -f ${folder_name}/*
python  ${dir}/examples/examples_main.py  -dim ${dim} -pes ${pes} -m ${mass} -sys_name ${sys_name} -method ${method} -ens ${ens} -temp ${temp} -temp_tau ${temp_tau} -dt ${dt} -dt_therma ${dt_therma} -time_therma ${time_therma} -time_total ${time_total} -n_traj ${n_traj} -corr_func ${corr_func} -nbeads ${nbeads} -nseeds ${nseeds} -chunk ${chunk_size} -label ${label} -folder ${folder_name} -pes_param ${param1} ${param2} ${param3} -pile_lambda ${pile_lambda}


python ${dir}/PISC/tools/average_tcf.py -pre ${method}  -folder ${folder_name}
mv average.out ${folder_name}
python ${dir}/PISC/tools/rotate_time.py ${folder_name}/average.out 2 0.1 aux.dat
mv aux.dat ${folder_name}/rot_average.out
./get_spectra.sh ${folder_name} rot_average.out ${ndim}
