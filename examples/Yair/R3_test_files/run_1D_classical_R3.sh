
sys_name='Tosca2'
method='Classical'
corr_func='R3eq'

pes='mildly_anharmonic'
dim=1
param1=1.0 #m
param2=0.1 #a
param3=0.01 #b
mass=1.0
nbeads='1'


ens='thermal'
temp=1.0
temp_tau=1.0
pile_lambda=1000.0
dt_therma=0.05
time_therma=30.0

dt=0.01
time_total=30
ndim=121
time_total=50
ndim=201
dt_tcf=0.5

n_traj=1000
nseeds=10
chunk_size=10
label='RPMD'
folder_name="Datafile_${method}_${corr_func}_nbeads${nbeads}_time${time_total}_dt001"

dir=/scratch2/yl899/gitlab/PISC

#----------------------------------------
python  ${dir}/examples/examples_main.py  -dim ${dim} -pes ${pes} -m ${mass} -sys_name ${sys_name} -method ${method} -ens ${ens} -temp ${temp} -temp_tau ${temp_tau} -dt ${dt} -dt_therma ${dt_therma} -time_therma ${time_therma} -time_total ${time_total} -n_traj ${n_traj} -corr_func ${corr_func} -nbeads ${nbeads} -nseeds ${nseeds} -chunk ${chunk_size} -label ${label} -folder ${folder_name} -pes_param ${param1} ${param2} ${param3} -pile_lambda ${pile_lambda}


 python ${dir}/PISC/tools/average_tcf.py -pre ${method}  -folder ${folder_name}
 mv average.out ${folder_name}
 python ${dir}/PISC/tools/rotate_time.py ${folder_name}/average.out 3 0.5 aux.dat
 mv aux.dat ${folder_name}/rot_average.out
 python ${dir}/PISC/tools/get_slice.py  ${folder_name}/rot_average.out t2 0 aux.dat
 mv aux.dat ${folder_name}/slice_t2eq0_average.out

 ./get_spectra.sh ${folder_name} slice_t2eq0_average.out ${ndim} ${dt_tcf}
