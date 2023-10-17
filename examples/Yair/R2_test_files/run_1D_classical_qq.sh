
sys_name='Tosca2'
method='Classical'
corr_func='qq_TCF'

pes='mildly_anharmonic'
dim=1
param1=1.0 #m
param2=0.4 #a
param3=0.16 #b
mass=1.0
nbeads='1'


ens='thermal'
temp=1.0
temp_tau=1.0
pile_lambda=1000.0
dt_therma=0.05
time_therma=50.0

dt=0.01
time_total=20.0

n_traj=1000
nseeds=10
chunk_size=10
label='RPMD'
folder_name='classical_qq'

dir=/scratch2/yl899/gitlab/PISC/examples
python  ${dir}/examples_main.py  -dim ${dim} -pes ${pes} -m ${mass} -sys_name ${sys_name} -method ${method} -ens ${ens} -temp ${temp} -temp_tau ${temp_tau} -dt ${dt} -dt_therma ${dt_therma} -time_therma ${time_therma} -time_total ${time_total} -n_traj ${n_traj} -corr_func ${corr_func} -nbeads ${nbeads} -nseeds ${nseeds} -chunk ${chunk_size} -label ${label} -folder ${folder_name} -pes_param ${param1} ${param2} ${param3} -pile_lambda ${pile_lambda}

