
dim=1
pes='double_well'
mass=0.5
sys_name='selene'
method='Classical'
ens='thermal'
temp=6.36620 # 0.5*lambda/np.pi, with lambda=20
temp_tau=1
dt=0.005
dt_therma=0.05
time_total=5.0
time_therma=50.0
n_traj=20
corr_func='OTOC'
nbeads='1'
nseeds='100'
chunk_size='10'
label='classical'
folder_name='Datafile2'

param1=2 #lambda
param2=0.02 #g

python  examples_main.py  -dim ${dim} -pes ${pes} -m ${mass} -sys_name ${sys_name} -method ${method} -ens ${ens} -temp ${temp} -temp_tau ${temp_tau} -dt ${dt} -dt_therma ${dt_therma} -time_therma ${time_therma} -time_total ${time_total} -n_traj ${n_traj} -corr_func ${corr_func} -nbeads ${nbeads} -nseeds ${nseeds} -chunk ${chunk_size} -label ${label} -folder ${folder_name} -pes_param ${param1} ${param2}

