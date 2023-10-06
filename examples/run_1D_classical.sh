
dim=1
pes='double_well'
mass=0.5
sys_name='selene'
method='Classical'
ens='thermal'
temp=20 #Fake
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
lambda=0
g=0.02

python  main.py  -dim ${dim} -pes ${pes} -m ${mass} -sys_name ${sys_name} -method ${method} -ens ${ens} -temp ${temp} -temp_tau ${temp_tau} -dt ${dt} -dt_therma ${dt_therma} -time_therma ${time_therma} -time_total ${time_total} -n_traj ${n_traj} -corr_func ${corr_func} -nbeads ${nbeads} -nseeds ${nseeds} -chunk ${chunk_size} -label ${label} -folder ${folder_name} -pes_param ${lambda} ${g} 

