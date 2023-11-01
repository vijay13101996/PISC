
sys_name='Tosca2'
method='Classical'
corr_func='R3eq'

pes='mildly_anharmonic'
dim=1
param1=1.0 #m
param2=${1} #a
param3=${2} #b
mass=1.0
nbeads='1'


ens='thermal'
temp=0.250
temp_tau=1.0
pile_lambda=1000.0
dt_therma=0.05
time_therma=100.0

dt=0.01
#time_total=50
#ndim=201

time_total=25
ndim=101

dt_tcf=0.50

n_traj=1000
nseeds=064
chunk_size=10
label='RPMD'
folder_name="Datafile_${method}_${corr_func}_nbeads${nbeads}_nseeds${nseeds}_temp${temp}_a${param2}"

dir=/scratch2/yl899/gitlab/PISC

#----------------------------------------
python  ${dir}/examples/examples_main.py  -dim ${dim} -pes ${pes} -m ${mass} -sys_name ${sys_name} -method ${method} -ens ${ens} -temp ${temp} -temp_tau ${temp_tau} -dt ${dt} -dt_therma ${dt_therma} -time_therma ${time_therma} -time_total ${time_total} -n_traj ${n_traj} -corr_func ${corr_func} -nbeads ${nbeads} -nseeds ${nseeds} -chunk ${chunk_size} -label ${label} -folder ${folder_name} -pes_param ${param1} ${param2} ${param3} -pile_lambda ${pile_lambda}


 python ${dir}/PISC/tools/average_tcf.py -pre ${method}  -folder ${folder_name}
 mv average.out ${folder_name}
# python ${dir}/PISC/tools/rotate_time.py ${folder_name}/average.out 3 0.5 aux.dat
# mv aux.dat ${folder_name}/rot_average.out
 #python ${dir}/PISC/tools/get_slice.py  ${folder_name}/average.out t2 0 aux.dat
 #mv aux.dat ${folder_name}/slice_t2eq0_average.out

 #./get_spectra_2D.sh ${folder_name} slice_t2eq0_average.out ${ndim} ${dt_tcf} &
 ./get_spectra_3D.sh ${folder_name} average.out ${ndim} ${dt_tcf}
 mv FFT_w21.0.png FFT_a${param2}_${nseeds}.png
