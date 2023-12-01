import numpy as np

"""
Test file to ensure that the fortran code and python code are producing the same data.
To check this, first run examples/2D/rpmd/RPMD_simfile.py with the following parameters:
    double well 2D potential (called 'quartic_bistable' in RPMD_simfile.py)
    alpha = 0.382
    D = 9.375
    lamda = 2.0
    g = 0.08
    z = 1.0
    T = 2.0*Tc
    N = 1000
    dt = 0.02
    nbeads = 32
    seed = 0

    (The comparison can be done for any parameters, but these are the ones I used to test)
    (The seed is set to 0 to ensure that the random number generator is the same for both codes,
    any seed will work as long as it is the same for both)

    NOTE: To confirm that 'fortran' is indeed being used, print 'fort' in all if statements in 
    1. integrators.py (all steps of the integrator)
    2. Quartic_bistable.py
    3. misc.py (hess_mul function)

"""


path = '/home/vgs23/PISC/examples/2D/rpmd/Datafiles'

Seed = 4

fname = 'RPMD_thermal_OTOC_Selene_testpy_double_well_2D_alpha_0.382_D_9.375_lamda_2.0_g_0.08_z_1.0_T_2.0Tc_N_1000_dt_0.02_nbeads_32_seed_{}.txt'.format(Seed)
pydata = np.loadtxt(path+'/'+fname)

fname = 'RPMD_thermal_OTOC_Selene_testfort_double_well_2D_alpha_0.382_D_9.375_lamda_2.0_g_0.08_z_1.0_T_2.0Tc_N_1000_dt_0.02_nbeads_32_seed_{}.txt'.format(Seed)
fortdata = np.loadtxt(path+'/'+fname)

#print('pydata',pydata)
#print('fortdata',fortdata)

print('Comparing python and fortran datafiles...')
assert np.allclose(pydata,fortdata,atol=1e-10,rtol=1e-10)
print('Datafiles are the same!')
