# PISC

This repository mainly contains the numerical tools required to propagate classical dynamical variables. Specifically it can propagate the ‘monodromy matrix’ in time, which measures the deviation of nearby trajectories as a function of time - this is useful, for instance, in chaos theory, spectroscopic response functions and wavepacket reconstruction. The code also has provisions
to propagate the dynamical variables of 'ring-polymers' which are used to approximate
the quantum dynamics of a Boltzmann distributed quantum state.

The 'core' files of the repository are the following:
1. beads.py
2. integrators.py
3. ensemble.py
4. simulation.py
5. thermostat.py
6. motion.py
7. PI_sim_core.py

Please see the documentation in each of the files above to understand more about how the
code is structured. To start a simulation, a simulation file needs to be created to specify the system, ensemble and simulation method. An example of a simulation file is provided under:
        	.	../examples/2D/rpmd/RPMD_simfile.py

The (ring-polymer) dynamics is obtained by symplectic integration of Newton's
equations of motion (order 2 and order 4 are available for now) and is implemented using steps
B, A, O and Mi that update momenta, position and monodromy matrix elements. The positions and momenta are updated in both Cartesian and ‘normal mode’ coordinates (specific to ring-polymer propagation). 

By default the simulation runs entirely in python, but that can be changed simply by setting
'fort=True' in ...set_simparams(fort=True) (see the aforementioned simulation file). Setting fort=True makes the following changes by default:
1. The steps B, b, A, O and Mi (i=1,2,3,4) are implemented in fortran (a reference of the python variables is passed to fortran, so arrays are not copied to make them fortran-contiguous).
2. The potential, dpotential and ddpotential functions (to compute the potential, forces and hessian from the PES) are implemented in fortran.
3. The transformation from Cartesian to normal modes is implemented in fortran.

Note that a fortran file needs to be written and imported into fortran when a new PES is to be added to the codebase. This is straightforward to do and a model file (called base_fort.f90) is available under PISC/PISC/potentials/. The convention to import the fortran module and calling fortran functions from python is given in harmonic_2D.py

The code has been tested thoroughly and it is expected to be accurate. The following test files are most helpful to diagnose the code in the event of a bug (all available under PISC/test_files/):
test_fort_integrator.py
test_fort_utils.py
test_fort_monodromy.py
test_fort_datafiles.py

Some errors may still be present - this is because there are some minor changes to the code structure in the latest commit. However, these errors are expected to be nothing more than minor issues with importing class objects and can be easily fixed. Also, no error is expected in the core files anyway. 

The code can be easily parallelised in python using the multiprocessing library and it is already implemented in the example code mentioned before. Note that it is also possible to parallelise the code from within fortran using openMP: this can be done by uncommenting the lines starting with ‘!omp’ in integrator_f.f90, nmtrans_f.f90, misc_f.f90 and all PES modules implemented in fortran (see base_fort.f90 for reference). 

However, openMP parallelisation is only helpful when a large number of particles/systems are propagated together or in the presence of a complicated force field (with interparticle interactions). For small numbers of trajectories, the overhead coming from the f2py interface is too large to allow any fortran advantage. Hence, it is preferable to use python mode for simulating classical dynamics and fortran mode for simulating ring-polymer dynamics.

The following command is to be used to generate the .so bridge files for calling fortran functions from python:
f2py3 -c --f90flags="-fopenmp" -m mod_f mod_f.f90  -lgomp

where mod_f.f90 is the fortran file with the desired functions and mod is the name of the python module to generate from it.  This is available in a script file in the main folder (PISC) - just executing “sh fort.sh” inside PISC/engine/ and PISC/utils/  will generate the necessary files for enabling fortran mode. However, when a new PES is written in fortran, the .so bridge file needs to be generated for it manually (a reference for the same is provided in PISC/potentials). 
# PISC
