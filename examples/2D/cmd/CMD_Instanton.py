import numpy as np
import nlopt
from copy import deepcopy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Instanton_configuration(_sim,fname):
    global potglob, count
    potglob = 0.0
    count = 0
	rp_arr = []
    def function_to_optimise(_rp, grad):
        global potglob, count
        _sim.rp.qcart = _rp.reshape(_sim.rp.qcart.shape)
        _sim.rp.q = _sim.rp.nmtrans.cart2mats(_sim.rp.qcart, _sim.rp.q)
        if grad.size > 0:
            # Update the force and potential energy for current configuration
            _sim.pes.update()
            # Get forces
            force = _sim.pes.fcart 
            spring_force = -_sim.rp.m3*_sim.rp.freqs2*_sim.rp.q  
            spring_force = _sim.rp.nmtrans.mats2cart(spring_force)
            assert(force.shape == spring_force.shape) 
            force += spring_force 
            grad[:] = -force.flatten()
        # Get potential
        potential = np.sum(_sim.pes.pot) + _sim.rp.get_spring_tot()[0]  # PES potential + springs
        if(count%20==0):   
            print('count, potential', count, potential)        
 
		count+=1
        return potential

    def cmd_constraint(_rp, grad, dim, atom):
        global count
        if grad.size > 0:
            temp = np.zeros(_sim.rp.qcart.shape)  
            temp[atom, dim, :] = 1.0 / number_of_beads
            grad[:] = temp.flatten()
		# Centroid constraint
        _rp = _rp.reshape(_sim.rp.qcart.shape)
        return np.mean(_rp, axis=2)[atom,dim] - centroid[atom,dim]

    # Save initial value of qcart, needed to enforce centroid constraint
    _sim.pes.bind(_sim.rp.ens, _sim.rp)
    number_of_beads = _sim.rp.qcart.shape[2]
    centroid = np.mean(_sim.rp.qcart, axis=2)
     
    qcart_initial_flat = _sim.rp.qcart.reshape(-1,)  # flatten causes them not to share memory
    lower_bound = np.array([-10*abs(np.min(qcart_initial_flat)) for i in range(len(qcart_initial_flat))])
    upper_bound = np.array([10*abs(np.max(qcart_initial_flat)) for i in range(len(qcart_initial_flat))])
    opt = nlopt.opt(nlopt.LD_SLSQP, _sim.rp.qcart.size)
    opt.set_lower_bounds(lower_bound)
    opt.set_upper_bounds(upper_bound)
    opt.set_min_objective(function_to_optimise)
    opt.add_equality_constraint(lambda x, grad: cmd_constraint(x, grad, 0, 0), 1e-8)
    opt.add_equality_constraint(lambda x, grad: cmd_constraint(x, grad, 0, 1), 1e-8)
        
	opt.set_maxeval(20000)
    opt.set_xtol_rel(1e-12)
    opt.set_ftol_rel(1e-12)
    output = opt.optimize(qcart_initial_flat)
    min_func = opt.last_optimum_value()
                             
    print('x, minf',np.mean(output.reshape(_sim.rp.qcart.shape),axis=1),centroid,min_func, opt.last_optimize_result())
    return output.reshape(_sim.rp.qcart.shape)
