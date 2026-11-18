import numpy as np
from matplotlib import pyplot as plt
from PISC.potentials import double_well, quartic, harmonic1D, mildly_anharmonic
from scipy import optimize as opt

"""
Module for propagating classical dynamics of a driven system using Velocity verlet.
We assume a Hamiltonian of the form:
        H(x,p,t) = p^2/(2m) + V(x) + x*g*sin(w_F*t)
"""

class Integrator:
    def __init__(self, m, pes, g, w_F, dt, x_init, p_init, t_init=0.0):
        self.m = m
        self.pes = pes
        self.g = g
        self.w_F = w_F
        self.x = x_init
        self.p = p_init
        self.t = t_init
        self.T = 2*np.pi / self.w_F  # Period of the driving force
        self.n_steps_T = round(self.T / dt)  # Number of steps in one period
        self.dt = self.T / self.n_steps_T  # Adjust dt to fit exactly into one period

    def B(self):
        # Half step for momentum
        self.p -= 0.5 * self.dt * (self.pes.dpotential(self.x) + self.g * np.sin(self.w_F * self.t))

    def A(self):
        # Full step for position
        self.x += self.dt * (self.p / self.m)

    def step(self):
        self.B()
        self.A()
        self.t += self.dt
        self.B()

    def Tstep(self):
        # Perform a full step of size T (the period of the driving force)
        for n in range(self.n_steps_T):
            self.step()

    def strob_section(self, nT):
        # Store the stroboscopic points
        x_strob = []
        p_strob = []
        for n in range(nT):
            self.Tstep()
            #print('t', self.t, n*self.T)
            x_strob.append(self.x)
            p_strob.append(self.p)
        return np.array(x_strob), np.array(p_strob)

class hw_Integrator:
    def __init__(self, m, g, w_F, dt, L, x_init, p_init, t_init=0.0):
        self.m = m
        self.g = g
        self.w_F = w_F
        self.dt = dt
        self.x = x_init #It needs to be a scalar here, because of the root finding algorithm.
        self.p = p_init
        self.t = t_init
        self.T = 2*np.pi / self.w_F
        self.L = L  # Length of the box
        self.lb = -L/2  
        self.ub = L/2
        self.coll_times = []
        self.tol = 1e-6  # Tolerance for boundary checks
    
    def f_boundary(self, x):
        # Note: this is for a 1D box between -L/2 and L/2.
        return abs(x) - self.L/2

    def x_free(self, x, delta_t):
        # Free-space solution
        #Check if any of the dataype is None
        return self.x + self.g/(self.m * self.w_F**2) * (np.sin(self.w_F * (self.t + delta_t)) - np.sin(self.w_F * self.t)) +\
                (self.p - self.g/(self.m * self.w_F) * np.cos(self.w_F * self.t)) / self.m * delta_t

    def p_free(self, p, delta_t):
        # Free-space solution
        return self.p + self.g/(self.w_F) * (np.cos(self.w_F * (self.t + delta_t)) - np.cos(self.w_F * self.t))

    def f_coll(self, delta_t):
        return self.f_boundary(self.x_free(self.x, delta_t))

    def time_to_collision(self, t_max = 1e6, t_min=0.0):
        # Check if we are already outside the boundary
        #print('func(x)', self.f_boundary(self.x))
        if self.f_boundary(self.x) > self.tol:
            print("Current position:", self.x)
            raise ValueError("Particle is outside the boundary!")
        # Check if we are already at the boundary
        if self.f_boundary(self.x) == 0.0:
            print("Current position:", self.x)
            return 0.0

        print('t_curr', self.f_boundary(self.x), self.f_boundary(self.x_free(self.x, t_max)))

        # Use scipy's root finding algorithm to find the collision time
        t_coll = opt.root_scalar(self.f_coll, bracket=[t_min, t_max], method='bisect').root

        #Check if f_coll(t_coll) is slightly BELOW zero
        if self.f_coll(t_coll) >0:
            t_coll-= self.tol/4
        return t_coll

    def update_col(self, t_coll, t_remaining):
        # Update position and momentum at collision time
        t_coll -= self.tol/4
        self.x = self.x_free(self.x, t_coll)
        self.p = self.p_free(self.p, t_coll)
        self.t += t_coll
        self.p = -self.p
        self.coll_times.append(self.t)
        return t_remaining - t_coll

    def Tstep_collision(self):
        """ Perform a full step of size T (the period of the driving force) with collision handling """
        # Check if we are already outside the boundary
        if self.f_boundary(self.x) > self.tol:
            print("Current position:", self.x)
            raise ValueError("Particle is outside the boundary!")
        if self.f_boundary(self.x) == 0.0:
            print("Current position:", self.x)
            raise ValueError("Particle is at the boundary!")

        t_remaining = self.T
        t_init = self.t
        #while self.t-t_init < self.T:
        while t_remaining > 0:
            tp_crit = np.cos(self.w_F*self.t)-self.p / (self.g) * self.w_F
            if abs(tp_crit) > 1: # x(t) is monotonic, so we can find collision time directly
                #print('tp_crit > 1', tp_crit)
                if(self.f_coll(0) * self.f_coll(t_remaining) < 0):
                    #print('here')
                    t_coll = opt.root_scalar(self.f_coll, bracket=[0, t_remaining], method='bisect').root
                    t_remaining = self.update_col(t_coll, t_remaining)
                else:
                    #print('there')
                    self.x = self.x_free(self.x, t_remaining)
                    self.p = self.p_free(self.p, t_remaining)
                    self.t += t_remaining
                    t_remaining = 0
            else: # x(t) is non-monotonic, so we need to check segments
                #print('tp_crit < 1', tp_crit)
                phi = (self.w_F * self.t) % (2 * np.pi)
                acos_val = np.arccos(tp_crit)

                t_tp1 = ((acos_val - phi) % (2 * np.pi)) / self.w_F
                t_tp2 = (((2 * np.pi - acos_val) - phi) % (2 * np.pi)) / self.w_F 

                t_tp_min = min(t_tp1, t_tp2)
                t_tp_max = max(t_tp1, t_tp2)
                
                tp_min_valid = (t_tp_min >= 0) and (t_tp_min <= t_remaining)
                tp_max_valid = (t_tp_max >= 0) and (t_tp_max <= t_remaining)
                #print('t_tp_min', t_tp_min, 't_tp_max', t_tp_max, 'tp_min_valid', tp_min_valid, 'tp_max_valid', tp_max_valid)
                #print('p_min', self.p_free(self.p, t_tp_min), 'p_max', self.p_free(self.p, t_tp_max))
                if(self.f_coll(0) * self.f_coll(t_tp_min) < 0 and tp_min_valid):
                    t_coll = opt.root_scalar(self.f_coll, bracket=[0, t_tp_min], method='bisect').root
                    t_remaining = self.update_col(t_coll, t_remaining)
                elif(self.f_coll(t_tp_min) * self.f_coll(t_tp_max) < 0 and tp_max_valid and tp_min_valid):
                    t_coll = opt.root_scalar(self.f_coll, bracket=[t_tp_min, t_tp_max], method='bisect').root
                    t_remaining = self.update_col(t_coll, t_remaining)
                elif(self.f_coll(t_tp_max) * self.f_coll(t_remaining) < 0 and tp_max_valid):
                    t_coll = opt.root_scalar(self.f_coll, bracket=[t_tp_max, t_remaining], method='bisect').root
                    t_remaining = self.update_col(t_coll, t_remaining)
                elif(self.f_coll(0) * self.f_coll(t_remaining) < 0 and tp_min_valid and not tp_max_valid):
                    t_coll = opt.root_scalar(self.f_coll, bracket=[0, t_remaining], method='bisect').root
                    t_remaining = self.update_col(t_coll, t_remaining) 
                else:
                    self.x = self.x_free(self.x, t_remaining)
                    self.p = self.p_free(self.p, t_remaining)
                    self.t += t_remaining
                    t_remaining = 0
        #print('self.t', self.t-t_init, self.T)

    def Tstep(self):
        """ Perform a full step of size T (the period of the driving force) """
        if self.x < self.lb or self.x > self.ub:
            # Raise error if the particle is outside the boundary
            raise ValueError(f"Particle is outside the boundary at time {self.t:.2f}. Position: {self.x:.2f}, Momentum: {self.p:.2f}")
        
        t_remaining = self.T
        x_current = self.x
        p_current = self.p

        while t_remaining > 0:
            t_coll = self.time_to_collision_direct()  # Find time to next collision
            if t_coll > t_remaining:
                print('t_coll-t_remaining', t_coll-t_remaining)
                print('t_coll', t_coll, 't_remaining', t_remaining)
                # No collision within remaining time, propagate freely
                self.x = self.x_free(x_current, t_remaining)
                self.p = self.p_free(p_current, t_remaining)
                self.t += t_remaining
                x_current = self.x
                p_current = self.p
                print(f"Step completed without collision. New position: {self.x}, New momentum: {self.p}, Time: {self.t}")
                t_remaining = 0
            else:
                print('Collision detected! t_coll:', t_coll, 't_remaining:', t_remaining)
                # Collision happens within remaining time, propagate to collision point and reflect
                self.x = self.x_free(x_current, t_coll)
                self.p = self.p_free(p_current, t_coll)
                self.t += t_coll
                self.p = -self.p  # Reflect momentum, NOTE: for a 2D potential, this would need to be modified to reflect the appropriate component of momentum
                x_current = self.x
                p_current = self.p
                t_remaining -= t_coll
                self.coll_times.append(self.t)
        print('\n')
        
    def strob_section(self, nT):
        """ Store the stroboscopic points """
        x_strob = []
        p_strob = []
        for n in range(nT):
            self.Tstep()
            x_strob.append(self.x)
            p_strob.append(self.p)
        return np.array(x_strob), np.array(p_strob)

if(0): # Test 1D box
    m = 1.0
    g = 3.0
    w_F = 12.25# 19.75
    dt = 0.005
    L = 2.0
    E1 = np.pi**2/(2*m*L**2)  # Ground state energy of a particle in a box
    En = np.arange(1, 25)**2 * E1  # Energy levels for the particle in the box 
    print('w_F', w_F,'E3-E1', En[2]-En[0],'E4-E2', En[3]-En[1],'E5-E3', En[4]-En[2], 'E6-E4', En[5]-En[3], 'E7-E5', En[6]-En[4])
    print('w_F', w_F, 'E2-E1', En[1]-En[0], 'E3-E2', En[2]-En[1], 'E4-E3', En[3]-En[2], 'E5-E4', En[4]-En[3], 'E6-E5', En[5]-En[4])

    E_tot = En[3]  # Total energy for the particle in the box
    rng = np.random.default_rng(seed=100)
    x_init_arr = rng.uniform(-L/2+0.05, L/2-0.05,20)  # Sample initial position from a uniform distribution
    p_init_arr = np.sqrt(2 * m * E_tot - 2 * m * 0.0)*np.ones_like(x_init_arr)  # Sample initial momenta based on energy conservation (V(x) = 0 inside the box)
    
    nT = 200
    #x_init = x_init_arr[28]
    #p_init = p_init_arr[28]

    for x_init, p_init in zip(x_init_arr, p_init_arr):
        x_list = []
        p_list = []
        hw_int = hw_Integrator(m, g, w_F, dt, L, x_init, p_init)    
        for n in range(nT):
            hw_int.Tstep_collision()
            #print(f"Step {n+1}: x = {hw_int.x}, p = {hw_int.p}, t = {hw_int.t}")
            x_list.append(hw_int.x)
            p_list.append(hw_int.p)
        
        plt.scatter(x_list, p_list, color='blue', alpha=0.9, s=4)
    
    plt.xlim(-L/2, L/2)
    plt.ylim(-1.5*np.sqrt(2 * m * E_tot), 1.5*np.sqrt(2 * m * E_tot))
    #plt.ylim(-6,6)

    #plt.figure(figsize=(8, 6))
    plt.axhline(np.sqrt(2 * m * E_tot), color='red', linestyle='--', label='p = sqrt(2mE)')
    plt.axhline(-np.sqrt(2 * m * E_tot), color='red', linestyle='--')
    plt.xlabel("Position (x)")
    plt.ylabel("Momentum (p)")
    plt.title("Phase Space Trajectory for Particle in a Box")
    
    plt.show()
    
    if(0):
        x_all = []
        p_all = []
        for x_init, p_init in zip(x_init_arr, p_init_arr):
            hw_int = hw_Integrator(m, g, w_F, dt, L, x_init, p_init)
            for n in range(nT):
                hw_int.Tstep()
                x_all.append(hw_int.x)
                p_all.append(hw_int.p)

        x_all = np.array(x_all)
        p_all = np.array(p_all)
        print('x_all', x_all.shape, 'p_all', p_all.shape)
        exit()
    
    exit()

if(0): # Test harmonic potential
    m = 1.0
    omega = 1.0
    a = 0.0
    b = 0.0

    pes = mildly_anharmonic(m,a,b,w=omega,n=4)

    potkey = 'MAH_w_{}_a_{}_b_{}_n_{}'.format(omega,a,b,4)

    x_max = 2
    E_tot = pes.potential_func(x_max)
    rng = np.random.default_rng(seed=42)
    x_init = rng.uniform(-x_max, x_max)  # Sample initial positions from a uniform distribution
    p_init = np.sqrt(2 * m * E_tot - 2 * m * pes.potential_func(x_init))  # Sample initial momenta based on energy conservation

    g_opt=3.0
    w_F = 10.0
    harm_int = Integrator(m, pes, g_opt, w_F, dt=0.005, x_init=x_init, p_init=p_init)
    nT = 100

    A = x_init
    B = p_init/(m*omega) - g_opt *w_F/omega * 1/(m*(w_F**2-omega**2))
    for n in range(nT):
        harm_int.Tstep()
        x_anal = A * np.cos(omega*harm_int.t) + B * np.sin(omega*harm_int.t) + g_opt/(m*(w_F**2-omega**2)) * np.sin(w_F*harm_int.t)
        p_anal = -A * m * omega * np.sin(omega*harm_int.t) + B * m * omega * np.cos(omega*harm_int.t) + g_opt * w_F/(w_F**2-omega**2) * np.cos(w_F*harm_int.t)
        print(f"Step {n+1}: x = {harm_int.x}, p = {harm_int.p}, t = {harm_int.t}")
        print(f"Analytical: x = {x_anal}, p = {p_anal}")
        
        print('% diff', 100*abs(harm_int.x - x_anal)/abs(x_anal), 100*abs(harm_int.p - p_anal)/abs(p_anal),'\n')

    exit()

if(0): # Test quartic potential
    m = 1.0
    n_anharm = 4
    omega = 0.0
    a = 0.0
    b = 1.0

    pes = mildly_anharmonic(m,a,b,w=omega,n=n_anharm)

    potkey = 'MAH_w_{}_a_{}_b_{}_n_{}'.format(omega,a,b,n_anharm)

    x_max = 3.
    E_tot = pes.potential_func(x_max)
    E_tot = 30.0
    x_max = (E_tot/b)**(1/n_anharm)  # Calculate the maximum x based on the total energy and potential parameters
    print('E_tot', E_tot)
    rng = np.random.default_rng(seed=42)

    x_init_arr = rng.uniform(-x_max, x_max, 20)  # Sample initial positions from a uniform distribution
    p_init_arr = np.sqrt(2 * m * E_tot - 2 * m * pes.potential_func(x_init_arr))  # Sample initial momenta based on energy conservation

    g_opt=3.0
    w_F = 10.0
    #quart_int = Integrator(m, pes, g_opt, w_F, dt=0.005, x_init=x_init, p_init=p_init)
    
    nT = 200
    #x_init = x_init_arr[28]
    #p_init = p_init_arr[28]

    for x_init, p_init in zip(x_init_arr, p_init_arr):
        x_list = []
        p_list = []
        quart_int = Integrator(m, pes, g_opt, w_F, dt=0.001, x_init=x_init, p_init=p_init)
        for n in range(nT):
            quart_int.Tstep()
            print(f"Step {n+1}: x = {quart_int.x}, p = {quart_int.p}, t = {quart_int.t}")
            x_list.append(quart_int.x)
            p_list.append(quart_int.p)
        
        plt.scatter(x_list, p_list, color='blue', alpha=0.9, s=7)
    
    plt.xlim(-1.3*x_max, 1.3*x_max)
    plt.ylim(-1.5*np.sqrt(2 * m * E_tot), 1.5*np.sqrt(2 * m * E_tot))
    
    #plt.figure(figsize=(8, 6))
    #plt.axhline(np.sqrt(2 * m * E_tot), color='red', linestyle='--', label='p = sqrt(2mE)')
    #plt.axhline(-np.sqrt(2 * m * E_tot), color='red', linestyle='--')
    
    x_arr = np.linspace(-1*x_max, 1.*x_max, 101)
    y_arr = np.sqrt(2 * m * E_tot - 2 * m * pes.potential_func(x_arr))
    plt.plot(x_arr, y_arr, color='red', linestyle='--', label='p = sqrt(2m(E-V(x)))')
    plt.plot(x_arr, -y_arr, color='red', linestyle='--')    


    plt.xlabel("Position (x)")
    plt.ylabel("Momentum (p)")
    plt.title("Phase Space Trajectory for Quartic Potential")
    
    plt.show()
    exit()

    plt.figure(figsize=(8, 6))
    for n in range(nT):
        quart_int.Tstep()
        #print(f"Step {n+1}: x = {quart_int.x}, p = {quart_int.p}, t = {quart_int.t}")
        print('E', quart_int.pes.potential_func(quart_int.x) + quart_int.p**2/(2*quart_int.m))
        plt.clf()
        plt.scatter(quart_int.x, quart_int.p, color='blue', alpha=0.9, s=10)
        #plt.xlim(-5, 5)
        #plt.ylim(-5, 5)
        plt.pause(0.1)

    plt.title("Phase Space Trajectory for Quartic Potential")
    plt.xlabel("Position (x)")
    plt.ylabel("Momentum (p)")
    plt.grid()
    plt.show()



