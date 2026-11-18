import numpy as np
import scipy.optimize as opt

"""
This code computes the classical dynamics of a particle moving in a 
hard wall potential V. To begin with, we will only consider cases
where V(x)= 0 except at the boundaries, where V(x) = infinity. 

Given 
    (x,v): position and velocity
    t    : final time
    func : function that defines the boundary, of the form f(r) = 0, where r = (x,y) (or r = (x,y,z) in 3D)
we want to compute the position and velocity at time t.

The algorithm is as follows:
1. Compute the time to the next collision with the boundary, t_coll.
2. If t_coll > t, then we can simply update the position and velocity using the
    free particle equations of motion, and we are done.
3. If t_coll <= t, then we need to update position and velocity at collision,
    and then repeat the process until we reach time t.

We will store the collision times in a class variable for reference.
For now, this will only be for classical dynamics, but in future, we will 
try to extend this to ring-polymer dynamics as well.

"""
class Integrator_HW:
    def __init__(self, func):
        self.func = func
        self.collision_times = []
        self.tol = 1e-8  # Tolerance for checking if we are at the boundary
 
    def bind(self, func, grad_func=None):
        self.func = func 
        if grad_func is not None:
            self.grad_func = grad_func
        else:
            return NotImplementedError("Gradient function is not implemented, please provide grad_func for reflection!")

    def time_to_collision(self, x, v, t_max=1e3):
        """

        This function computes the time to the NEXT collision with the boundary
        given the current position x and velocity v. We will use a simple 
        root-finding algorithm to find the time of collision. The function we want to
        find the root of is f(x + v*t) = 0, where f is the boundary function.

        We assume f(x)<0 inside the boundary and f(x)>0 outside the boundary. 
            1. If f(x)>0, then we need to raise an error. 
            2. If f(x)=0, then we are already at the boundary, so we can return t_coll = 0.
            3. If f(x)<0, then we need to find the time t_coll such that f(x + v*t_coll) = 0.

        Algorithm:
            1. We assume collision happens within an upper bound t_max. 
               (we can easily check if f(x+v*t_max)>0, then raise an error if not)
            2. If f(x)<0 and f(x+v*t_max)>0, we check if f(x+v*t_max/2)>0. 
            3. If yes, we further repeat to check if f(x+v*t_max/4)>0 until
               we find a time t such that f(x+v*t)>0 and f(x+v*(t/2))<0.
            4. Then we know collision happens between t/2 and t.
            5. We continue this further to converge to the collision time t_coll.

        In practice, we will do this with python's built-in root-finding algorithm!
        """
        # Check if we are already outside the boundary
        if self.func(x) > self.tol:
            print("Current position:", x)
            raise ValueError("Particle is outside the boundary!")
        # Check if we are already at the boundary
        if self.func(x) == 0:
            return 0.0
        # Check if we can find a collision within t_max
        if self.func(x + v*t_max) <= 0:
            raise ValueError("No collision found within t_max, please increase t_max!")
        
        # Define the function for root finding
        def f_coll(t):
            return self.func(x + v*t)
        print('f_coll(0)', f_coll(0), 'f_coll(t_max)', f_coll(t_max))
        # Use scipy's root finding algorithm to find the collision time
        t_coll = opt.root_scalar(f_coll, bracket=[0, t_max], method='bisect').root
        
        #Check if f_coll(t_coll) is slightly BELOW zero
        if f_coll(t_coll) >0:
            t_coll-= self.tol/4

        return t_coll
     
    def reflect(self, x, v):
        """
        This function reflects the velocity v at the boundary defined by func.
        We will compute the normal vector to the boundary at the point of collision, 
        and then reflect the velocity accordingly.

        The normal vector is simply the gradient of the boundary function at the point of collision.
        Since we already demand f(x)<0 inside and f(x)>0 outside, the normal vector will point outward from the boundary.
        The reflection of the velocity is given by:
            v_reflected = v - 2*(v . n)*n
        where n is the unit normal vector.

        """
        #Check if we are at the boundary
        if abs(self.func(x)) > self.tol:
            print('func(x)', self.func(x))
            raise ValueError("Particle is not at the boundary, cannot reflect!")
        # Compute the normal vector (gradient of the boundary function)
        n = self.grad_func(x)
        n = n / np.linalg.norm(n)  # Normalize the normal vector
        # Reflect the velocity
        v_reflected = v - 2 * np.dot(v, n) * n
        print('incoming x, v', x, v)
        print('normal vector', n)
        print('reflected velocity', v_reflected)
        return x, v_reflected

    def propagate(self, x, v, t, record_traj=False):
        """
        This function propagates the particle from (x,v) for time t 
        (assuming we start from time t=0), taking into account collisions with the boundary.
        """
        t_remaining = t
        x_current = x
        v_current = v
        if record_traj:
            traj = [(0, x_current, v_current)]

        while t_remaining > 0:
            t_coll = self.time_to_collision(x_current, v_current)
            if t_coll > t_remaining:
                # No collision within remaining time, propagate freely
                x_current = x_current + v_current * t_remaining
                t_remaining = 0

            else:
                # Collision happens within remaining time, propagate to collision point and reflect
                x_current = x_current + v_current * t_coll
                x_current, v_current = self.reflect(x_current, v_current)
                if record_traj:
                    traj.append((t, x_current, v_current))
                t_remaining -= t_coll
                self.collision_times.append(t - t_remaining)
        if record_traj:
            return x_current, v_current, traj
        else:
            return x_current, v_current

if(0):
    # Testing for a simple 1D box potential, where the boundary is defined by f(x) = x - L/2 and f(x) = -x - L/2
    L = 1.0
    def f(x):
        return abs(x) - L/2 

    def grad_f(x):
        if x >= 0:
            return np.array([1.0])
        else:
            return np.array([-1.0])

    integrator = Integrator_HW(f)
    integrator.bind(f, grad_f)
    x0 = 0.25
    v0 = -2.0
    t_final = 10.0
    x_final, v_final = integrator.propagate(x0, v0, t_final)
    coll_times = integrator.collision_times
    print("Collision times:", coll_times)
    print("Final position:", x_final)

if(0):
    # Testing for a simple 2D circular boundary, where the boundary is defined by f(x,y) = x^2 + y^2 - R^2
    R = 1.0
    def f(r):
        x, y = r
        return x**2 + y**2 - R**2

    def grad_f(r):
        x, y = r
        return np.array([2*x, 2*y])

    integrator = Integrator_HW(f)
    integrator.bind(f, grad_f)
    r0 = np.array([0.0, 0.5])
    v0 = np.array([-1.0, 0.5])
    #v0/= np.linalg.norm(v0)  # Normalize the velocity
    t_final = 20.0
    r_final, v_final, traj = integrator.propagate(r0, v0, t_final, record_traj=True)
    coll_times = integrator.collision_times
    print("Collision times:", coll_times)
    print("Final position:", r_final)

    #Plot the circle and the trajectory
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6,6))
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = R * np.cos(theta)
    y_circle = R * np.sin(theta)
    ax.plot(x_circle, y_circle, 'r-', label='Boundary')
    
    x_traj = [r[0] for t, r, v in traj]
    x_traj.append(r_final[0])
    y_traj = [r[1] for t, r, v in traj]
    y_traj.append(r_final[1])

    dx = np.diff(x_traj)
    dy = np.diff(y_traj)
    ax.quiver(x_traj[:-1], y_traj[:-1], dx, dy, angles='xy', scale_units='xy', scale=1, color='b', label='Trajectory')

    ax.scatter(x_traj, y_traj)

    #Draw normal vectors at collision points
    for t, r, v in traj:
        if abs(f(r)) < integrator.tol:
            n = grad_f(r)
            n = n / np.linalg.norm(n)
            ax.quiver(r[0], r[1], n[0], n[1], angles='xy', scale_units='xy', scale=0.5, color='g', label='Normal Vector')
    
    plt.show()

if(0): #Testing the code for a stadium billiards
    L = 1.0
    R = 0.5
    def f(r):
        x, y = r
        if abs(x) <= L/2:
            return abs(y) - R
        else:
            return (x - np.sign(x)*L/2)**2 + y**2 - R**2

    def grad_f(r):
        x, y = r
        if abs(x) <= L/2:
            return np.array([0.0, np.sign(y)])
        else:
            dx = 2*(x - np.sign(x)*L/2)
            dy = 2*y
            return np.array([dx, dy])

    integrator = Integrator_HW(f)
    integrator.bind(f, grad_f)
    r0 = np.array([0.0, 0.25])
    v0 = np.array([-1.0, 0.5])
    t_final = 50.0
    r_final, v_final, traj = integrator.propagate(r0, v0, t_final, record_traj=True)
    coll_times = integrator.collision_times
    print("Collision times:", coll_times)
    print("Final position:", r_final)
    
    import matplotlib.pyplot as plt
    mul = 4
    fig, ax = plt.subplots(figsize=(mul*(L+2*R), mul*(2*R)))
    # Plot the stadium boundary
    x = np.linspace(-L/2-R, L/2+R, 100)
    y_top = np.where(abs(x) <= L/2, R, np.sqrt(R**2 - (x - np.sign(x)*L/2)**2))
    y_bottom = np.where(abs(x) <= L/2, -R, -np.sqrt(R**2 - (x - np.sign(x)*L/2)**2))
    ax.plot(x, y_top, 'r-', label='Boundary')
    ax.plot(x, y_bottom, 'r-')
    

    x_traj = [r[0] for t, r, v in traj]
    x_traj.append(r_final[0])
    y_traj = [r[1] for t, r, v in traj]
    y_traj.append(r_final[1])   

    dx = np.diff(x_traj)
    dy = np.diff(y_traj)
    ax.quiver(x_traj[:-1], y_traj[:-1], dx, dy, angles='xy', scale_units='xy', scale=1, color='b', label='Trajectory')
    ax.scatter(x_traj, y_traj)

    for t, r, v in traj:
        if abs(f(r)) < integrator.tol:
            n = grad_f(r)
            n = n / np.linalg.norm(n)
            ax.quiver(r[0], r[1], n[0], n[1], angles='xy', scale_units='xy', scale=0.5, color='g', label='Normal Vector')


    plt.show()



