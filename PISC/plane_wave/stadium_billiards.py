import numpy as np
from scipy.linalg import svd

"""
This code computes the eigenstates of a stadium billiard by quantising the boundary, 
i.e. by using plane-wave expansions of the wavefunction and enforcing the boundary conditions 
at a finite number of points on the boundary. In this module, we solve for a quarter billiard
and then use symmetry to construct the full eigenstates. 
"""

class StadiumBilliard:
    def __init__(self, a, r, num_points=None):
        self.a = a  # Length of the straight sections
        self.r = r  # Radius of the semicircular sections
        self.num_points = num_points  # Number of points to discretize the boundary

    def boundary_points(self,M=None):
        """Generate points on the boundary of the quarter stadium billiard."""
        points = []
        if M is not None:
            self.num_points = M
        else:
            M = self.num_points
        Mrect = M // 2  # Number of points for the straight section
        # Straight section along x-axis
        for i in range(1,Mrect+1):
            #We avoid the point x=0 to prevent singularities in the plane wave expansion
            x = self.a * i / Mrect  
            points.append((x, self.r))
        
        Mcurv = M - Mrect  # Remaining points for the semicircular section
        # Semicircular section
        for i in range(1,Mcurv+1):
            #We avoid theta=0 and pi/2 to prevent singularities in the plane wave expansion
            theta = np.pi/2 * (1 - i/(Mcurv+1))
            x = self.a + self.r * np.cos(theta)
            y = self.r * np.sin(theta)
            points.append((x, y))
        
        return np.array(points)

    def boundary_points_box(self,M=None):
        """Generate points on the boundary of a rectangular box of size a x r."""
        points = []
        if M is not None:
            self.num_points = M
        else:
            M = self.num_points
        Mx = M // 2  # Number of points for the x-direction
        My = M - Mx  # Remaining points for the y-direction
        # Straight section along x-axis
        for i in range(1,Mx+1):
            x = self.a * i / Mx
            points.append((x, self.r))
        # Straight section along y-axis
        for i in range(1,My+1):
            y = self.r * i / My
            points.append((self.a, y))

        return np.array(points) 

    def DD_wf(self, x, y, j, k, N):
        """Dirichlet wavefunction for the j-th mode."""
        # We use half-integer values of j to avoid singularities from theta_j=0,pi/2
        theta_j = np.pi * (j+0.5) / (2 * N)
        return np.sin(k*x*np.cos(theta_j)) * np.sin(k*y*np.sin(theta_j))

    def DN_wf(self, x, y, j, k, N):
        """Mixed Dirichlet-Neumann wavefunction for the j-th mode."""
        theta_j = np.pi * j / (2 * N)
        return np.sin(k*x*np.cos(theta_j)) * np.cos(k*y*np.sin(theta_j))

    def ND_wf(self, x, y, j, k, N):
        """Mixed Neumann-Dirichlet wavefunction for the j-th mode."""
        # We use half-integer values of j to avoid singularities from theta_j=0,pi/2
        theta_j = np.pi * (j+0.5) / (2 * N)
        return np.cos(k*x*np.cos(theta_j)) * np.sin(k*y*np.sin(theta_j))

    def NN_wf(self, x, y, j, k, N):
        """Neumann wavefunction for the j-th mode."""
        theta_j = np.pi * j / (2 * N)
        return np.cos(k*x*np.cos(theta_j)) * np.cos(k*y*np.sin(theta_j))

    def plane_wave_expansion(self, k, bdtype='DD',N=None):
        """
        Construct the plane wave expansion for a given wave number k.
        1. We create a matrix B where each row corresponds to a boundary point 
           and each column corresponds to a plane wave mode.
        2. We find non-trivial solutions to the homogeneous system B * c = 0, 
           which correspond to a linear combination of plane waves that
           satisfies the boundary conditions at the chosen points.
        3. We use SVD decomposition to find the minimum singular value of B, 
           which indicates how close we are to an eigenstate (the smaller, the closer).
        4. We use Dirichlet (DD), Neumann (NN), or mixed (DN, ND) wavefunctions for the plane wave expansion,
           depending on the specified boundary conditions.
        """
        #M = self.num_points
        L = self.a + np.pi * self.r/2 # Perimeter of the quarter stadium
        if N is None:
            N = int(1.2*k*L/2)   # Number of plane waves (can be adjusted)
        M = 3*N  # Number of boundary points (can be adjusted)
        #points = self.boundary_points_box(M)  # Get boundary points (using box for simplicity)
        points = self.boundary_points(M)  # Get boundary points
        B = np.zeros((M, N))
        
        for i in range(M):
            for j in range(N):
                # Plane wave components
                if bdtype == 'DD':
                    B[i, j] = self.DD_wf(points[i][0], points[i][1], j, k, N)
                elif bdtype == 'DN':
                    B[i, j] = self.DN_wf(points[i][0], points[i][1], j, k, N)
                elif bdtype == 'ND':
                    B[i, j] = self.ND_wf(points[i][0], points[i][1], j, k, N)
                elif bdtype == 'NN':
                    B[i, j] = self.NN_wf(points[i][0], points[i][1], j, k, N)
        return B, M

    def solve_eigenstates(self, k_values, bdtype='DD',N=None):
        """Solve for the eigenstates at given wave numbers."""
        eigenstates = []
        eigenvectors = []
        for k in k_values:
            #ktarg = 7#min(k_values) + (max(k_values) - min(k_values))*0.1   # Target wave number for determining N
            #N = int(1.2*ktarg*(self.a + np.pi * self.r/2)/2)  # Number of plane waves (can be adjusted)
            if N is None:
                N = int(min(k_values)) + 2 
            print('N', N)
            Bk, Mk = self.plane_wave_expansion(k, bdtype, N=N)
            # Solve the homogeneous system Bk * c = 0 for non-trivial solutions
            #S = svd(Bk, compute_uv=False)
            #eigenstates.append(np.min(S)/Mk**0.5)
            
            U, S, Vh = svd(Bk)
            argmin = np.argmin(S)
            eigenstates.append(S[argmin]/Mk**0.5)  # Store the minimum singular value as a measure of how close we are to an eigenstate
            eigenvectors.append(Vh[argmin])
        return eigenstates, eigenvectors


