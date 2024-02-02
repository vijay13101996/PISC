import sympy
from sympy import *

x = sympy.Symbol('x')
y = sympy.Symbol('y')

if(0):
    ws = sympy.Symbol('self.ws')
    wu = sympy.Symbol('self.wu')
    lamda = sympy.Symbol('self.lamda')

    V = 0.5*wu**2*y**2 + 0.5*ws**2*x**2 + lamda*y**2*x
        
    print('V',V)
    print('Vx=',diff(V,x))
    print('Vy=',diff(V,y))
    print('Vxx=',diff(V,x,x))
    print('Vxy=',diff(V,x,y))
    print('Vyx=',diff(V,y,x))
    print('Vyy=',diff(V,y,y))

if(0):
    g = sympy.Symbol('self.g')
    lamda = sympy.Symbol('self.lamda')

    V = 0.1875*(g*x**3 - lamda*x)**2

    print('V',V)
    print('Vx',diff(V,x))
    print('Vxx',diff(V,x,x))

if(0):
    T11 = sympy.Symbol('self.T11')
    T12 = sympy.Symbol('self.T12')
    T21 = sympy.Symbol('self.T21')
    T22 = sympy.Symbol('self.T22')
    m = sympy.Symbol('self.m')
    omega1 = symbol.Symbol('self.omega1')
    omega2 = symbol.Symbol('self.omega2')
    xi  = T11*x + T12*y
    eta = T21*x + T22*y
            
    V = 0.5*m*omega1**2*xi**2 + 0.5*m*omega2**2*eta**2

    print('V',V)
    print('Vx=',diff(V,x))
    print('Vy=',diff(V,y))
    print('Vxx=',diff(V,x,x))
    print('Vxy=',diff(V,x,y))
    print('Vyx=',diff(V,y,x))
    print('Vyy=',diff(V,y,y))


if(0): ###Tanimura's system bath potential
    D = sympy.Symbol('self.D')  
    alpha = sympy.Symbol('self.alpha')  
    m = sympy.Symbol('self.m')  
    mb = sympy.Symbol('self.mb')    
    wb = sympy.Symbol('self.wb')    
    VLL = sympy.Symbol('self.VLL')  
    VSL = sympy.Symbol('self.VSL')  
    cb = sympy.Symbol('self.cb')        

    s = sympy.Symbol('s')
    b = sympy.Symbol('b')

    Vs = D*(1-exp(-alpha*s))**2
    Vc = VLL*s + 0.5*VSL*s**2
    Vsb = 0.5*mb*wb**2*(b - cb*Vc/(mb*wb**2))**2
    V = Vs + Vsb

    print('V',V)
    print('Vs=',diff(V,s))
    print('Vb=',diff(V,b))
    print('Vss=',diff(V,s,s))
    print('Vsb=',diff(V,s,b))
    print('Vbs=',diff(V,b,s))
    print('Vbb=',diff(V,b,b))

if(0):### Double-double well potential
    lamdax = sympy.Symbol('self.lamdax')    
    gx = sympy.Symbol('self.gx')
    lamday = sympy.Symbol('self.lamday')    
    gy = sympy.Symbol('self.gy')
    
    z = sympy.Symbol('self.z')

    quartx = (x**2 - lamdax**2/(8*gx))
    Vbx = lamdax**4/(64*gx)

    quarty = (y**2 - lamday**2/(8*gy))
    Vby = lamday**4/(64*gy)
        
    vx = gx*quartx**2
    vy = gy*quarty**2
    vxy = z*x**2*y**2

    V = vx + vy + vxy

    print('V',V)
    print('Vx =',diff(V,x))
    print('Vy =',diff(V,y))
    print('Vxx = ',diff(V,x,x))
    print('Vxy =',diff(V,x,y))
    #print('Vyx',diff(V,y,x))
    print('Vyy =',diff(V,y,y))

if(0):### DW_harm potential
    lamda = sympy.Symbol('self.lamda')
    g = sympy.Symbol('self.g')
    w = sympy.Symbol('self.w')
    z = sympy.Symbol('self.z')
    m = sympy.Symbol('self.m')
    T = sympy.Symbol('self.T')

    quartx = (x**2 - lamda**2/(8*g))
    vy = 0.5*m*w**2*y**2*T
    vx = g*quartx**2
    #vxy = (vx-lamda**4/(64*g))*(-z*y) 
    
    vxy = z*y**2*x**2
    V = vx + vy + vxy

    print('V',V)
    print('Vx =',diff(V,x))
    print('Vy =',diff(V,y))
    print('Vxx = ',diff(V,x,x))
    print('Vxy =',diff(V,x,y))
    #print('Vyx',diff(V,y,x))
    print('Vyy =',diff(V,y,y))

    print('hess',diff(V,x,x)*diff(V,y,y) - diff(V,x,y)**2) 

if(1): ### DW_morse_harm potential
    alpha = sympy.Symbol('self.alpha')
    lamda = sympy.Symbol('self.lamda')
    g = sympy.Symbol('self.g')
    D = sympy.Symbol('self.D')
    z = sympy.Symbol('self.z')

    quartx = (x**2 - lamda**2/(8*g))
    eay = exp(-alpha*y)
    
    vx = g*quartx**2
    vy = D*(1-eay)**2
    vxy = z**2*alpha**2*x**2*y**2/2
             
    V = vx + vy + vxy

    print('V',V)
    print('Vx =',diff(V,x))
    print('Vy =',diff(V,y))
    print('Vxx = ',diff(V,x,x))
    print('Vxy =',diff(V,x,y))
    #print('Vyx',diff(V,y,x))
    print('Vyy =',diff(V,y,y))

if(0):### Quartic Bistable potential
    alpha = sympy.Symbol('alpha')#'self.alpha')
    lamda = sympy.Symbol('lamda')#'self.lamda')
    g = sympy.Symbol('g')#self.g')
    D = sympy.Symbol('D')#self.D')
    z = sympy.Symbol('z')#self.z')
    k = sympy.Symbol('k')#self.k')
    eta = sympy.Symbol('eta')#self.eta')

    eay = exp(-alpha*y)
    quartx = (x**2 - lamda**2/(8*g))
    Vb = lamda**4/(64*g)
    vy = D*(1-eay)**2
    vx = g*quartx**2
    vxy = (vx-Vb)*(exp(-z*alpha*y) - 1)

    V = vx + vy + vxy

    print('V',V)
    print('Vx =',diff(V,x))
    print('Vy =',diff(V,y))
    print('Vxx = ',diff(V,x,x))
    print('Vxy =',diff(V,x,y))
    #print('Vyx',diff(V,y,x))
    print('Vyy =',diff(V,y,y))

    print('hess',diff(V,x,x)*diff(V,y,y) - diff(V,x,y)**2) 

if(0):### Adams function
    vx = 2*x**2*(4-x)
    vy = y**2*(4+y)
    vxy = x*y*(6 - 17*exp(-(x**2 + y**2)/4))    

    V = vx + vy + vxy

    print('V',V)
    print('Vx',diff(V,x))
    print('Vy',diff(V,y))
    print('Vxx',diff(V,x,x))
    print('Vxy',diff(V,x,y))
    print('Vyx',diff(V,y,x))
    print('Vyy',diff(V,y,y))

if(0):### Henon Heiles
    lamda = sympy.Symbol('lamda')
    g = sympy.Symbol('g')
    
    vh = 0.5*(x**2+y**2)
    vp = lamda*(x**2*y - y**3/3.0)
    vs = g*(x**2+y**2)**2   

    V = vh + vp + vs

    print('V',V)
    print('Vx',diff(V,x))
    print('Vy',diff(V,y))
    print('Vxx',diff(V,x,x))
    print('Vxy',diff(V,x,y))
    print('Vyx',diff(V,y,x))
    print('Vyy',diff(V,y,y))


            



