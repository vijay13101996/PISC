import sympy
from sympy import *

x = sympy.Symbol('x')
y = sympy.Symbol('y')

if(1):
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
	

if(0):### Quartic Bistable potential
	alpha = sympy.Symbol('self.alpha')
	lamda = sympy.Symbol('self.lamda')
	g = sympy.Symbol('self.g')
	D = sympy.Symbol('self.D')
	z = sympy.Symbol('self.z')

	eay = exp(-alpha*y)
	quartx = (x**2 - lamda**2/(8*g))
	vy = D*(1-eay)**2
	vx = g*quartx**2
	vxy = (vx-lamda**4/(64*g))*(exp(-z*alpha*y) - 1)

	V = vx + vy + vxy

	print('V',V)
	print('Vx',diff(V,x))
	print('Vy',diff(V,y))
	print('Vxx',diff(V,x,x))
	print('Vxy',diff(V,x,y))
	print('Vyx',diff(V,y,x))
	print('Vyy',diff(V,y,y))

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


			



