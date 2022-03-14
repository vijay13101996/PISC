import sympy
from sympy import *

x = sympy.Symbol('x')
y = sympy.Symbol('y')


if(0):### Quartic Bistable potential
	alpha = sympy.Symbol('alpha')
	lamda = sympy.Symbol('lamda')
	g = sympy.Symbol('g')
	D = sympy.Symbol('D')
	z = sympy.Symbol('z')

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

if(1):### Henon Heiles
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


			



