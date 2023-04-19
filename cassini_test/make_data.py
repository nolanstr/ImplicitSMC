from sympy import plot_implicit, symbols, Eq, solve

e = 1.2
a = 1
b = a*e

x, y = symbols('x y')

eq = Eq((x**2 + y**2)**2 - 2*a**2*(x**2-y**2) - k**4 + a**4)

