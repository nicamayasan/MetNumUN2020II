︠26309b9d-b1b8-4a3e-a5fd-5589f6c4096d︠
# **1. Polinomios con SymPyLab**
SymPy's documentation
SymPy’s polynomials\


(x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)*(x-9)*(x-10) = \
x^10 - 55 x^9 + 1320 x^8 - 18150 x^7 + 157773 x^6 - 902055 x^5 + 3416930 x^4 - 8409500 x^3 + 12753576 x^2 - 10628640 x + 3628800
︡f094afa0-8e13-4f5a-abcf-6536d6eedd46︡
︠0f99b823-3923-409e-93b6-1a1e2507ac5bs︠
from sympy import *
import copy

x = Symbol('x')

print("\tFormula de uso\n")
formula = x**10 - 55*x**9 + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800

print(formula,"\n")

#P, r = div(P, x-1)

formula1 = copy.copy(formula)

t = factor_list(formula)

#print(len(t[1]))
print("\tDescomposicion de la polinomial\n")
i = 1
for c in range (len(t[1])):

  for b in range (t[1][c][1]):
    divi = t[1][c][0]
    punto = "Punto " + str(i)
    print (punto)
    i+=1

    print("Dato: ", divi,"\n")
    formula, r = div(formula, divi)
    print("Salida\n", formula,"\n")
︡52df3338-1ade-4a18-9df7-e35455830bb3︡{"stdout":"\tFormula de uso\n\n"}︡{"stdout":"x**10 - 55*x**9 + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800 \n\n"}︡{"stdout":"\tDescomposicion de la polinomial\n\n"}︡{"stdout":"Punto 1\nDato:  x - 10 \n\nSalida\n x**9 - 45*x**8 + 870*x**7 - 9450*x**6 + 63273*x**5 - 269325*x**4 + 723680*x**3 - 1172700*x**2 + 1026576*x - 362880 \n\nPunto 2\nDato:  x - 9 \n\nSalida\n x**8 - 36*x**7 + 546*x**6 - 4536*x**5 + 22449*x**4 - 67284*x**3 + 118124*x**2 - 109584*x + 40320 \n\nPunto 3\nDato:  x - 8 \n\nSalida\n x**7 - 28*x**6 + 322*x**5 - 1960*x**4 + 6769*x**3 - 13132*x**2 + 13068*x - 5040 \n\nPunto 4\nDato:  x - 7 \n\nSalida\n x**6 - 21*x**5 + 175*x**4 - 735*x**3 + 1624*x**2 - 1764*x + 720 \n\nPunto 5\nDato:  x - 6 \n\nSalida\n x**5 - 15*x**4 + 85*x**3 - 225*x**2 + 274*x - 120 \n\nPunto 6\nDato:  x - 5 \n\nSalida\n x**4 - 10*x**3 + 35*x**2 - 50*x + 24 \n\nPunto 7\nDato:  x - 4 \n\nSalida\n x**3 - 6*x**2 + 11*x - 6 \n\nPunto 8\nDato:  x - 3 \n\nSalida\n x**2 - 3*x + 2 \n\nPunto 9\nDato:  x - 2 \n\nSalida\n x - 1 \n\nPunto 10\nDato:  x - 1 \n\nSalida\n 1 \n\n"}︡{"done":true}
︠b2e88d0f-4041-475a-a9bc-3cc73b90d40as︠
factor(formula1)
︡3f866f7a-2261-48dd-9af4-b16354c0ffc8︡{"stdout":"(x - 10)*(x - 9)*(x - 8)*(x - 7)*(x - 6)*(x - 5)*(x - 4)*(x - 3)*(x - 2)*(x - 1)\n"}︡{"done":true}
︠2ac46a67-1b85-4cdb-847b-cc032c9abc07s︠
solveset(Eq(formula1,0),x)
︡9089222e-4109-46b9-bcf8-92554b8d4dcc︡{"stdout":"FiniteSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)"}︡{"stdout":"\n"}︡{"done":true}
︠d50fc9e5-4c16-4b19-b964-22ca786b2b19s︠
from sympy import *

x0, y0, y1, z0, z1= symbols("x0 y0 y1 z0 z1", real=True)
x = I*x0
y = y0 - I*y1
z = z0 + 2*I*z1

print("\tSalida del result para x*y/z:\n")
print(expand(x*y)/z)
print("\n\tSalida de Alternate from.\n")
print(x*y/z)
print("\n\tSalida de Expanded from.\n")
print(expand(x*y/z))
print("\n\tSalida de Alternate form assuming x0, y0, y1, z0, and z1 are real:\n")
print((x*y/z)+expand(x*y/z))
print("\n\tCalcular la distribución.\n")
w = N(1/(pi + I), 20)
print ("w = ", w)
︡74f5218c-8d07-4488-a4e0-18fb7f6df4c8︡{"stdout":"\tSalida del result para x*y/z:\n\n"}︡{"stdout":"(I*x0*y0 + x0*y1)/(z0 + 2*I*z1)\n"}︡{"stdout":"\n\tSalida de Alternate from.\n\n"}︡{"stdout":"I*x0*(y0 - I*y1)/(z0 + 2*I*z1)\n"}︡{"stdout":"\n\tSalida de Expanded from.\n\n"}︡{"stdout":"I*x0*y0/(z0 + 2*I*z1) + x0*y1/(z0 + 2*I*z1)\n"}︡{"stdout":"\n\tSalida de Alternate form assuming x0, y0, y1, z0, and z1 are real:\n\n"}︡{"stdout":"I*x0*y0/(z0 + 2*I*z1) + x0*y1/(z0 + 2*I*z1) + I*x0*(y0 - I*y1)/(z0 + 2*I*z1)\n"}︡{"stdout":"\n\tCalcular la distribución.\n\n"}︡{"stdout":"w =  0.28902548222223624241 - 0.091999668350375232456*I\n"}︡{"done":true}
︠6ec96966-4409-4f36-9517-c42bf1dab1aaw︠
# **2. SymPy's integrals**
Comencemos con un problema de integración simple en 1D,
$$\int_5^{10}\frac{12}{z^2}\,\mathrm{dz}$$
Esto es fácil de resolver analíticamente y podemos usar la biblioteca SymPy en caso de que haya olvidado cómo resolver integrales simples.

import sympy
#Guardar los resultados
resultados = {}

z = sympy.Symbol("z")
l = sympy.integrate(12/(z**2))
print("\tProdución de la integral.\n")
print(l)
resultados["Analisis"] = float(l.subs(z,10)-l.subs(z,5))
print("\n\tAnalisiss de resultados:\n {}".format(resultados["Analisis"]))
︡b49705f7-8bb8-42b3-bc47-63138384d167︡{"stdout":"\tProdución de la integral.\n\n"}︡{"stdout":"-12/z\n"}︡{"stdout":"\n\tAnalisiss de resultados:\n 1.2\n"}︡{"done":true}
︠fdfd8279-6188-4011-ae0f-57f5f8bdd74bw︠
# Integrando con Monte Carlo
︡92636cb9-7d9f-4779-b5a5-0ec6fc3776ef︡
︠115ca6d8-ef97-494a-b37b-009f6c6b45ebr︠
import numpy
N = 1_000_000
count = 0
for i in range(N):
  x = numpy.random.uniform(5,10)
  count += 12/x**2
volumen = 10-5
resultados["MC"] = volumen * count / float(N)
print("\tResultado de Monte Carlo: \n {}".format(resultados["MC"]))
︡91c574e0-8173-459e-9914-fc71b1815371︡
︠c520f7d5-6f85-4044-a557-152d804d1944w︠
import sympy
import math
from fractions import Fraction
x = Symbol("x")
i = integrate(cos(x)*sin(x)**2)
print("\tDerivada de la funcion de cos(x)*sin(x)^2 seria.\n")
print(i)
print("\n\tEntre los limites de pi/2 y 0.\n")
a = i.subs(x,math.pi/2) - i.subs(x,0)
print(Fraction(str(a)).limit_denominator(), " = ", a)
︡884ed990-b0e2-496a-ac1d-b6022a01d598︡
︠eb5ba248-2b47-4ca1-b676-510c02b0d741w︠
import numpy
import math
N = 1_000_000
count = 0
k = []
for i in range(N):
  x = numpy.random.uniform(0, math.pi/2)
  count += math.cos(x)*math.sin(x)**2
volumen = math.pi/2 - 0
resultados["MC"] = volumen * count / float(N)
print("Resultado estándar de Monte Carlo: {}".format(resultados["MC"]))
︡306f06b5-75ff-4df9-9f1b-afba317e6e09︡
︠f03d10da-e603-4ef5-9bbd-4c75c62a9e7fw︠
import sympy
from fractions import Fraction
x = sympy.Symbol("x")
y = sympy.Symbol("y")
z = sympy.Symbol("z")

expr = z
resultado = sympy.integrate(expr, (z,0,1-y**2),(y,x,1),(x,0,1))

resultados = {}
resultados["analisis1"] = float(resultado)
resultados["analisis2"] = (Fraction(str(resultado)).limit_denominator())

print("Analisis de resultado: ", resultados["analisis2"], " = ", resultados["analisis1"])
︡c2878f34-deb7-4bea-bef4-f97a5aba707c︡
︠c481c159-d1c1-4325-aa34-906e28a19ec7w︠
# Integral varias dimensiones Halton
︡ebe96d69-322b-4597-bd19-f48965846a78︡
︠34e75fab-6b77-43e3-b4dd-8ad563bddd09w︠
import sympy

x1 = sympy.Symbol("x1")
x2 = sympy.Symbol("x2")
x3 = sympy.Symbol("x3")
expr = sympy.sin(x1) + 7*sympy.sin(x2)**2 + 0.1 * x3**4 * sympy.sin(x1)
res = sympy.integrate(expr,
                      (x1, -sympy.pi, sympy.pi),
                      (x2, -sympy.pi, sympy.pi),
                      (x3, -sympy.pi, sympy.pi))
# Note: we use float(res) to convert res from symbolic form to floating point form
result = {}
result["analytical"] = float(res)
print("Analisis de resultado: {}".format(result["analytical"]))
︡ab5039ca-e634-406d-85d0-5f49bf6e83d5︡
︠f6e9723b-794c-4d91-a798-43c3d344e075︠

︡fdb953a5-00fe-48fa-9d70-d59dd698a90b︡
︠3d70fed9-3dbd-4cff-a99b-58f663242c29w︠
N = 10_000
accum = 0
for i in range(N):
    xx1 = numpy.random.uniform(-numpy.pi, numpy.pi)
    xx2 = numpy.random.uniform(-numpy.pi, numpy.pi)
    xx3 = numpy.random.uniform(-numpy.pi, numpy.pi)
    accum += numpy.sin(xx1) + 7*numpy.sin(xx2)**2 + 0.1 * xx3**4 * numpy.sin(xx1)
volume = (2 * numpy.pi)**3
result = {}
result["MC"] = volume * accum / float(N)
print("Resultado Monte Carlo: {}".format(result["MC"]))
︡299c3022-0120-40ae-a398-a8932cead097︡
︠8f64a915-fa6c-4647-b0d8-d112c33d78d2w︠
import math
import numpy

def halton(dim: int, nbpts: int):
    h = numpy.full(nbpts * dim, numpy.nan)
    p = numpy.full(nbpts, numpy.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognbpts = math.log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(math.ceil(lognbpts / math.log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1))

        for j in range(nbpts):
            d = j + 1
            sum_ = math.fmod(d, b) * p[0]
            for t in range(1, n):
                d = math.floor(d / b)
                sum_ += math.fmod(d, b) * p[t]

            h[j*dim + i] = sum_
    return h.reshape(nbpts, dim)
︡314686a3-d8f0-4cfe-b1e7-575315743641︡
︠ae833d76-8e86-4865-ab83-51487b6505a6w︠
import matplotlib.pyplot as plt
N = 1000
seq = halton(2, N)
plt.title("2D Secuencia Halton")
plt.axes().set_aspect('equal')
plt.scatter(seq[:,0], seq[:,1], marker=".", alpha=0.5);
︡121187c5-7505-4545-bff2-1d111c83d36c︡
︠a8199480-2de9-4741-9de4-fa241cbd6792w︠
N = 10000

seq = halton(3, N)
accum = 0
for i in range(N):
    xx1 = -numpy.pi + seq[i][0] * numpy.pi * 2
    xx2 = -numpy.pi + seq[i][1] * numpy.pi * 2
    xx3 = -numpy.pi + seq[i][2] * numpy.pi * 2
    accum += numpy.sin(xx1) + 7*numpy.sin(xx2)**2 + 0.1*xx3**4 * numpy.sin(xx1)
volume = (2 * numpy.pi)**3
result = {}
result["MC"] = volume * accum / float(N)
print("Resultado Monte Carlo Secuencia Halton : {}".format(result["MC"]))
︡6b584dc4-3859-4cd6-b95a-7d1cd48c4a1d︡
︠b035bdd4-0f62-4dce-b675-20b4b633c5b8w︠
###Las secuencias de Sobol
También llamadas secuencias $LP_{\tau}$ o secuencias (t, s) en la base 2) son un ejemplo de secuencias cuasialeatorias de discrepancia baja . Fueron introducidos por primera vez por el matemático ruso Ilya M. Sobol (Илья Меерович Соболь) en 1967.

Estas secuencias utilizan una base de dos para formar sucesivamente particiones uniformes más finas del intervalo unitario y luego reordenar las coordenadas en cada dimensión.
︡23373442-861b-456b-9308-fbd6c0b15e86︡
︠609411de-a26b-45b5-96a5-995e5baf8b4cw︠
import sobol_seq

seq = sobol_seq.i4_sobol_generate(2, N)
plt.title("2D Sobol sequence")
plt.scatter(seq[:,0], seq[:,1], marker=".", alpha=0.5);
︡d4c6e004-a687-49dc-bf22-1829cf21a1fc︡
︠3609687b-41b5-429c-98bb-057036a0a372w︠
N = 1_000_000

seq = halton(3, N)
count = [0,0,0]
for i in range(N):
  x1 =  0 + seq[i][1] * 1
  y1 =  x1 + seq[i][2] * 1
  z1 =  abs(seq[i][0] * (1-(y1**2)))

  #print(i," | ",x1, " | ", y1, " | ", z1)
  count[0] += x1
  count[1] += y1
  count[2] += z1

x1 = count[0]/float(N)
y1 = count[1]/float(N)
z1 = count[2]/float(N)
print(x1, " | ", y1, " | ", z1)
volumen = (1-x1)*(1-(y1**2))
resultados = {}
resultados["MC"] = volumen * z1
print("Resultado estándar de Monte Carlo: {}".format(resultados["MC"]))
︡2f926888-c5dd-423c-872c-d5164ba5cbf6︡
︠102fdd47-7a16-4d4b-b4e0-b297593fde3cw︠
︡082e913e-42fc-4f88-a200-4ddecf691444︡









