import numpy as np
import sympy as sp

#Real solid harmonics

def binom(u,l):
    #( upper , lower )
    #print u, l
    return sp.factorial(int(u))/(sp.factorial(int(l))*sp.factorial(int(u-l)))

def deltax(n,d,o):
    ret = 1
    if d==o:
        ret = n
    return ret

def vm(m):
    ret = 0
    if m<0:
        ret = 0.5
    return ret

def N(l,m):
    return sp.sqrt(2*sp.factorial(l-abs(m))*sp.factorial(l+abs(m))/deltax(2,0,m))/(2**abs(m)*sp.factorial(l))

def nfac(a,l):
    return (2*a/np.pi)**(l/2.0 + 0.75)
    
def C(l,m,t,u,v):
    return (-1)**(t+v-vm(m))*0.25**t*binom(l,t)*binom(l-t, abs(m)+t)*binom(t,u)*binom(abs(m), 2*v)
    
    
def Slm(l,m):
    x,y,z = sp.symbols("x y z")
    #N = N(l,m)
    ret = 0
    #print ret
    #print np.arange((l-abs(m))/2 +1)
    for t in np.arange((l-abs(m))/2 +1):
        for u in np.arange(t+1):
            for v_ in np.arange(abs(m)/2.0 - vm(m) +1):
                #print t
                v = v_ + vm(m)
                #print C(l,m,t,u,v)* x**(2*t + abs(m) - 2*(u+v))*y**(2*(u+v))*z**(l-2*t-abs(m))
                ret += C(l,m,t,u,v)* x**(2*t + abs(m) - 2*(u+v))*y**(2*(u+v))*z**(l-2*t-abs(m))
                #print(ret)
    return (N(l,m)*ret).simplify()    
            


def get_ao(exponent, l, m):
    x,y,z = sp.symbols("x y z")
    ao_sympy = nfac(exponent, l)*Slm(l,m)*sp.exp(-exponent*sp.sqrt(x**2 + y**2 + z**2))
    return ao_sympy, sp.lambdify([x,y,z], ao_sympy, "numpy")

def get_contracted(exponents, weigths, l, m):
    x,y,z = sp.symbols("x y z")
    ao_sympy = 0
    for i in range(len(exponents)):
        ao_sympy += nfac(exponents[i], l)*weigths[i]*Slm(l,m)*sp.exp(-exponents[i]*(x**2 + y**2 + z**2))
        #ao_sympy += weigths[i]*Slm(l,m)*sp.exp(-exponents[i]*(x**2 + y**2 + z**2))
    return ao_sympy, sp.lambdify([x,y,z], ao_sympy, "numpy")
