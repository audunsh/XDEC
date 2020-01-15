#import numpy as np

import autograd.numpy as np

class lie_component():
    """
    evaluatable component to suN matrix
    """
    def __init__(self, n, sgn, tr):
        self.n = n
        self.sgn = sgn
        self.tr = tr
    def __mul__(self, other):
        #print(type(self), type(other))
        if other==0:
            return 0
        else:
            n = self.n + other.n
            tr = self.tr + other.tr
            sgn = self.sgn*other.sgn
            return lie_component(n, sgn, tr)

            
    def s(self):
        """
        return string of instance,
        for subsequent use in evaluation
        """
        ev = "%i" % self.sgn
        for i in np.arange(len(self.tr)):
            if self.tr[i] == "c":
                ev += "*np.cos(%s)" % self.n[i]
            if self.tr[i] == "s":
                ev += "*np.sin(%s)" % self.n[i]
        #print(ev)
        return ev  
    
    def l(self):
        """
        return lambda function of instance
        """
        ev = "lambda "
        for i in np.arange(len(self.n)):
            ev += "%s ," % self.n[i]
        ev = ev[:-1]
        ev += " : %i" % self.sgn
        for i in np.arange(len(self.tr)):
            if self.tr[i] == "c":
                ev += "*np.cos(%s)" % self.n[i]
            if self.tr[i] == "s":
                ev += "*np.sin(%s)" % self.n[i]
        #print(ev)
        return eval(ev)   
    
def lie_eye(N):
    """
    Returns a N by N object ndarray
    with lie components corresponding
    to the identity matrix.
    """
    ret = np.zeros((N,N), dtype = object)
    for i in np.arange(N):
        for j in np.arange(N):
            ret[i,j] = lie_term([lie_component([],0,[])])
    for i in np.arange(N):
        ret[i,i] = lie_term([lie_component([],1,[])])
    return ret

def lie_shed_zeros(M):
    """
    shed zeros in every term
    of lie matrix M
    """
    for i in np.arange(M.shape[0]):
        for j in np.arange(M.shape[1]):
            lt_new = []
            for component in M[i,j].lc:
                if component.sgn != 0:
                    lt_new.append(component)
            if len(lt_new)==0:
                M[i,j] = lie_term([lie_component([],0,[])])
            else:
                M[i,j] = lie_term(lt_new)


class lie_term():
    def __init__(self, lc):
        self.lc = lc
    def __mul__(self, other):
        if other == 0:
            return 0
        else:
            lret = []
            for i in self.lc:
                for j in other.lc:
                    lret.append(i*j)
            return lie_term(lret)
    def __add__(self, other):
        if other == 0:
            return 0
        else:
            lret = self.lc + other.lc
            return lie_term(lret)
    def lt2l(self):
        #gather all variables
        var = []
        for i in self.lc:
            var += i.n
        var = np.unique(np.array(var))
        print(var)
        
        ev = "lambda "
        for i in np.arange(len(var)):
            ev += "%s ," % var[i]
        ev = ev[:-1] + ": "
        
        
        for i in self.lc:
            ev += "%s +" % i.s()
        
        ev = ev[:-1]
        return eval(ev)
    def lt2s(self):
        ev = ""
        for i in self.lc:
            ev += "%s +" % i.s()
        
        ev = ev[:-1]
        return ev
        
        
def get_su2(x):
    lt1 = lie_term([lie_component([x],  1, ["c"])])
    lt2 = lie_term([lie_component([x], -1, ["s"])])
    lt3 = lie_term([lie_component([x],  1, ["s"])])
    lt4 = lie_term([lie_component([x],  1, ["c"])])
    
    ret = np.zeros((2,2), dtype = object)
    ret[0,0] = lt1
    ret[0,1] = lt2
    ret[1,0] = lt3
    ret[1,1] = lt4
    
    
    return ret

def sun_ia(x, n, i, a):
    """
    Generate rotation of i a 
    """
    ret = lie_eye(n)
    ret_ia = get_su2(x)
    
    ret[i,i] = ret_ia[0,0]
    ret[i,a] = ret_ia[0,1]
    ret[a,i] = ret_ia[1,0]
    ret[a,a] = ret_ia[1,1]
    
    return ret

def suNia(n, ni, na):
    """
    Generate all rotations between indices in lists ni and na
    for a n by n matrix.
    
    Example usage:
    
    nx, ux = suNia(4,[0,1],[2,3])
    
    where nx is the number of parameters and
    ux is the rotation matrix, thus
    
    x = np.array([0.0, 0.0, 0.0, 0.0]) #length nx array
    
    print(ux(x))
    
    [ [1.0 0.0 0.0 0.0 ] 
      [0.0 1.0 0.0 0.0 ]
      [0.0 0.0 1.0 0.0 ]
      [0.0 0.0 0.0 1.0 ]  ]
    
    """
    
    x = []
    for i in np.arange(len(ni)*len(na)):
        x.append("x[%i]" % i)
    x = np.array(x)
    
    ret = lie_eye(n)
    
    j = 0
    for i in ni:
        for a in na:
            ret = np.dot(ret, sun_ia(x[j], n, i,a))
            lie_shed_zeros(ret)
            j += 1   
    return len(x), eval(array2lambda(x, ret))



def sun_blockdiag(x, N, o = 0):
    ret = lie_eye(N)
    for i in np.arange(x.size):
        ret[(2*i):2*(i+1),(2*i):2*(i+1)] = get_su2(x[i])
    return np.roll(ret, (o,o), axis = (0,1))

def full_suN(n):
    c = 0
    U = lie_eye(n)
    X = []
    for i in np.arange(n):
        for j in np.arange(i+1,n):
            Um = lie_eye(n)
            x = "x[%i]" % c
            X.append(x)
            m = get_su2(x)
            Um[i,i] = m[0,0]
            Um[j,j] = m[1,1]
            Um[i,j] = m[0,1]
            Um[j,i] = m[1,0]
            c += 1
            U = np.dot(U, Um)
            lie_shed_zeros(U)
    return c, eval(array2lambda(X, U))

def suN(n):
    """
    Generate N-dimensional SU(n) matrix
    parametrized with n-1 degrees of freedom
    
    Usage:
        
        > n, U = suN(3)
        
        > print( U([0.0,0.0,0.0]) ) #identity
        
        [[1.0 0.0 0.0],
         [0.0 1.0 0.0],
         [0.0 0.0 1.0]]
         
        
    """
    x = []
    for i in np.arange(n-1):
        x.append("x[%i]" % i)
    x = np.array(x)
    
    M = np.dot(sun_blockdiag(x[:n//2], n), sun_blockdiag(x[n//2:], n, o=1))
    
    return  n-1, eval(array2lambda(x, M))


def array2lambda(x, M):
    ev = "lambda x :"
    #for i in np.arange(len(x)):
    #    ev += "%s ," % x[i]
    #ev = ev[:-1] + ": "
    
    ev += "np.array(["
    Nx, Ny = M.shape
    for i in np.arange(Nx):
        ev += "["
        for j in np.arange(Ny):
            #print(M[i,j], type(M[i,j]))
            if M[i,j]==0:
                ev += "0 ,"
                
            else:
                ev += "%s ," % M[i,j].lt2s()
        ev = ev[:-1] + "],"
    ev = ev[:-1] + "])"
    return ev