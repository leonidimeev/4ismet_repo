from numpy import tan, cos

def f1(x1,x2):
    return ( tan(x1*x2+0.3)-x1*x1 )

def f2(x1,x2):
    return ( (0.9*x1*x1)+(2*x2*x2)-1 )

def F1_x1(x1,x2):
    return ( x2/(cos(x1*x2+0.3)*(cos(x1*x2+0.3))) - 2*x1 )

def F1_x2(x1,x2):
    return ( x1/(cos(x1*x2+0.3)*(cos(x1*x2+0.3))) )

def F2_x1(x1,x2):
    return ( 1.8*x1 )

def F2_x2(x1,x2):
    return ( 4*x2 )

# def X1(x1,x2):
#     try:
#         return ( x1 - (det2x2(f1(x1,x2),F1_x2(x1,x2),f2(x1,x2),F2_x2(x1,x2)))/(det2x2(F1_x1(x1,x2),F1_x2(x1,x2),F2_x1(x1,x2),F2_x2(x1,x2)))   )
#     except ValueError:
#         print('MaybeKorenb')
#         return 0

# def X2(x1,x2):
#     try:
#         return ( x2 - (det2x2(F1_x1(x1,x2),f1(x1,x2),F2_x1(x1,x2),f2(x1,x2)))/(det2x2(F1_x1(x1,x2),F1_x2(x1,x2),F2_x1(x1,x2),F2_x2(x1,x2)))   )
#     except ValueError:
#         print('MaybeKorenb')
#         return 0

# def lab_1():
#     i = 0
#     x1 = 1.0
#     x2 = 0.5
#     e = pow(10,-30)

#     #while x1-X1(x1,x2)>e or x2-X2(x1,x2)>e:
#     while i<=9:
#         xi = X1(x1,x2)
#         xj = X2(x1,x2)
#         x1 = xi
#         x2 = xj
#         i += 1

#     print(x1,x2,i)

# def ur2lab(i, x1, x2, x3, x4):
#     if i == 1:
#         return (0.1*x1 + 0.2*x2 - 0.3*x3 + 0.2*x4 - 0.4)
#     elif i == 2:
#         return (0.2*x1 + 0.3*x2 - 0.1*x3 - 0.1*x4 + 0.2)
#     elif i == 3:
#         return (0.3*x1 - 0.2*x2 + 0.1*x3 + 0.2*x4 - 0.1)
#     elif i ==4:
#         return (0.4*x1 + 0.1*x2 + 0.2*x3 - 0.2*x4 + 0.3)

def lab1_iteration(i,j):
    from numpy import linalg, array
    A1 = array([[f1(i,j), F1_x2(i,j),], [f2(i,j), F2_x2(i,j)]])
    A2 = array([[F1_x1(i,j), f1(i,j)], [F2_x2(i,j), f2(i,j)]])
    J = array([[F1_x1(i,j), F1_x2(i,j)], [F2_x1(i,j), F2_x2(i,j)]])

    det_A1 = linalg.det(A1)
    det_A2 = linalg.det(A2)
    det_J = linalg.det(J)

    new_i = i - det_A1/det_J
    new_j = j - det_A2/det_J

    d = (abs(new_i - i), abs(new_j - j))

    return new_i, new_j, d

def lab_1():
    eps = 0.0000001
    i = 1
    j = 1
    d = (1,1)
    iteration = 1
    while d[0] > eps or d[1] > eps:
        i,j,d = lab1_iteration(i,j)
        print(i, j, 'iteration number:' + str(iteration))
        iteration += 1

def lab_2_1():
    A = [
    [0.1, 0.2, -0.3, 0.2],
    [0.2, 0.3, -0.1, -0.1],
    [0.3, -0.2, 0.1, 0.2],
    [0.4, 0.1, 0.2, -0.2]
    ]

    B = [-0.4, 0.2, -0.1, 0.3]

    X = [0,0,0,0]
    XX = [0,0,0,0]

    e = 0.001
    k = 0
    N = 4


    while True:
        k += 1
        for i in range(0,N):
            s = B[i]
            for j in range(0,i):
                s = s + A[i][j]*XX[j]
            for j in range(i,N):
                s = s + A[i][j]*X[j]
            XX[i] = s
        max_x = max(abs(X[i] - XX[i]) for i in range(0,N))
        for i in range(0,N):
            X[i] = XX[i]

        if max_x < e:
            break
    print(X,k)

    for i in range(0, N):
        s = B[i]
        for j in range(0,N):
            s += A[i][j]*X[j]

        print(s)

def lab_3():

    def y(x):
        from numpy import cos
        return(2-cos(x)*cos(x))

    def dy(x):
        from numpy import cos, sin
        return(2*cos(x)*sin(x))

    def L(x):
        s = 0
        for k in range(10):
            upper = 1
            lower = 1
            for i in range(10):
                if i != k:
                    upper *= (x - X[i])
                    lower *= (X[k] - X[i])
            s += Y[k] * (upper/lower)
        return s

    

    X = []
    Y = []
    x = 0
    from numpy import pi
    h = pi / 20
    for i in range(10):
        X.append(x)
        Y.append(y(x))
        x += h

    print(L(pi/4) - y(pi/4))
    print(L(pi/6) - y(pi/6))
    print(L(pi/4))

    

def lab_3N():

    def y(x):
        from numpy import cos
        return(2-cos(x)*cos(x))

    def dy(x):
        from numpy import cos, sin
        return(2*cos(x)*sin(x))
    
    def deltax(x):
        return abs(0-y(x))

    def N(x0,e):
        delta = deltax(x0)
        while delta > e:
            x0 = x0 - y(x0)/dy(x0)
            delta = deltax(x0)
            print ('Root is at: ', x0)
            print ('f(x) at root is: ', y(x0))

#    x0s = []
#    for x in range(-100 ,101):
#        x0s.append(x/100)
#    for x0 in x0s:
#        N(x0, 0.01)
    N(0.2,0.001)

def Newton():
    
    def y(x):
        from numpy import cos
        return(2-cos(x)*cos(x))

    def dy(x):
        from numpy import cos, sin
        return(2*cos(x)*sin(x))

    def t(x,x0,h):
        return((y(x)-x0)/h)

    def delta_y(x,deg):
        z=1
        delta_y = y(x+deg) - y(x)
        while z != deg:
            z=z+1
            delta_y(delta_y,deg)
        
def DU1por():

    '''
    In [1]: from sympy import *
    In [2]: import numpy as np
    In [3]: x = Symbol('x')
    In [4]: y = x**2 + 1
    In [5]: yprime = y.diff(x)
    In [6]: yprime
    Out[6]: 2⋅x

    In [7]: f = lambdify(x, yprime, 'numpy')
    In [8]: f(np.ones(5))
    Out[8]: [ 2.  2.  2.  2.  2.]
    '''
    
    from numpy.lib.scimath import logn
    from math import log
    from math import e

    # xy'-y2lnx+y=0
    # y'=(y2lnx-y)/x
    
    def f(x,y):
        return ((y*y*logn(e, x)-y)/x)

    def fmath(x,y):
        return ((y*y*log(x,e)-y)/x)

    y = [1,1,1,1,1,1,1,1,1,1]
    x = [1,1,1,1,1,1,1,1,1,1]
    y[0] = 1
    x[0] = 1
    h = 0.1
    
    for i in range (10):
        
        x[i] = 1+i*h
        y[i] = y[i-1]+( h * f(x[i],y[i]))
        print(y[i]-(1/(1+logn(e, x[i]))))

    print('-----------------------------------')

    y2 = [1,1,1,1,1,1,1,1,1,1]
    x2 = [1,1,1,1,1,1,1,1,1,1]
    for i in range (10):
        
        x2[i] = 1+i*h
        y2[i] = y2[i-1]+( h * f(x2[i],y2[i]))
        print(y2[i]-(1/(1+log(x2[i],e))))

def DU1por2():

    '''
    In [1]: from sympy import *
    In [2]: import numpy as np
    In [3]: x = Symbol('x')
    In [4]: y = x**2 + 1
    In [5]: yprime = y.diff(x)
    In [6]: yprime
    Out[6]: 2⋅x

    In [7]: f = lambdify(x, yprime, 'numpy')
    In [8]: f(np.ones(5))
    Out[8]: [ 2.  2.  2.  2.  2.]
    '''
    
    from numpy.lib.scimath import logn
    from math import log
    from math import e

    # xy'-y2lnx+y=0
    # y'=(y2lnx-y)/x
    
    def f(x,y):
        return ((y*y*logn(e, x)-y)/x)

    y = [1,1,1,1,1,1,1,1,1,1]
    x = [1,1,1,1,1,1,1,1,1,1]
    y[0] = 1
    x[0] = 1
    h = 0.1
    
    for i in range (10):
        x[i] = 1+i*h
        k1 = h*f(x[i-1],y[i-1])
        k2 = h*f(x[i-1]+(h/2), y[i-1]+(1/2)*k1)
        k3 = h*f(x[i-1]+(h/2), y[i-1]+(1/2)*k2)
        k4 = h*f(x[i-1]+h, y[i-1]+k3)    
        y[i] = y[i-1]+(1/6)*(k1+2*k2+2*k3+k4)
        print(y[i]-(1/(1+logn(e, x[i]))))


def DU1por3():
    
    from numpy.lib.scimath import logn
    from math import log
    from math import e

    # xy'-y2lnx+y=0
    # y'=(y2lnx-y)/x
    
    def f(x,y):
        return ((y*y*logn(e, x)-y)/x)

    y = [1,1,1,1,1,1,1,1,1,1]
    x = [1,1,1,1,1,1,1,1,1,1]
    y[0] = 1
    x[0] = 1
    h = 0.1
    
    for i in range (3):
        x[i] = 1+i*h
        k1 = h*f(x[i-1],y[i-1])
        k2 = h*f(x[i-1]+(h/2), y[i-1]+(1/2)*k1)
        k3 = h*f(x[i-1]+(h/2), y[i-1]+(1/2)*k2)
        k4 = h*f(x[i-1]+h, y[i-1]+k3)    
        y[i] = y[i-1]+(1/6)*(k1+2*k2+2*k3+k4)
        print(y[i]-(1/(1+logn(e, x[i]))))

    i = 3
    
    while i < 10:
        # y[i] - y[i-1])/h = A
        # y[i] = A*h+y[i-1]
        x[i] = 1+i*h
        y[i] = ((1/24)*(55*f(x[i-1],y[i-1])-59*f(x[i-2],y[i-2])+37*f(x[i-3],y[i-3])-9*f(x[i-4],y[i-4])))*h+y[i-1]
        print(y[i]-(1/(1+logn(e, x[i]))))
        i += 1


def RungeKuttaDlyaSistem():
    
    import numpy as np
    from numpy import sin
    from math import exp

    # xy'-y2lnx+y=0
    # y'=(y2lnx-y)/x

    def xt(t):
        #return (pow(e,-t)*(sin(t)-t))
        return(exp(-t))

    def yt(t):
        #return (pow(e,-t)*(sin(t)-t))
        return(-exp(-t))

    
    
    def fx(x,y):
        #return (-x-y-(1+t*t*t)*np.power(e,-t))
        return(2*x+3*y)
    def fy(x,y):
        #return (-x-y-(1-3*t*t)*np.power(e,-t))
        return(2*x+y)

    y = [1,1,1,1,1,1,1,1,1,1]
    x = [1,1,1,1,1,1,1,1,1,1]
    t = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    #y[0] = 1
    y[0] = -1
    x[0] = 1
    h = 0.2
    k = 1

    while True:
    	k1 = fx(x[k-1], y[k-1])*h
    	m1 = fy(x[k-1], y[k-1])*h
    	k2 = fx(x[k-1] + k1/2, y[k-1] + m1/2)*h
    	m2 = fy(x[k-1] + k1/2, y[k-1] + m1/2)*h
    	k3 = fx(x[k-1] + k2/2, y[k-1] + m2/2)*h
    	m3 = fy(x[k-1] + k2/2, y[k-1] + m2/2)*h
    	k4 = fx(x[k-1] + k3, y[k-1] + m3)*h
    	m4 = fy(x[k-1] + k3, y[k-1] + m3)*h
    	x[k] = x[k-1]+(1./6.)*(k1+2*k2+2*k3+k4)
    	y[k] = y[k-1]+(1./6.)*(m1+2*m2+2*m3+m4)

    	print(xt(t[k]) - x[k],yt(t[k]) - y[k])
    	k+=1 
    	if k == 9:
    		break
    print(x)
    print('-------------------------')
    print(y)

def interpolating(x):
    from scipy import interpolate

    x_points = [ 0, 1, 2, 3, 4, 5]
    y_points = [12,14,22,39,58,77]

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)
	
def kubspline(x0):
    import numpy as np
    from math import sqrt
    x = [0, 1, 2, 3, 4, 5 ]
    y = [12,14,22,39,58,77]
    x = np.asfarray(x)
    y = np.asfarray(y)


    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)

    xdiff = np.diff(x)
    ydiff = np.diff(y)

    Li = np.empty(size)
    Li_1 = np.empty(size-1)
    z = np.empty(size)

    Li[0] = sqrt(2*xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0 
    z[0] = B0 / Li[0]

    for i in range(1, size-1, 1):
        Li_1[i] = xdiff[i-1] / Li[i-1]
        Li[i] = sqrt(2*(xdiff[i-1]+xdiff[i]) - Li_1[i-1] * Li_1[i-1])
        Bi = 6*(ydiff[i]/xdiff[i] - ydiff[i-1]/xdiff[i-1])
        z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    i = size - 1
    Li_1[i-1] = xdiff[-1] / Li[i-1]
    Li[i] = sqrt(2*xdiff[-1] - Li_1[i-1] * Li_1[i-1])
    Bi = 0.0 
    z[i] = (Bi - Li_1[i-1]*z[i-1])/Li[i]

    i = size-1
    z[i] = z[i] / Li[i]
    for i in range(size-2, -1, -1):
        z[i] = (z[i] - Li_1[i-1]*z[i+1])/Li[i]

    index = x.searchsorted(x0)
    np.clip(index, 1, size-1, index)

    xi1, xi0 = x[index], x[index-1]
    yi1, yi0 = y[index], y[index-1]
    zi1, zi0 = z[index], z[index-1]
    hi1 = xi1 - xi0

    # calculate
    f0 = zi0/(6*hi1)*(xi1-x0)**3 + \
         zi1/(6*hi1)*(x0-xi0)**3 + \
         (yi1/hi1 - zi1*hi1/6)*(x0-xi0) + \
         (yi0/hi1 - zi0*hi1/6)*(xi1-x0)
    return f0

#lab_1()
#lab_2_1()
#lab_3()
#lab_3N()
# print('First method:')          
# DU1por()
# print('-----------------------------------')
# print('Runge-Kutta`s method:')
# DU1por2()
# print('-----------------------------------')
# print('Adams method:')
# DU1por3()
'''
from numpy import *
from scipy.integrate import *

def RUN():
    import matplotlib.pyplot as plt
    
    def odein():         
             #dy1/dt=y2
             #dy2/dt=y1**2+1:
             def f(y,t):
                       return y**2+1
             t =arange(0,1,0.01)
             y0 =0.0
             y=odeint(f, y0,t)
             y = array(y).flatten()
             return y,t
    def oden():
             f = lambda t, y: y**2+1
             ODE=ode(f)
             ODE.set_integrator('dopri5')
             ODE.set_initial_value(0, 0)
             t=arange(0,1,0.01)
             z=[]
             t=arange(0,1,0.01)
             for i in arange(0,1,0.01):
                      ODE.integrate(i)
                      q=ODE.y
                      z.append(q[0])
             return z,t         
    def rungeKutta(f, to, yo, tEnd, tau):
             def increment(f, t, y, tau):
                      if z==1:
                               k0 =tau* f(t,y)
                               k1 =tau* f(t+tau/2.,y+k0/2.)
                               k2 =tau* f(t+tau/2.,y+k1/2.)
                               k3 =tau* f(t+tau, y + k2)
                               return (k0 + 2.*k1 + 2.*k2 + k3) / 6.
                      elif z==0:
                               k1=tau*f(t,y)
                               k2=tau*f(t+(1/4)*tau,y+(1/4)*k1)
                               k3 =tau *f(t+(3/8)*tau,y+(3/32)*k1+(9/32)*k2)
                               k4=tau*f(t+(12/13)*tau,y+(1932/2197)*k1-(7200/2197)*k2+(7296/2197)*k3)
                               k5=tau*f(t+tau,y+(439/216)*k1-8*k2+(3680/513)*k3 -(845/4104)*k4)
                               k6=tau*f(t+(1/2)*tau,y-(8/27)*k1+2*k2-(3544/2565)*k3 +(1859/4104)*k4-(11/40)*k5)
                               return (16/135)*k1+(6656/12825)*k3+(28561/56430)*k4-(9/50)*k5+(2/55)*k6   

             t = []
             y= []
             t.append(to)
             y.append(yo)
             while to < tEnd:
                      tau = min(tau, tEnd - to)
                      yo = yo + increment(f, to, yo, tau)
                      to = to + tau
                      t.append(to)
                      y.append(yo)         
             return array(t), array(y)
    def f(t, y):
             f = zeros([1])
             f[0] = y[0]**2+1    
             return f
    to = 0.
    tEnd = 1
    yo = array([0.])
    tau = 0.01
    z=1
    t, yn = rungeKutta(f, to, yo, tEnd, tau)
    y1n=[i[0] for i in yn]
    plt.figure()
    plt.title("Абсолютная погрешность численного решения(т.р.- u(t)=tan(t)) ДУ\n\
    du/dt=u**2+1 c u(0)=0 при t>0")
    plt.plot(t,abs(array(y1n)-array(tan(t))),label='Метод Рунге—Кутта \n\
    четвертого порядка - расчёт по алгоритму')
    plt.xlabel('Время')
    plt.ylabel('Абсолютная погрешность.')
    plt.legend(loc='best')
    plt.grid(True)
    z=0
    t, ym = rungeKutta(f, to, yo, tEnd, tau)
    y1m=[i[0] for i in ym]
    plt.figure()
    plt.title("Абсолютная погрешность численного решения(т.р.- u(t)=tan(t)) ДУ\n\
    du/dt=u**2+1 c u(0)=0 при t>0")
    plt.plot(t,abs(array(y1m)-array(tan(t))),label='Метод Рунге—Кутта— Фельберга \n\
    пятого порядка - расчёт по алгоритму')
    plt.xlabel('Время')
    plt.ylabel('Абсолютная погрешность.')
    plt.legend(loc='best')
    plt.grid(True)
    plt.figure()
    plt.title("Абсолютная погрешность численного решения (т.р.- u(t)=tan(t)) ДУ\n\
    du/dt=u**2+1 c u(0)=0 при t>0")
    y,t=odein()
    plt.plot(t,abs(array(tan(t))-array(y)),label='Функция odein')
    plt.xlabel('Время')
    plt.ylabel('Абсолютная погрешность.')
    plt.legend(loc='best')
    plt.grid(True)
    plt.figure()
    plt.title("Абсолютная погрешность численного решения (т.р.- u(t)=tan(t)) ДУ\n\
    du/dt=u**2+1 c u(0)=0 при t>0")
    z,t=oden()
    plt.plot(t,abs(tan(t)-z),label='Функция ode метод Рунге—Кутта— Фельберга \n\
    пятого порядка')
    plt.xlabel('Время')
    plt.ylabel('Абсолютная погрешность.')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
'''

print('-----------------------------------')
print(interpolating(1.25))
print('-----------------------------------')
#kubspline(1.25)
print('-----------------------------------')
print('Runge-Kutta dlya ODU method:')
RungeKuttaDlyaSistem()
print('-----------------------------------')
#RUN()






