#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8

# Из курса по численным методам, 3 курс, 2 семестр
# Уравнения математической физики и их решение


def GridMaker( M, N, tau, l): # Построитель сеток

    h = l / N 
    #dx
    x = []
    for i in range(N + 1):
        x.append(i * h)
    #dt
    t = []
    for j in range(M + 1):
        t.append(j * tau)
    #building a matrix
    Ut = []
    for i in range(N + 1):
        Ut.append([])
        for j in range(M + 1):
            Ut[i].append(0)
    return(Ut)

#ЛАБОРАТОРНАЯ РАБОТА № 1

def numericalmethods_lab_1():

    Uxt = lambda t, x, a : 4*(x**3+6*a*t*x)+2
    U0t = lambda t, a : Uxt(t,0,a)
    U1t = lambda t,a : Uxt(t,1,a)
    Ux0 = lambda x,a : Uxt(0,x,a)


    M = 10
    N = 10
    a = 1.
    T = 0.01
    tau = T/M
    l = 1.
    h = l / N

    Ut = GridMaker(M,N,tau,l)

    #bottom line
    for i in range(N + 1):
        Ut[0][i] = Ux0(i*h,a)
    #gamma for comfort
    gamma = a*(tau/(h**2))
    #walls
    for j in range(M + 1):
        Ut[j][0] = U0t(j*tau,a)
        Ut[j][N] = U1t(j*tau,a)

    # Функция вывода разницы между элементами полученной сетки и точных значений
    def output(Ut):
        for j in range(M + 1):
            print('-----------------')
            for i in range(N + 1):
                print(abs(Ut[j][i] - Uxt(j*tau, i*h, a)))


    # По явной схеме
    def explicit():
        #calculation
        for j in range(M):
            for i in range(1, N):
                Ut[j+1][i] = Ut[j][i] + gamma*(Ut[j][i+1] - 2*Ut[j][i] + Ut[j][i-1])
        output(Ut)
        
    # По неявной схеме
    def unexplicit():

        #koefs of tridiagonal matrix
        A = gamma
        B = gamma
        C = 2 * gamma + 1 
        #calculation
        for j in range(1,M+1):
            alpha = [0]
            beta = [Uxt(j*tau,0,a)]
            for i in range(N+1):
                alpha.append(B/(C - alpha[i]*A))
                beta.append((A*beta[i] + Ut[j-1][i])/(C - alpha[i]*A))
            for i in reversed(range(1,N)):
                Ut[j][i] = alpha[i+1]*Ut[j][i+1] + beta[i+1]
        output(Ut)

    print('------------------------------------')
    print(' :')
    unexplicit()
    print('------------------------------------')
    print(' :')
    explicit()


#ЛАБОРАТОРНАЯ РАБОТА №2

def numericalmethods_lab_2():

    from math import cos, pi 

    Utx0 = lambda x : 2*x + 1 #F(x)

    U1t = lambda t : 0 #psi(x)
    U0t = lambda t : 2*t + 1 #fi(x)
    Ux0 = lambda x : (1 - x)*cos(pi*x/2) #f(x)


    M = 10
    N = 10
    a = 1.
    T = 0.01
    tau = T/M
    l = 1.

    Ut = GridMaker(M,N,tau,l)

    #gamma for comfort
    gamma = a*(tau/(h**2))

    # По явной схеме
    def explicit():
        #calculation
        for j in range(M):
            for i in range(1, N):
                Ut[j+1][i] = Ut[j][i] + gamma*(Ut[j][i+1] - 2*Ut[j][i] + Ut[j][i-1])
        output(Ut)

    def alpha_solution(alpha, beta, a, b, c):
        for i in range(1,n):
            alpha[i] = b/(c-a*alpha[i-1])

    def beta_solution(alpha, beta, a, b, c):
        for i in range(1,n):
            beta[i] = (a * beta[i-1] + alpha[i-1])/(c - a*alpha[i-1])

    alpha = []
    beta = []
    alpha.append(0)
    beta.append(Ut[0])
    a = tau*tau/h*h
    b = a
    c = 2*a + 1
    for line in Ut:
        print(line)






def numericalmethods_lab_3_GRAMMAR():

    #f(x,y) = sin2(pixy)
    #myu(0,y) = myu(1,y) = sin(piy)
    #myu(x,0) = myu(x,1) = x(1-x)
   
    from numpy import sin
    from math import pi
    import numpy as np

    f = lambda x, y : (sin(pi*x*y))**2
    myu = lambda x, y : x*(1-x) + sin(pi*y)


    def Jacobi_sol(A, eps=0.001):

        iteration = 0

        max_eps = eps
        u = []
        for i in range(N+1):
             u.append([])
             for j in range(M+1):
                 u[i].append(0)
        u0 = Ut

        while max_eps >= eps:
             max_eps = 0
             for i in range(1, N):
                 for j in range(1, M):
                     u[i][j] = (u0[i][j-1] + u0[i][j+1] + u0[i-1][j] + u0[i+1][j] + h*h*f(h*j, h*i))/4
                     if abs(f(h*j, h*i) - u[i][j]) > max_eps:
                         max_eps = abs(f(h*j, h*i) - u[i][j])
             iteration += 1
             u0 = []
             for i in range(N+1):
                 u0.append([])
                 for j in range(M+1):
                     u0[i].append(u[i][j])
             if iteration % 100 == 0:
                 print(max_eps)
        print(max_eps)


    M = 10
    N = 10
    a = 1.
    T = 0.01
    tau = T / M
    l = 1.
    
    Ut = GridMaker(M,N,tau,l)

    #bottom line
    for i in range(N + 1):
        Ut[0][i] = myu(i*h,a)
    #gamma for comfort
    gamma = a*(tau/(h**2))
    #walls
    for j in range(M + 1):
        Ut[j][0] = myu(j*tau,a)
        Ut[j][N] = myu(j*tau,a)
    

    Jacobi_sol(Ut)
    for line in Ut:
        print(line)

# Листок Вар2 Татьяна Семеновна 23/04/2020
def lab2():

    from math import exp
    Uxt = lambda x, t, C : C * exp(x + a*t)
    Ux0 = lambda x : exp(x)

    U_t_x0 = lambda x : exp(x)

    # Границы
    U0t = lambda t : exp(t)
    U1t = lambda t : exp(1+t)

    c = 1
    a = 1

    M = 10
    N = 10
    a = 1.
    T = 0.01
    tau = T/M
    l = 1.
    h = l / N

    # Построитель сеток
    Ut = GridMaker(M,N,tau,l)

    #bottom line
    for i in range(N + 1):
        Ut[0][i] = Ux0(i*h)
        Ut[1][i] = Ut[0][i] + tau*exp(i*h)

    #gamma for comfort
    gamma = a*(tau**2/(h**2))
    #walls
    for j in range(M + 1):
        Ut[j][0] = U0t(j*tau)
        Ut[j][N] = U1t(j*tau)

    # Функция вывода разницы между элементами полученной сетки и точных значений
    def output(Ut):
        for j in range(M + 1):
            print('-----------------')
            for i in range(N + 1):
                print(abs(Ut[j][i] - Uxt(1, j*tau, i*h)))

    # По явной схеме
    def explicit():
        #calculation
        for j in range(1, M):
            for i in range(1, N):
                Ut[j][i] = 2*Ut[j][i] - Ut[j-1][i] + gamma*(Ut[j][i+1] - 2*Ut[j][i] + Ut[j][i-1])
        output(Ut)
        
        
    # По неявной схеме
    def unexplicit():

        #koefs of tridiagonal matrix
        A = 1
        B = gamma
        C = 1 
        #calculation
        for j in range(1,M+1):
            alpha = [0]
            beta = [Uxt(j*tau,0,a)]
            for i in range(N+1):
                alpha.append(B/(C - alpha[i]*A))
                beta.append((A*beta[i] + Ut[j-1][i])/(C - alpha[i]*A))
            for i in reversed(range(1,N)):
                Ut[j][i] = alpha[i+1]*Ut[j][i+1] + beta[i+1]
        output(Ut)

    # print('------------------------------------')
    # print(' :')
    # unexplicit()
    print('------------------------------------')
    print(' :')
    explicit()

# Отредактированное
def TTS():
	import numpy as np

	c=1; a=1; M=10; N=10; a=1.; T=1.; tau=T/M; l=1.; h=l/N; r=(tau**2)/(h**2)

	y=np.zeros((M+1,N+1),'float'); x=np.zeros((N+1),'float'); t=np.zeros((M+1),'float')
	for i in range(N+1): x[i]=i*h
	for j in range(M+1): t[j]=j*tau
	for i in range(N+1): y[0,i]=np.exp(x[i]); y[1,i]=y[0,i]+tau*np.exp(x[i])        # Initial conditions
	for j in range(1, M):                                                           # calculation
	    y[j,0]=np.exp(t[j]); y[j,N]=np.exp(l+t[j])                                  # walls
	    for i in range(1, N): y[j+1,i]=2*y[j,i]-y[j-1,i]+r*(y[j,i+1]-2*y[j,i]+y[j,i-1])
	for j in range(2,M + 1):
	    print (j,t[j])
	    for i in range(N+1): print(abs(y[j,i]-np.exp(x[i]+t[j])))

# Эллиптический переведенный с С++
def Elliptic():
	from numpy import sin
	from math import pi
	f = lambda x, y : (sin(pi*x*y))**2
	x = []
	y = []
	u = []
	v = []
	z = []
	for i in range(11):
		u.append([])
		z.append([])
		v.append([])
		for j in range(11):
			u[i].append(0)
			z[i].append(0)
			v[i].append(0)
	for i in range(11):
		x.append(i*0.1)
		y.append(i*0.1)
		u[i][0] = x[i]*(1-x[i])
		u[i][10] = u[i][0]
		u[0][i] = y[i]*(1-y[i])
		u[10][i] = u[0][i]

	max1 = 123
	while (max1>0.001):
		for i in range(11):
			v[i][0] = x[i]*(1-x[i])
			v[i][10] = v[i][0]
			v[0][i] = y[i]*(1-y[i])
			v[10][i] = v[0][i]
		for i in range(1,10):
			for j in range(1,10):
				v[i][j] = (f(x[i],y[i]) * 0.1 * 0.1 + u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])/4.
				z[i][j] = abs(v[i][j] - u[i][j])
		max1 = 0
		for i in range(1,10):
			for j in range(1,10):
				if (z[i][j] >= max1):
					max1 = z[i][j]
		for i in range(1,10):
			for j in range(1,10):
				u[i][j] = v[i][j]
	for i in range(11):
		for j in range(11):
			print(v[i][j])
		print('\n')

def Perenos():
	from numpy import exp
	tr = lambda a,x,t : a*exp(x+t)
	u0 = lambda a,x : a*exp(x)
	myu = lambda a,t : a*exp(t)
	f = lambda a,x,t : 2*a*exp(x+t)
	N = 10; M = 10; tau = 0.001; l = 1.; a = 1; h = 0.1
	U = GridMaker( N, M, tau, l)
	# Нижняя граница ( t = 0 )
	for x in range(11):
	    U[x][0] = u0(a,x*h)
	#wall
	for t in range(11):
	    U[0][t] = myu(a,t*tau)
	for i in range(1,11):
		for j in range(10):
			U[i][j+1] = ((U[i][j]/tau) + (U[i-1][j+1]/h) + f(a,i*h,j*tau))/((1/tau) + (1/h) - 1)  
	for line in U:
		print(line),
	for i in range(11):
	    print('-----------------')
	    for j in range(11):
	        print((abs(U[i][j] - tr(1, i*h, j*tau))), end = ' ')



if __name__ == "__main__":

    Perenos()

#            Ut[i][j+1] = 2*Ut[i][j] - Ut[i][j-1] + (gamma**2)*(Ut[i+1][j] - 2*Ut[i][j] + Ut[i-1][j])
#numericalmethods_lab_1()









