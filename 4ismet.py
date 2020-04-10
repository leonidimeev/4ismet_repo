
def GridMaker( M, N, tau, l):
    h = l / N 
    #dx
    x = []
    for i in range(N + 1):
        x.append(i * h)
    #dt
    t = []
    for j in range(M + 1):
        t.append(j * tau)
    #buildin a matrix
    Ut = []
    for i in range(N + 1):
        Ut.append([])
        for j in range(M + 1):
            Ut[i].append(0)
    return(Ut)


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

    def output(Ut):
        for j in range(M + 1):
            print('-----------------')
            for i in range(N + 1):
                print(abs(Ut[j][i] - Uxt(j*tau, i*h, a)))

    def explicit():
        #calculation
        for j in range(M):
            for i in range(1, N):
                Ut[j+1][i] = Ut[j][i] + gamma*(Ut[j][i+1] - 2*Ut[j][i] + Ut[j][i-1])
        output(Ut)
        

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


def numericalmethods_lab_2():

    from math import cos, pi 

    Utx0 = lambda x : 2*x + 1

    U1t = lambda t : 0
    U0t = lambda t : 2*t + 1
    Ux0 = lambda x : (1 - x)*cos(pi*x/2)


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
        Ut[0][i] = Ux0(i*h)
    #gamma for comfort
    gamma = a*(tau/(h**2))
    #walls
    for j in range(M + 1):
        Ut[j][0] = U0t(j*tau)
        Ut[j][N] = U1t(j*tau)

    def output(Ut):
        for j in range(M + 1):
            print('-----------------')
            for i in range(N + 1):
                print(abs(Ut[j][i] - Uxt(j*tau, i*h, a)))


    def explicit():
        #calculation
        for j in range(M):
            for i in range(1, N):
                Ut[j+1][i] = Ut[j][i] + gamma*(Ut[j][i+1] - 2*Ut[j][i] + Ut[j][i-1])
        output(Ut)
        

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

    explicit()



def numericalmethods_lab_3_GRAMMAR():

    #f(x,y) = sin2(pixy)
    #myu(0,y) = myu(1,y) = sin(piy)
    #myu(x,0) = myu(x,1) = x(1-x)
   
    from numpy import sin
    from math import pi
    import numpy as np

    f = lambda x, y : (sin(pi*x*y))**2
    myu = lambda x, y : x*(1-x) + sin(pi*y)

    #matrixmul(A,B) - произведение матриц
    def matrixmul(A,B):
        if (len(A[0])!=len(B)):
            return None
        Bt=trans(B)
        n,m=len(A),len(Bt)
        C=zeros(n,m)
        for i in range(0,n):
            for j in range(0,m):
                C[i][j]=scalmul(A[i],Bt[j])
        return C


    def Jacobi_sol_GRAMMAR(A,b,epsilon=1e-10,x0=None):

        from pandas import DataFrame
        #sub(x,y) - поэлементное вычитание векторов
        sub=lambda x,y:map(lambda a,b:a-b,x,y)
        #mulmatbyarr(M,A) - произведение матрицы на вектор (удобно)
        mulmatbyarr = lambda M,A:trans(matrixmul(M,trans([A])))[0]
        #trans(X) - транспонирование матрицы
        trans=lambda X:map(list,DataFrame.apply(zip,X))
        #norma(x) - норма вектора (корень скалярного квадрата)
        norma=lambda x:pow(scalmul(x,x),0.5)
        #scalmul(x,y) - скалярное произведение векторов
        scalmul=lambda x,y:reduce(lambda a,b:a+b,map(lambda a,b:a*b,x,y),0.0)


        if (len(A)!=len(A[0])) or (len(A)!=len(b)):
            return None

        Mx=lambda r,A=A:map(lambda rj,j,A=A:rj/A[j][j],r,range(0,len(r)))

        k,xk=0,x0
        if (x0==None): xk=b[:]
        rk=sub(b,mulmatbyarr(A,xk))
        while (norma(rk)>epsilon):
            xk=add(xk,Mx(rk))
            k+=1
            rk=sub(b,mulmatbyarr(A,xk))
        print(xk)
        return(xk)

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
    h = l / N

    #dx
    x = []
    for i in range(N + 1):
        x.append(i * h)
    #dt
    t = []
    for j in range(M + 1):
        t.append(j * tau)
    #buildin a matrix
    Ut = []
    for i in range(N + 1):
        Ut.append([])
        for j in range(M + 1):
            Ut[i].append(0)
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

    


#fynjy
#ktyz
#fynjy 

if __name__ == "__main__":

    numericalmethods_lab_2()

#            Ut[i][j+1] = 2*Ut[i][j] - Ut[i][j-1] + (gamma**2)*(Ut[i+1][j] - 2*Ut[i][j] + Ut[i-1][j])
#numericalmethods_lab_1()









