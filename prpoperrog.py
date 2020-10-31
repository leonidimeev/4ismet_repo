import numpy as np
 
h=0.1
tau=0.01
 
def u(h,tau):
    x=[0]
    z=[0]
    t=[0]
    for i in range(0,11):
        x.append(x[i]+h)
        z.append(z[i]+h)
        t.append(t[i]+tau)
    y=[]
    Y=[]
    for i in range(0,11):
        Y.append([])
        y.append([])
        for j in range(0,11):
            Y[i].append(0)
            y[i].append([])
            for k in range(0,11):
                y[i][j].append(0)
 
    for i in range(0,11):
        for j in range(0,11):
            y[i][j][0]=x[i]+2*z[j]
    # for i in range(0,11):
    #     for j in range(2,11):
    #         y[0][i][j]=np.sin(t[i])
    #         y[10][i][j]=np.sin(t[i])
    #         y[i][0][j]=np.sin(t[i])
    #         y[i][10][j]=np.sin(t[i])
     
    a=[]
    b=[]
    c=[]
    d=[]
    for i in range(0,11):
        a.append([])
        b.append([])
        c.append([])
        d.append([])
        for j in range(0,11):
            a[i].append(0)
            b[i].append(0)
            c[i].append(0)
            d[i].append(0)
     
    for k in range(0,10):

        for j in range(1,10):
            A=[0,0]
            B=[0,y[0][j][k+1]]
            Y[0][j]=np.sin(t[k])
            Y[10][j]=np.sin(t[k])
            Y[j][0]=np.sin(t[k])
            Y[j][10]=np.sin(t[k])

            for i in range(1,10):
                a[i][j]=-0.5*tau
                b[i][j]=a[i][j]
                c[i][j]=h*h+tau
                d[i][j]=0.5*tau* (y[i][j+1][k]-2*y[i][j][k]+y[i][j-1][k]) +h*h*y[i][j][k]+3*t[k]*h*h*0.5*tau

                A.append(b[i][j]/(c[i][j]-a[i][j]*A[i]))
                B.append((a[i][j]*B[i]+d[i][j])/(c[i][j]-a[i][j]*A[i]))
            for i in range(9,0,-1):
                Y[i][j]=A[i+1]*Y[i+1][j]+B[i+1]
     
            for i in range(1,10):
                A=[0,0]
                B=[0,y[i][0][k+1]]
                for j in range(1,10):
                    a[i][j]=-0.5*tau
                    b[i][j]=a[i][j]
                    c[i][j]=h*h+tau
                    d[i][j]=0.5*tau*(Y[i+1][j]-2*Y[i][j]+Y[i-1][j])+h*h*Y[i][j]
                    A.append(b[i][j]/(c[i][j]-a[i][j]*A[j]))
                    B.append((a[i][j]*B[j]+d[i][j])/(c[i][j]-a[i][j]*A[j]))
                for j in range(9,0,-1):
                    y[i][j][k+1]=A[j+1]*y[i][j+1][k+1]+B[j+1]
    return(y)
 
 
y=u(0.1,0.01)
tr=u(0.2,0.01)
for i in range(0,11):
    for j in range(0,11):
        for k in range(0,11):
            print(y[i][j][k]-tr[i][j][k])



# Ut = 2*Uxx + 3*Uyy (x,y,t) in R x R x (0,T)
# U(x,y,0) = x+ 2*y  (x,y) in R2
# T > 0 


































