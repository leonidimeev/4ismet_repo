import math
def f(x):
    return x**3 - 3*x*x - 3*x + 10


    
def dy(i, j):
    if i+1 < j:
        return (dy(i+1, j) - dy(i, j-1))/(X[j] - X[i])
    elif i+1 == j:
        return (Y[j]-Y[i])/(X[j] - X[i])
    elif i == j:
        return Y[i]  


def Newton(x):
    S = 0
    product = 1
    for i in range(31):
##        S += product*dy(0, i)
        S += product*du[i][i]
        product *= x-X[i]  
    return S




        
X = []
Y = []
for i in range(31):
    X.append(-1 + 0.2*i)
    Y.append(f(-1 + 0.2*i))


du = [[]]
for i in range(31):
    du[0].append(Y[i])
for i in range(1, 31):
    du.append([])
    for j in range(i):
        du[i].append(0)
    for j in range(i, 31):
        du[i].append((du[i-1][j]-du[i-1][j-1])/(0.2*i)) 

       
for i in range(25):
    x = -1 + 0.25*i
    y = f(x)
    print(abs(y-Newton(x)), "\n")

        

            

     
      