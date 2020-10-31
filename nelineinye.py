from numpy import tan, cos, exp
 
xi = 1
l = 1.
ro = lambda u : u
k = lambda u : u*u
q = lambda u: u+u*u
m1 = lambda t : exp(t)
m2 = lambda t, l : exp(l+t)
tr = lambda x, t : exp(x+t)
 
n = 11
h = 0.1
tau = 0.001
x = []; u = []; t = []; al = []; be = []
 
for i in range(n):
	x.append(0.)
	u.append([])
	t.append(0.)
	al.append(0.)
	be.append(0.)
	for j in range(n):
		u[i].append(0.)
 
for i in range(n):
	x[i] = i*h
	u[i][0] = exp(x[i])
	t[i] = i*tau
 
for j in range(n-1):
	u[0][j+1] = m1(t[j+1])
	u[10][j+1] = m2(t[j+1],l)
	al[1] = 0.
	be[1] = u[0][j+1]
	for i in range(1,n-1):
		a = tau/(h*h*ro(u[i][j])) * k((u[i-1][j] + u[i][j])/2.) 
		b = tau/(h*h*ro(u[i][j]))*k((u[i+1][j] + u[i][j])/2.) 
		c = 1.+a+b+q(u[i][j])*tau/ro(u[i][j])
		d = u[i][j] 
		al[i+1] = b/(c-a*al[i])
		be[i+1] = (a*be[i]+d)/(c-a*al[i])
	 
	for i in reversed(range(n-1)):
		u[i][j+1] = u[i+1][j+1]*al[i+1]+be[i+1]
 
	for i in range(n):
		print(u[i][j+1] - tr(x[i],t[j+1]), ' \n')













