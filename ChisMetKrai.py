##u = k*exp(x)
##ddu + p*du - q*u - k*exp(x)*(1+p-q)
import math  

def MetodProgonki():
	P = [b[0]/c[0]]
	Q = [-d[0]/c[0]]
	for i in range(1, n-1):
		P.append(b[i]/(-a[i]*P[i-1] + c[i]))
		Q.append((-d[i] + a[i]*Q[i-1])/float((-a[i]*P[i-1] + c[i])))
	x = [(-d[n-1] + a[n-1]*Q[n-2])/(-a[n-1]*P[n-2] + c[n-1])]
	for i in range(n-1):
		x.append(P[n-2-i]*x[i] + Q[n-2-i])
	return list(reversed(x))

        
    

h = 1/10
n = 10
x = []
for i in range(n):
	x.append(h*i)


k = 1
p = 1
q = 10
##(y[1] - y[0])/h=1
##(y[N-1] - y[N])/h=exp(1)
b1 = 1
b2 = 1*math.exp(1)
##y[0] = P1*y[1] + M1
##y[N] = P1*y[N-1] + M1
a = [0]
c = [1]
b = [1]
d = [b1*h]
for i in range(1,n-1):
	a.append(1/h/h - p/h)
	b.append(1/h/h)
	c.append(2/h/h - p/h + q)
	d.append(k*math.exp(x[i])*(1+p-q))
a.append(1)
c.append(1)
b.append(0)
d.append(-b2*h)

resh = MetodProgonki()
for i in range(n):
	print((resh[i]-k*math.exp(x[i]))/resh[i])