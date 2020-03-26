##a[i]*x[i-1] - c[i]*x[i] + b[i]*x[i+1] = d[i], a[0] = b[n-1] = 0
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






n = 4
a = [0,1,1,1]
b = [1,-5,2,0]
c = [-2,-10,5,-4]
d = [-5,-18,-40,-27]
x = MetodProgonki()
print(x)