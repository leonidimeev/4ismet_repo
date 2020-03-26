
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

def det2x2(a,b,c,d):
	return ( (a*c) - (b*d) )

def X1(x1,x2):
    try:
        return ( x1 - (det2x2(f1(x1,x2),F1_x2(x1,x2),f2(x1,x2),F2_x2(x1,x2)))/(det2x2(F1_x1(x1,x2),F1_x2(x1,x2),F2_x1(x1,x2),F2_x2(x1,x2)))   )
    except ValueError:
        print('MaybeKorenb')
        return 0

def X2(x1,x2):
    try:
        return ( x2 - (det2x2(F1_x1(x1,x2),f1(x1,x2),F2_x1(x1,x2),f2(x1,x2)))/(det2x2(F1_x1(x1,x2),F1_x2(x1,x2),F2_x1(x1,x2),F2_x2(x1,x2)))   )
    except ValueError:
        print('MaybeKorenb')
        return 0
i = 0
x1 = 1.0
x2 = 0.5
e = pow(10,-30)

#while x1-X1(x1,x2)>e or x2-X2(x1,x2)>e:
while i<=9:
	xi = X1(x1,x2)
	xj = X2(x1,x2)
	x1 = xi
	x2 = xj
	i += 1

print(x1,x2,i)
