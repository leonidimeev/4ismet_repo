import math
def f1():
	return math.sin(x - 0.8) - 2*y - 1.6

def f2():
	return 3*x - math.cos(y) - 0.9

def det(a, b, c, d):
	return a*d - b*c

def f1x():
	return math.cos(x - 0.8)

def f1y():
	return -2

def f2x():
	return 3

def f2y():
	return math.sin(y)

def J():
	return det(f1x(), f1y(), f2x(), f2y())

def A1():
	return det(f1(), f1y(), f2(), f2y())

def A2():
	return det(f1x(), f1(), f2x(), f2())


x = 0.4
y = -0.9

xPred = 0
yPred = 0

e = 0.001
k = 0
while abs(x - xPred) > e and abs(y - yPred) > e:
	k += 1
	xPred = x
	yPred = y
	x = x - A1()/J()
	y = y - A2()/J()

print(k, f1(), f2())






