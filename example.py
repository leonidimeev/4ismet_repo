import matplotlib.pyplot as plt

f = lambda x, y : x - y
g = lambda x, y : y - x**2

A = []
for i in range(10):
	A.append([])
	for j in range(10):
		A[i].append(0)
for line in A:
	print(line)

for i in range(10):
	for j in range(10):
		A[i][j] = g(i,j)

plt.plot(A)
plt.show()