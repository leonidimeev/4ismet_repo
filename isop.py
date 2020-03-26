

A = [[0,0,0,0],[16,16,16,18],[20,23,21,18],[23,26,25,27],[34,33,31,35],[44,42,41,45]]
kom = []
sumkom = []
test = []
z = []
for i in range(6):
	for j in range(4):
		for k in range(6):
			for l in range(4):
				for p in range(6):
					for o in range(4):
						for u in range(6):
							for y in range(4):
								if ((i+k+p+u)==5) and (j!=k) :
									kom.append(A[i][j] + A[k][l] + A[p][o] + A[u][y])
									sumkom.append([i,k,p,u])
									test.append([j,k,o,y])
max = kom[0]
maxi = -1
n = 0
for n in range(len(kom)):
	if kom[n] > max:
		max = kom[n]
		maxi = n
print(max,sumkom[n],test[n])