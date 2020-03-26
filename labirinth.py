import random

def Get_Type(x):
	if x % 3 == 0:
		return ('+0')
	elif ( N - 1 ) % 3 == 0:
		return ('+1')
	elif ( N - 2 ) % 3 == 0:
		return ('+2')
	else:
		return ('TYPE ERROR')

def Get_End_Type(N,M):
	if M % 2 == 0:
		return ('With 2')
	else:
		return ('With 1')

def Rand():
	return random.randint(0,3)

def ELEMENT_INDEX(x):
	return x

array = []
'''
N_type = {
	'+0':Type_1,
	'+1':Type_2,
	'+2':Type_3
}

M_type = {
	'+0':Type_1,
	'+1':Type_2,
	'+2':Type_3
}

Pre_end_type ={
	'With 1': Pre_end_1,
	'With 2': Pre_end_2
}
'''

N1 = int(input("Введите N -->"))
M1 = int(input("Введите M -->"))
f = True


Nf1 = False
Nf2 = False
Mf1 = False
Mf2 = False

if N1 % 3 == 1:
	Nf1 = True
	N = (N1 // 3) + 1
elif N1 % 3 == 2:
	Nf2 = True
	N = (N1 // 3) + 1
else:
	N = (N1 // 3)
if M1 % 3 == 1:
	Mf1 = True
	M = (M1 // 3) + 1 
elif M1 % 3 == 2:
	Mf2 = True
	M = (M1 // 3) + 1 
else:
	M = (M1 // 3)

Ntype = Get_Type(N)
Mtype = Get_Type(M)


for i in range(0,N):
	array.append([])
	for j in range(0,M):
		array[i].append(0)
		if i == N - 2 and j == M - 1:
			if Get_End_Type(N,M) == 'With1':
				if Mf1:
					array[i][j] = 'p3-1-1' # последняя в предпоследней строке - тип с 1 верхний остаток 1
				elif Mf2:
					array[i][j] = 'p3-2-1' # последняя в предпоследней строке - тип с 1 верхний остаток 2
				else:
					array[i][j] = 'p3-3-1' # последняя в предпоследней строке - тип с 1 верхний остаток 2
				if f:
						f = False
				else:
						f = True
			else: 
				if Mf1:
					array[i][j] = 'p3-1-2' # последняя в предпоследней строке - тип с 2 верхний остаток 1
				elif Mf2:
					array[i][j] = 'p3-2-2'
				else:
					array[i][j] = 'p3-3-2' # последняя в предпоследней строке - тип с 2 верхний остаток 1
				if f:
						f = False
				else:
						f = True
		elif i == N - 1 and j == M - 2:
			if Get_End_Type(N,M) == 'With1':
				if Nf1:
					array[i][j] = 'p1-3-1' # препоследняя в последней строке - тип с 1 нижний
					End_Type = 'With 1'
				elif Nf2:
					array[i][j] = 'p2-3-1'
					End_Type = 'With 1'
				else:
					array[i][j] = 'p3-3-1'
					End_Type = 'With 1'
			else:
				if Nf1:
					array[i][j] = 'p1-3-2' # препоследняя в последней строке - тип с 2 нижний
					End_Type = 'With 2'
				elif Nf2:
					array[i][j] = 'p2-3-2'
					End_Type = 'With 2'
				else:
					array[i][j] = 'p3-3-2'
					End_Type = 'With 1'

		elif i == N - 1 and j == M - 1:
			if End_Type == 'With 1':
				if Mf1 and Nf1:	
					array[i][j] = 'posl1-1-1' # последняя тип 1
				elif Mf2 and Nf1:
					array[i][j] = 'posl1-2-1'
				elif Mf1 and Nf2:
					array[i][j] = 'posl2-1-1'
				elif Mf2 and Nf2:
					array[i][j] = 'posl2-2-1'
				else:
					array[i][j] = 'posl3-3-1'
			else:
				if Mf1 and Nf1:	
					array[i][j] = 'posl1-1-2' # последняя тип 1
				elif Mf2 and Nf1:
					array[i][j] = 'posl1-2-2'
				elif Mf1 and Nf2:
					array[i][j] = 'posl2-1-2'
				elif Mf2 and Nf2:
					array[i][j] = 'posl2-2-2'
				else:
					array[i][j] = 'posl3-3-2' # последняя тип 2
		elif (j == 0) and (i != N-1):
			if M % 2 == 1:
				if f:
					array[i][j] = '3-3-2'
					f = False
				else:
					array[i][j] = '3-3-1'
					f = True
			else:
				if f:
					array[i][j] = '3-3-1'
					f = True
				else:
					array[i][j] = '3-3-2'
					f = False
		else:
				
			if f:
				if i == N - 1:
					if Nf1:
						array[i][j] = '1-3-2' #нижние если остаток 1 (1)
						f = False
					elif Nf2:
							array[i][j] = '2-3-1' #нижние если остаток 2 (1)
							
							f = False
					else:
						array[i][j] = '3-3-1'
						f = False
				elif j == M - 1:
					if Mf1:
						
							array[i][j] = '3-1-2' #правые если остаток 1 (1)
							
							f = False
					elif Mf2:
						
							array[i][j] = '3-2-2' #правые если остаток 2 (1)
							
							f = False
					else:
						array[i][j] = '3-3-1'
						f = False

				if i != N-1 and j != M-1:
					array[i][j] = '3-3-2'
					f = False
			else:
				if i == N - 1:
					if Nf1:
						
							array[i][j] = '1-3-1' #нижние если остаток 1 (1)
							
							f = True
					elif Nf2:
						
							array[i][j] = '2-3-2' #нижние если остаток 2 (1)
						
							f = True
					else:
						array[i][j] = '3-3-2'
						f = True
				elif j == M - 1:
					if Mf1:
						
							array[i][j] = '3-1-1' #правые если остаток 1 (1)
						
							f = True
					elif Mf2:
							array[i][j] = '3-2-1' #правые если остаток 2 (1)
							
							f = True
					else:
						array[i][j] = '3-3-2'
						f = True

				if (i != N-1) and (j != M-1):
					array[i][j] = '3-3-1'
					f = True

'''
Types_array = []
for i in range(0,N):
	Types_array.append([])
	for j in range(0,M):
		Types_array[i].append(0)

		if array[i][j] == 1:
			Types_array[i][j] = random.randint(1,3)
		elif array[i][j] == 2:
			Types_array[i][j] = random.randint(4,6)
		else:
			Types_array[i][j] = 0

'''
i = 0
while i < N:
	for index in array[i]:
		print (index, end=' ')
	i += 1
	print('\n')

print('---------------------------------------')

'''
i = 0
for index in Types_array:
	while i < N:
		for index in Types_array[i]:
			print (index, end=' ')
		i += 1
		print('\n')

labirinth = []																#[1]
for i in range(0,N):														
	labirinth.append([])													#[[11]
	for j in range(0,M):													#[11]]
		labirinth[i].append([])												#[[1][1]
		for TOP_WALL in range(0,5):
			labirinth[i][j].append([])										#[1][1]]
			for SIDE_WALL in range(0,5):
				labirinth[i][j][TOP_WALL].append([])
				labirinth[i][j][TOP_WALL][SIDE_WALL].append(ELEMENT_INDEX(array[i][j]))						

for i in range(0,N):
	for j in range(0,M):
		for k in range(0,5):
			for l in range(0,5):
				print(labirinth[i][j][k][l], end=' ')
			print('\n')
'''	







