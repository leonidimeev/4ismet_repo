
#
#        --------------------------------------
#        ||||||||||||||DISCLAIMER||||||||||||||
#        Leonya`s Programming Practice Work`s 
#        Sda4a Program (LPPWSP)
#        This software comes with no guarantee
#        Use at your own risk
#        --------------------------------------
#        Yakutsk, 2019
#

import sys
from tkinter import *
import math
import string
import numpy as np
import os
from datetime import date
import calendar
from scipy import integrate
import random
import glob

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib as mmm
    mmm.use("TkAgg")

import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D




def clickAbout(): 
    name = ("click")
    return

def Error_Message_Box_Input():
    messagebox.showinfo('Error, check your input')

def Input_exc_num():
    input_lesson = inputzad.get()
    input_exc = inputnum.get()
    if (input_lesson == '')&(input_exc == ''):
        Error_Message_Box_Input()
    else:
        case(input_lesson,input_exc)

def case(les_num, exc_num):
  if les_num == 'zad1':
    if exc_num in ['1','2','3','4','5','6','7','8','9','10']:
      if exc_num in ['1']:
        functions['inp'](3,les_num,exc_num)
      if exc_num == '2':
        functions['inp'](4,les_num,exc_num)
      if exc_num in ['3']:
        functions['inp'](6,les_num,exc_num)
      if exc_num in ['4','5','6','7']:
        a = les_num+exc_num
        functions[a]()
        
      if exc_num == '8':
        functions['inp'](4,les_num,exc_num)
    else: Error_Message_Box_Input()
  if les_num == 'zad4':
    if exc_num in ['1','2','3','4','5','6','7','8']:
      if exc_num in ['1','2','3']:
        a = les_num+exc_num
        functions[a]()
  if les_num == 'zad5':
    if exc_num in ['1','2','3']:
      if exc_num in ['1','2','3']:
        a = les_num+exc_num
        functions[a]()
  if les_num == 'a':
      if exc_num in ['1','2','3','4','5','6','7']:
        a = les_num+exc_num
        functions[a]()
  if les_num == 'b':
      if exc_num in ['1','2','3','4','5','6','7','8','9','10','11','12','13']:
        a = les_num+exc_num
        functions[a]()
  if les_num == 'c':
      if exc_num in ['1','2','3','4','5','6']:
        a = les_num+exc_num
        functions[a]()
  if les_num == 'd':
      if exc_num in ['1','2','3','4','5','6','7','8','9','10','11','12']:
        a = les_num+exc_num
        functions[a]()
  if les_num == 'mp':
    if exc_num in ['1','2','3','4','5']:
        a = les_num+exc_num
        functions[a]()
  if les_num == 'g':
      if exc_num in ['1','2','3','4','5']:
        print('Current version on matplotlib library is', mmm.__version__)
        graphinit(exc_num)
  if les_num == 'six' : six()
  if les_num == 'ping':
    if exc_num == 'pong':
      ping_pong()
  if les_num == 'post':
    POST()
  if les_num == 'class':
    Class_lab()
  if les_num == 'point':
    Point_adventure()
  if les_num == 'quad':
    quadrangle_def()
  if les_num == 'maze':
  	Maze()

def inp(n,les_num,exc_num,i=0):
  a = int(input('-> '))
  data = []
  source = les_num + exc_num
  print(source)
  while True:
    try:
      data.append(a)
      a = int(input('-> '))
      i+=1
    except:
      break
  if i!=n-1 :
    print('Ошибка при вводе, повторить?')
    answer = input('----> ')
    if answer == 'Y':
      functions['inp'](n, les_num, exc_num)
  else:
    functions[source](*data)
    
def zad1_1(a,b,c):
  discr = b**2 - 4 * a * c;
  print("Дискриминант D = %.2f" % discr)
  if discr > 0:
    x1 = (-b + sqrt(discr)) / (2 * a)
    x2 = (-b - sqrt(discr)) / (2 * a)
    print("x1 = %.2f \nx2 = %.2f" % (x1, x2))
  elif discr == 0:
    x = -b / (2 * a)
    print("x = %.2f" % x)
  else:
    print("Корней нет")

def zad1_2(a,b,c,d):
  p = (-1 * b)/(3 * a) 
  q = p ** 3 + (b * c - (3 * a * d))/ (6 * (a ** 2)) 
  r = c/(3 * a) 

  x = (q + (q**2 + (r - p**2)**3) **1/2) **1/3 + (q + (q**2 + (r - p**2)**3) **1/2) **1/3 + p 
  print("Корень = ", x) 

  total = (a * x**3) + (b * x**2) + (c * x) + d 
  total = round(total, 3) 
  return total 

def zad1_3(a1,b1,c1,a2,b2,c2):
  if a1 == c2 and b1 == b2 and c1 == a2 or a1 == c2 and b1 == a2 and c1 == b2:
    print('Boxes are equal')
  elif a1 == a2 and b1 == b2 and c1 == c2 or a1 == a2 and b1 == c2 and c1 == b2:
    print("Boxes are equal")
  elif a1 == b2 and b1 == c2 and c1 == a2 or a1 == b2 and b1 == a2 and c1 == c2:
    print("Boxes are equal")
  elif a1 >= a2 and b1 >= b2 and c1 >= c2 \
        or a1 >= a2 and b1 >= c2 and c1 >= b2 \
        or a1 >= b2 and b1 >= a2 and c1 >= c2 \
        or a1 >= b2 and b1 >= c2 and c1 >= a2:
    print("The first box is larger than the second one")
  elif a1 >= c2 and b1 >= b2 and c1 >= a2 or a1 >= c2 and b1 >= a2 and c1 >= b2:
    print("The first box is larger than the second one")
  elif a1 <= a2 and b1 <= b2 and c1 <= c2 \
        or a1 <= a2 and b1 <= c2 and c1 <= b2 \
        or a1 <= b2 and b1 <= a2 and c1 <= c2 \
        or a1 <= b2 and b1 <= c2 and c1 <= a2:
    print("The first box is smaller than the second one")
  elif a1 <= c2 and b1 <= b2 and c1 <= a2 or a1 <= c2 and b1 <= a2 and c1 <= b2:
    print("The first box is smaller than the second one")
  else:
    print('Boxes are incomparable')

def zad1_4():
  center = False
  left = True
  right = False
  s = input("Ваша строка -->")
  print (s)
  i = 0
  while i<len(s):
      if s[i] == 'A':
        if center == True:
          center = False
          left = True
        if left == True:
          left = False
          center = True
          
      if s[i] == 'B':
        if center == True:
          center = False
          right = True
        if right == True:
          right = False
          center = True

      if s[i] == 'C':
        if left == True:
          left = False
          right = True
        if right == True:
          right = False
          left = True
      i+=1
  if center == True: print("В центре")
  if left == True: print("В левом")
  if right == True: print("В правом")

def zad1_5(i = 1, s = 0):
  n = int(input("Кол-во кубиков -->"))
  vspm = False
  while vspm == False:
    s = s + i
    i+=1
    if s >= n: vspm = True
    
  print('Максимальное высокое = ',i-1)
#1 2 3 4 5 6 7 8 9 10 11
#1 3 6 10 15 21 28 37 47 58

def zad1_6():
  s = input('Введите уравнение-->')
  if s[0] == 'x':
    if s[1] == '+':
      x = int(s[4]) - int(s[2])
    else:
      x = int(s[4]) + int(s[2])
  if s[2] == 'x':
    if s[1] == '+':
      x = int(s[4]) - int(s[0])
    else:
      x = int(s[0]) - int(s[4])
  if s[4] == 'x':
    if s[1] == '+':
      x = int(s[0]) + int(s[2])
    else:
      x = int(s[0]) - int(s[2])
  print('x=',x)

def zad1_7(i = 0):
  n = input("Ваш многочлен -->")
  n = input("Ваш многочлен -->")
  data = []
  while i< len(n):
    a = n[i]
    
    i+=3
  np.roots(data)

  
def zad1_8(a,b,c):
  return a, b, c
def zad1_9(a,b,c):
  return a, b, c

def zad4_1():
  num = int(input("Ваше число -->"))
  base = int(input("Base (2-9): "))
  if not(2 <= base <= 9):
    quit()
  newNum = ''
  while num > 0:
    newNum = str(num % base) + newNum
    num //= base
  print(newNum)

def zad4_2():
  fname = "text.txt"
  lines = 0
  words = 0
  letters = 0
 
  for line in open(fname):
    lines += 1
    letters += len(line)
 
    pos = 'out'
    for letter in line:
        if letter != ' ' and pos == 'out':
            words += 1
            pos = 'in'
        elif letter == ' ':
            pos = 'out'
 
  print("Строк:", lines)
  print("Слов:", words)
  print("Букв:", letters)
  
'''def ex1(R1,R2,x1,y1,x2,y2):
    size = sqrt((x1-x2)**2 + (y1-y2)**2)
    if not size:
        if R1 != R2:
            print('Нет')
        else: print('Да')
    else:
        if size > R1 + R2:
            print('Нет') 
        else: print('Да')

def ex51(i=0):
    s = input("Ваша строка")
    print (s)
    data = map(int, s.split())
    answer = []
    while i<len(data):
        if data[i]>10:
            answer.append(data[i])
        i+=1
    print (answer)

def ex71(s=0,i=2):
    while(i<1000):
        if ((i%3==0) or (i%5==0)):
            s+=i
        i+=1
    print (s)

def ex72(i=999,j=999,f=True,Vspm=True):
    print('started')
    data = []
    while i>0:
        while j>0:
            x = i*j
            if (int(x/100000)==(x%10)):
                if (int(x/10000)%10)==(int((x%100)/10)):
                    if (int((x%10000)/1000)==((int((x%1000)/100)))):
                        data.append(i*j)
            j-=1
        j=999
        i-=1
    print(max(data))
        

def ex73(f=False,i=998):
    print('started')
    while f == False:
      for c in range (i,3,-1):  
        for b in range (i-1,2,-1):
            for a in range (i-2,1,-1):
                if ((a+b+c==1000)and(a<b<c)):
                    print('mb',a,b,c)
                    if (a*a+b*b==c*c):
                        print('ответ a = ', a ,' b = ', b , ' c = ', c)
                        f = True
                        break

def asd():
    print('hello')

def pod4_1(num,base):
    if not(2 <= base <= 9):
        quit()

    newNum = ''

    while num > 0:
        newNum = str(num % base) + newNum
        num //= base

    print('преобразованное число:',newNum)

def pod4_2():
    try:
        fail = open('C:\\Users\\Student\\Desktop\\input123.txt', 'r')
    except IOError:
        print ("No file")
    lines = 0
    words = 0
    chars = 0
    data = fail.read()
    print(data)
    with open('input123.txt', 'r') as fail:
        for line in fail:
            current_words = line.split()
            lines += 1
            words += len(current_words)
            chars += len(line)
    print(lines,words,chars)
'''
def zad4_3():
    Workers.input()

class Workers:
  def __init__(man, name, family, father, sex, age, salary, position ):
    man.name = name
    man.family = family
    man.father = fathersname
    man.sex = sex
    man.age = age
    man.salary = salary
    man.position = position
    
  def input():
    try:
      input = open('C:\\Users\\Student\\Desktop\\inputworker.txt', 'r')
    except IOError:
      print ("No input file")
      workersdata=input.read()
      num_of_worker = 0
      data =[]
      workersdata.split()
      num_of_worker += 1
      data[num_of_worker]=Workers(current_words)
      print(current_words)
      print(num_of_worker)
      print(workersdata)
      print(data)
      output()
      
  def output():
    try:
      output_file = open('C:\\Users\\Student\\Desktop\\outputworker.txt', 'w')
    except IOError:
      print ("No input file")
      i=0
      for line in man:
        output_file.write(man[i])

def zad5_1(dzo = 54, mjd = 68, yd = 366,f = False):
    K = int(input('введите К -->'))
    data = [0,0,0,0,0,0,0]
    '''if K in range(0,51):'''
    if K in range(0,331):
        '''
        Високосный год, 23 февраля - 54 число, 8 марта - 68
        
        K = 330
        в 36 7 вых
        K = 300
        в 66 19  
        K = 200
        в 166 48
        K = 54
        в 312 90 + dzo = 91
        '''
        dzo -= K
        mjd -= K
        yd -= K
        wd = [1,2,3,4,5,6,7]
        i = 1
        while i <= yd :
            j = 0
            while j <= 6:
                if i in [dzo,mjd]:
                    if wd[j] in [6,7]:
                        f = True
                    else:
                        data[j] +=1
                if wd[j] in [6,7]:
                    data[j] +=1
                if f == True:
                    if wd[j] == 1:
                        data[j] += 1
                        f = False
                wd[j] += 1
                if wd[j] > 7:
                    wd[j] = 1
                j += 1
            i += 1
        print(max(data))
    else:
        print('error')

def zad5_2():
    return 1

def zad5_3():
    return 1

    
def reksum(data,sum=0):
  print(sum)
  if not data:
    print('answer = ', sum)
    raise Exception("ok")
  else:
    sum+=data[0]
    try:
      data.pop(0)
      reksum(data,sum)
    except:
      raise Exception('ok2')
      
def a1():
  data = []
  while True:
    try:
      a = int(input('-> '))
      data.append(a)
    except:
      break
  reksum(data)

def factorial(n):
    if n == 0:
        return 1
    else:
        return (n * factorial(n-1))

def a2():
  n = int(input('введите Ваше число -->'))
  print ('fctorial etogo 4isla =' , factorial(n))

def fibonachi(n):
    if n in (1, 2):
        return 1
    return fibonachi(n - 1) + fibonachi(n - 2)

def a3():
  n = int(input('введите длину последовательности фибоначи -->'))
  print('ответ = ', fibonachi(n))

def a4():
  n = input('Введите Ваше число')
  s = 0
  for i in n:
    if i in ['1','2','3','4','5','6','7','8','9']:
      s += int(i)
  print("Сумма цифр:", s)

def sumpolchet(s0,x,s,i):
  if (x - s * i) <= 0:
    print('ответ = ',s0)
    return 1
  else:
    s0 += x - s * i
    i += 1
    sumpolchet(s0,x,s,i)

def a5():
  x = int(input('Введите начальное число -->'))
  sumpolchet(x,x,2,1)


def a6(EPS = 0.1):
  n = 2
  s = 1 + 1/n
  prev = 1
  while (abs(s - prev) >= EPS):
    prev = s
    n = n+1
    s = s + 1/n
  print('summa = ',s)

def a7(EPS = 0.1):
  n = 2
  s = 1/n
  prev = 1
  while (abs(s - prev) >= EPS):
    prev = s
    n = n * 2
    s = s + 1/n
  print('summa = ',s)

def b1():
  s = input('Введите Вашt имя, возраст и адрес через пробел -->')
  x = s.split()
  try:
    x[1] = int(x[1])
  except:
    pass
  if x[1] is int:
    Name = x[0]
    Age = x[1]
    Address = x[2]
    print('Имя : ', Name)
    print('Возраст : ', Age)
    print('Адрес : ', Address)
  else:
    print('Возраст не число!')

def b2():
  x = int(input('Введите вклад -->'))
  n = float(input('Введите ставку -->'))
  y = int(input('Введите срок -->'))
  
  while (y>0):
    x = x / 100 * n + x
    y -= 1

  print('Итог = ', x)

def b3():
  x1 = float(input('Введите x1 -->'))
  y1 = float(input('Введите y1 -->'))
  x2 = float(input('Введите x2 -->'))
  y2 = float(input('Введите y2 -->'))
  s = pow((math.fabs(x1-x2))*(math.fabs(x1-x2))+(math.fabs(y1-y2))*(math.fabs(y1-y2)) , 1/2)
  print('S = ', s)

def b4():
  folder = input('Введите полный путь к файлу -->')
  if not os.path.exists(folder):
    print('нету')
  else:
    print (folder)

def b5():
  print(sys.platform)

def b6():
  print(os.listdir(path="."))

def b7():
  x = float(input('Введите x -->'))
  y = float(input('Введите y -->'))
  s = pow(x*x+y*y,1/2)
  print('S = ', s)

def b8():
  c = input('Введите символ -->')
  print('ASCII = ', ord(c))

def b9():
  print('Размер main.py = ', os.path.getsize('main.py'),'  bytes')


def b10():
  x = int(input('Введите x -->'))
  y = int(input('Введите y -->'))
  print(x, ' + ', y ,' = ', x+y)


def b11():
  x = int(input('Введите x -->'))
  k = int(input('Введите систему счисления -->'))
 
  ls = []
  while x > 0:
    x, a = divmod(x, k)
    ls = [a] + ls
  print(ls)

def b12():
  m = int(input('Введите месяц -->'))
  y = int(input('Введите год -->'))
  print(calendar.month(y,m))

def b13():
  d1 = int(input('Введите день 1-го числа -->'))
  m1 = int(input('Введите месяц 1-го числа -->'))
  y1 = int(input('Введите год 1-го числа -->'))
  d2 = int(input('Введите день 2-го числа -->'))
  m2 = int(input('Введите месяц 2-го числа -->'))
  y2 = int(input('Введите год 2-го числа -->'))
  date1 = date(y1, m1, d1)
  date2 = date(y2, m2, d2)
  delta = date1 - date2
  print ('Дней между датами: ',fabs(delta.days))

def c1():
  l = np.array([[1., 1., 2., -1.], [1., 1., -2., 1.], [1., -1., 1., 2.], [1., -1., -1., -2.]]) # Матрица (левая часть системы)
  p = np.array([9., 7., -9., 5.]) # Вектор (правая часть системы)
  print(np.linalg.solve(l, p))

def c2():
  A = np.matrix('1 1; -1 1')
  B = np.matrix('1 -1 1; 1 1 -1')
  C = np.matrix('10 2 -2; 4 0 0')
  print ('x = ')
  print ((C * (B**-1)) )

def functionc3(x):
  return (x*x*x - 6*x*x + 13*x - 10)

def c3(f = False, x = -1, Chet = True):
  while f == False:
    if functionc3(x) == 0:
      f = True
    else:
      if Chet == True:
        x *= -1
        Chet = False
      else:
        x += 1
        Chet = True
    if x > 1000:
      f = True
      print('Решение по модулю > 1000 или не целые корни')
  print ('один из корней = ', x)

def c4():
  A = np.poly1d([1,2,3,4,5])
  B = np.poly1d([6,7,8])
  print (A*B)

def c5():
  '''p = np.poly1d.integ(m=1, k=0) [-1,4,-3]
  print(p.r)'''
  func = lambda x: -(x*x-4*x+3)
  answer = integrate.quad(func, 1, 3)
  print(answer)
  return 1

def c6v(x):
  return((1+x*x)*math.exp(-x)+math.sin(x))

def c6v2(x,y):
  G = 0
  if c6v(x)*c6v((x+y)/2) < 0:
    y = (x+y)/2
  if c6v(y)*c6v((x+y)/2) < 0:
    x = (x+y)/2
  if abs(c6v((x+y)/2))<0.00000000001:
    print((x+y)/2)
    return (x+y)/2
  if abs(x-y)<0.00000000001: 
    print('NET')
    return 1
  c6v2(x,y)

def c6():
  a = int(input('Введите начало рассчета -->'))
  b = int(input('Введите конец рассчета -->'))
  while a <= b:
    c6v2(a,b)
    a=c6v2(a,b)

def d1():
  f = open('text.txt', 'w')
  A = np.array([random.random() for i in range(10)])
  f.write('Ishodnyi:')
  for index in A:
    index = int(index*10)
    f.write(str(index))
    f.write(' ')
  f.write('Tchetnye:')
  for index in A:
    index = int(index*10)
    if index % 2 == 0:
      f.write(str(index))
      f.write(' ')
  f.close()

def d2():
  input = open('text.txt', 'r')
  output = open('output.txt', 'w')
  string = input.read()
  S = 0
  for index in string:
    if index in ['1','2','3','4','5','6','7','8','9','0']:
      S+=int(index)
    else:
      if index == ' ':
        output.write(str(S))
        output.write(' ')
        S = 0 
  input.close()
  output.close() 

def d3():
  inp = open('text.txt', 'r')
  f = inp.read()
  s = input('Введите искомую подстроку -->')
  if s in f:
    print('Входит')
  else:
    print('Не входит')
  inp.close()

def d4():
  fname = "text.txt"
  lines = 0
  words = 0
  letters = 0
  for line in open(fname):
    lines += 1
    letters += len(line)
 
    pos = 'out'
    for letter in line:
        if letter != ' ' and pos == 'out':
            words += 1
            pos = 'in'
        elif letter == ' ':
            pos = 'out'
 
  print("Строк:", lines)
  print("Слов:", words)
  print("Букв:", letters)

def d5():
  inp = open('text.txt', 'r')
  f = inp.read()
  s = input('Введите искомую подстроку -->')
  sz = input('Введите требуемую замену подстроки -->')
  if s in f:
    while f.find(s) > 0:
      i = f.find(s)
      f = f[:i] + sz + f[i+len(s):]
  else:
    print('Не входит')
  inp.close()
  out = open('text.txt', 'w')
  out.write(f)

def d6():
  input = open('text.txt', 'r')
  output = open('output.txt', 'w')
  string = input.read()
  slen = len(string) - 1
  while slen >= 0:
    output.write(string[slen])
    slen += -1
  input.close()
  output.close()
  return 1

def d7():
  input = open('text.txt', 'r')
  output = open('output.txt', 'w')
  string = input.read()
  for index in string:
    if index in ['a','e','y','u','o','i']:
      pass
    else:
      output.write(index)
  input.close()
  output.close() 

def d8():
  path = input("Enter path to folder: ")
  listOfFiles = os.listdir(path)
  countOfFiles = len(listOfFiles)
  os.chdir(path)
  for i in range(0, countOfFiles):
    os.rename(path+listOfFiles[i], str(i+1)+'.tedited')

def d9():
  input = open('text.txt', 'r')
  output = open('output.txt', 'w')
  string = input.read()
  cash = []
  result = []
  i = 0
  for Z in string:
    for index in string:
      if index not in cash:
        cash.append(index)
      else:
        j = i
        f = True
        for element in cash:
          if not len(cash) >= ( len(string) - j ):
            if not string[j] == element:
              f = False
            j+=1
        if f == True:
          result.append(cash)
      i+=1
  output.write(str(max(result)))
  output.write('otvet   ')
  output.write(str(len(max(result)))) 
  input.close()
  output.close()

def d10():
  input = open('text.txt', 'r')
  output = open('output.txt', 'w')
  data = []
  data_s = []
  for line in input:
    data.append(line.replace('\n', ''))
  x = data[1]
  #'5*x^2+6*x+7'
  data[0] = data[0].replace('-', '+-').split('+')
  #['5*x^2', '6*x', '7']
  for i, char in enumerate(data[0]):
    for j, element in enumerate(data[0][i]):
      if data[0][i][j] == 'x':
        stch = 0
        P = 1
        while (stch<int(data[0][i][j+2])):
          P *= int(x)  
          stch += 1
        data_s.append(P*int(data[0][i][j-2]))
  #['5*25', '6*5', '7']

  '''for i, char in enumarate(data[0]):
    for j, element in data[0][i]:
      if data[0][i][j] == '*':
        '''
  '''
  
  #[['5', 'x^2'], ['6', 'x'], '7']


  #  for j in range(len(data[0][i])):
  #    data[0][i][j] = data[0][i][j].split('^')
  # data[0] = '5*x+4'
  # data[1] = x '''
  S = 0
  for element in data_s:
    S+=element
  print (S)
  output.write(str(S))
  input.close()
  output.close()

'''
11. В некотором государстве был принят закон о политических партиях. По этому закону один
человек не может состоять более чем в одной партии. Партии, члены которых состоят в других
партиях, не должны регистрироваться. Была объявлена перерегистрация партий. В
центризбирком были поданы списки членов партий. Списки подавались в закодированном виде.
Каждому гражданину было заранее присвоено некоторое кодовое положительное число.
Подобный код был присвоен и партии. Таким образом, в списках были лишь числа. Первым шло
число – код гражданина, вторым – код партии. Списки составлялись не аккуратно, так что числа
располагались произвольно, например, так
2 20 2010 5
10
1 67
3
Впрочем, строго соблюдалось очевидное правило: количество чисел в списках было четным.
Задание
Выявить партии, не подлежащие регистрации.

Форматы данных
Примем, что количество совершеннолетних граждан не превышает 30000, а количество партий по
законодательству ограничено 100.
Исходный файл input содержит пары подряд идущих чисел, первое из которых - код человека,
второе - код партии. Числа друг от друга отделяются произвольным количеством пробелов и
переводов строк.
Результирующий файл output содержит список партий, подлежащих регистрации.
Примеры
Исходный файл
1
1 2 2
3 3 4 4
5
5 1 4
Выходной файл
2 3 5
'''

def d11():
  input = open('text.txt', 'r')
  output = open('output.txt', 'w')
  
  data_f = []
  data  =[]
  string = input
  f = False
  num_cash = ''

  for line in input:
    for char in line:
      data_f.append(char)
  for i, char in enumerate(data_f):
    if data_f[i] in ['1','2','3','4','5','6','7','8','9','0']:
      f = True
      num_cash += data_f[i]
    else:
      if f == True:
        data.append(int(num_cash))
        num_cash = ''
        f = False
  data.append(int(num_cash))
  num_cash = ''
  f = False

  parties_data = []
  citizens_data = []
  i = 0

  for num in data:
    if f == False:
      citizens_data.append(num)
      f = True
      i += 1
    else:
      parties_data.append(num)
      f = False
      i += 1
  j = 0
  check_data = []
  wrong_citizens = []
  wrong_parties = []
  while (j < int(i/2)):
    if citizens_data[j] in check_data:
      wrong_citizens.append(citizens_data[j])
    else:
      check_data.append(citizens_data[j])
    j += 1

  j = 0 
  while (j < int(i/2)):
    if citizens_data[j] in wrong_citizens:
      wrong_parties.append(parties_data[j])
    j += 1

  output.write('Checkpoint')
  output.write('\n')

  for party in parties_data:
    if int(party)>100:
      output.write('Unregistered party - ')
      output.write(str(party))
      output.write('\n')


  for citizen in citizens_data:
    if int(citizen)>30000:
      output.write('Unregistered citizen - ')
      output.write(str(citizen))
      output.write('\n')

  output.write('\n')
  output.write('wrong_parties - ')
  output.write('\n')
  output.write(str(wrong_parties))
  output.write('\n')
  output.write('ok_parties - ')
  output.write('\n')
  f = True
  for party in parties_data:
    if party not in wrong_parties:
      if int(party)<=100:
        if f == False:
          output.write(' , ')
        output.write(str(party))
        f = False

  input.close()
  output.close()

'''
12. Дан файл, состоящий из строк. Будем рассматривать только строчки, состоящие из
заглавных латинских букв. Например, рассмотрим строку AAAABCCCCCDDDD. Длина этой строки
равна 14. Поскольку строка состоит только из латинских букв, повторяющиеся символы могут
быть удалены и заменены числами, определяющими количество повторений. Таким образом,
данная строка может быть представлена как 4AB5C4D. Длина такой строки 7. Напишите
программу, которая берет упакованную строчку и восстанавливает по ней исходную строку.
Например
Исходный файл
3A4B7D
22D7AC18FGD
95AB
40AB39A
Выходной файл
AAABBBBDDDDDDD
DDDDDDDDDDDDDDDDDDDDDDAAAAAAACFFFFFFFFFFFFFFFFFFGD
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAAAAAAA
'''

def d12():
  input = open('text.txt', 'r')
  output = open('output.txt', 'w')
  
  data_f = []
  num_cash = 0
  RESULT = []

  for line in input:
    for char in line:
      data_f.append(char)
  for char in data_f:
    if char in ['1','2','3','4','5','6','7','8','9','0']:
      num_cash = int(char)
    else:
      i = 0
      while i < num_cash:
        RESULT.append(char)
        i+=1
  for element in RESULT:
    output.write(element)
  input.close()
  output.close()


def six():
    from numpy import exp, sin

    def f(x):
        return (1 + x**2)*exp(-x) + sin(x)
    #    a  b x=(a+b)/2  eps
    l = [0, 10, 5, 0.000000001]
    while abs(l[0]-l[1]) >= l[3]:
        l[2] = (l[0]+l[1])/2
        if f(l[0])*f(l[2]) < 0:
            l[1] = l[2]
        elif f(l[1])*f(l[2]) < 0:
            l[0] = l[2]
        else:
            print('Error', l)
    print(l[2], f(l[2]))

def get_a_lenta(max):
  lenta = np.zeros((max), dtype=bool)
  return (lenta)

def mp1():
  x = int(input('Input 1 num --> '))
  y = int(input('Input 2 num --> '))

  lenta_s = '-'
  for i in range(x):
    lenta_s += '^'
  lenta_s += '_'
  for i in range(y):
    lenta_s += '^'
  lenta_s += '/'
  print(lenta_s)

  lenta = list(lenta_s)

  F = False
  sec_num = True
  begunok = 1
  stop = False
  i = 0
  check = x + y + 3

  while F == False:
    #print('checkpoint [')
    #print(i)
    #print(']')
    try:
      if lenta[i] == '-':
        begunok = 1
        print('\ncheck - ')
    except:
      i += 1
    try:
      if lenta[i] == '/':
        begunok = -1
        print('\ncheck / ')
    except:
      i += -1
    try:
      if lenta[i] == '_':
        print('\ncheck _ ')
        if lenta[i-1] == '^' and lenta[i+1] == '^':
          lenta[i-1] = '_'
          while i <= (check - 1):
            lenta[i] = '^'
            i += 1
          i += -1
          lenta[i-2] = '/'
          lenta[i-1] = ''
          lenta[i] = ''
          check += -2
          i += -1
          print('\ncheck _ done ')
          #out
          print('\n')
          lenta_s = ''
          for k in range(check+1):
            lenta_s += lenta[k]
          print(lenta_s)
        else:
          F = True
          print('rass4et okon4en')
    except:
      i += 1
    try:
      if lenta[i] == '/':
        i += -1
    except:
      begunok = -1
      i += -1
    i += begunok
    #print('\ncheck shag ')
    #print(i)


def mt1():
  max = 100
#  lenta = get_a_lenta(max)

  x = int(input('Input 1 num --> '))
  y = int(input('Input 2 num --> '))

  lenta_s = '/'
  for i in range(79):
    if i != x - 1 and i != y - 1:
      lenta_s += '-'
    else:
      lenta_s += '^'
  lenta = list(lenta_s)
  #out
  lenta_s = '/'
  for i in range(79):
    lenta_s += lenta[i]
  print(lenta_s)

  i = y - 1
  
  F = False
  sec_num = True
  begunok = 1
  stop = False
  while F == False:
    if lenta[i] == '^':
      if sec_num == True:
        lenta[i] = '-'
        lenta[i-1] = '^'
        sec_num = False
        begunok = -1

        if stop == True:
          F = True

        #out
        lenta_s = '/'
        for k in range(79):
          lenta_s += lenta[k]
        print(lenta_s)
        print('done 1')

      else:
        if lenta[i-1] != '/':
          lenta[i] = '-'
          lenta[i-1] = '^'
          begunok = 1
          sec_num = True

          #out
          lenta_s = '/'
          for k in range(79):
            lenta_s += lenta[k]
          print(lenta_s)
          print('checkpoint')

        else:
          stop = True
          lenta[i] = '-'
          begunok = 1
          sec_num = True

          #out
          lenta_s = '/'
          for k in range(79):
            lenta_s += lenta[k]
          print(lenta_s)

        print('done 2')

    i+=begunok

  lenta_s = '/'
  for i in range(79):
    lenta_s += lenta[i]
  print(lenta_s)
  



  # x - y
'''
  i = 0
  lenta[x] = True
  lenta[y] = True
  f = False
  check = False
  while f == False:
    check = False
    while check == False:
      if lenta[i] == True:
        try:
          lenta[i] = False
          lenta[i-1] = True
          check = True
        except:
          f = True
          print('1 num becomes minusovoy')
      else:
        i += 1
    i -= 2
    check = False
    while check == False:
      if lenta[i] == True:
        try:
          lenta[i] = False
          lenta[i-1] = True
          check = True
        except:
          f = True
          print('1 num becomes minusovoy')
      else:
        i -= 1
'''



def mp2():
  x = int(input('Input num --> '))

  lenta_s = '-'
  for i in range(x):
    lenta_s += '^'
  lenta_s += '/'
  print(lenta_s)

  lenta = list(lenta_s)

  s4et4ik = 0
  stop = False
  i = 0
  len = x + 1

  while stop == False:
    if lenta[i] == '/':
      lenta[i] = '^'
      lenta.append('/')
      len += 1
      s4et4ik += 1
      print('\n')
      lenta_s = ''
      for k in range(len+1):
        lenta_s += lenta[k]
      print(lenta_s)
      if s4et4ik == 2:
        stop = True
        print('rass4et okon4en')
    i += 1

    #print('\ncheck shag ')
    #print(i)

def mp3():
  x = int(input('Input num --> '))

  lenta_s = '-'
  for i in range(x):
    lenta_s += '^'
  lenta_s += '/'
  print(lenta_s)

  lenta = list(lenta_s)

  s4et4ik = 0
  stop = False
  i = 0
  len = x + 1

  while stop == False:
    if lenta[i] == '/':
      lenta[i-1] = '/'
      lenta[i] = ''
      len += -1
      s4et4ik += 1
      print('\n')
      lenta_s = ''
      for k in range(len+1):
        lenta_s += lenta[k]
      print(lenta_s)
      if s4et4ik == 2:
        stop = True
        print('rass4et okon4en')
    i += -1

def mp4():
  x = int(input('Input num --> '))

  lenta_s = '-'
  for i in range(x):
    lenta_s += '^'
  lenta_s += '/'
  print(lenta_s)

  lenta = list(lenta_s)

  s4et4ik = 0
  stop = False
  i = 0
  cache = x
  len = x + 1

  while stop == False:
    if lenta[i] == '/':
      k = 1
      lenta[i] = '^'
      while k<= cache-1:
        lenta.append('^')
        k+=1
      lenta.append('/')
      len += cache
      cache = cache * 2
      s4et4ik += 1
      print('\n')
      lenta_s = ''
      for k in range(len+1):
        lenta_s += lenta[k]
      print(lenta_s)
      if s4et4ik == 1:
        stop = True
        print('rass4et okon4en')
    i += 1


def put_a_label():  # Добавить метку
      tape[current_position] = 1

def erase_a_label():  # Удалить метку
      tape[current_position] = 0

def move_left():  # Сдвинуться влево
      global current_position

      current_position -= 1

def move_right():  # Сдвинуться вправо
      global current_position

      current_position += 1

def is_there_a_label():  # Есть ли метка
      if tape[current_position]:
          return True
      else:
          return False

def end():  # Конец
      global in_work

      in_work = 0




def init_task(task_num):  # Выбор задания
    global lines, tape, current_position

    try:
        lines = tasks[task_num]
        tape = task_tape[task_num]
        current_position = task_position[task_num]
        return True
    except IndexError:
        return False



def POST():
  global cycle, last_lines, in_work, current_position, current_line, lines, tape


  commands = {
              '<': move_left,
              '>': move_right,
              'V': put_a_label,
              'X': erase_a_label,
              '?': is_there_a_label,
              '!': end
              }

  try:
      current_task = int(input('Enter task number(1-6): '))
  except ValueError:
      current_task = 0

  if not init_task(current_task):
      in_work = 0

  print('\nTape before:', tape)
  print('Task:', lines, '\n')

  while in_work:
    if len(last_lines) == 2:
        last_lines.pop(0)
        last_lines.append(current_line)
    else:
        last_lines.append(current_line)
    if current_line in last_lines:
        cycle += 1
    if cycle >= cycle_constraint:
        mb.showerror("Ошибка", "Что то пошло не так")
        break
    # Выполнение задачи
    line = lines[current_line].split(' ')
    while current_position >= len(tape):
        tape.append(0)
    result = commands[line[0]]()
    if result is not None:
        if_false, if_true = line[1].split(';')
        if result:
            current_line = int(if_true)
        else:
            current_line = int(if_false)
    else:
        if line[0] != '!':
            current_line = int(line[1])


  if not in_work:
      print('Tape after:', tape)


def build_a_graphic(f,x0,x1):
  root = Tk()

  canv = Canvas(root, width = 1000, height = 1000, bg = "white")
  canv.create_line(500,1000,500,0,width=2,arrow=LAST) 
  canv.create_line(0,500,1000,500,width=2,arrow=LAST) 

  First_x = -500;

  for i in range(16000):
    if (i % 800 == 0):
      k = First_x + (1 / 16) * i
      k_s = x0 + x1 * (i * (1 / 16000))
      canv.create_line(k + 500, -3 + 500, k + 500, 3 + 500, width = 0.5, fill = 'black')
      canv.create_text(k + 515, -10 + 500, text = str(k_s), fill="purple", font=("Helvectica", "10"))
      if (k != 0):
        canv.create_line(-3 + 500, k + 500, 3 + 500, k + 500, width = 0.5, fill = 'black')
        canv.create_text(20 + 500, k + 500, text = str(k_s), fill="purple", font=("Helvectica", "10"))
    try:
      canv.create_oval(x, y, x + 1, y + 1, fill = 'black')
    except:
      pass
  canv.pack() 
  root.mainloop()

def graphinit(cmd):
  step = 100
  tasks = [one, two, three, four, five]
  try:
      tasks[int(cmd)-1](step)
  except IndexError:
    pass

def one(step):
    def f(x):
        return abs(x + 2) + abs(x + 3) + abs(x - 2) + abs(x - 4)

    x = np.linspace(-5., 5., step)
    y = np.array(f(x))

    fig = mpl.figure()
    mpl.plot(x, y)

    mpl.show()

def two(step):
    def f(x):
        return np.sin(x)/x


    x = np.linspace(-10., 10., step)
    y = np.array(f(x))

    fig = mpl.figure()
    mpl.plot(x, y)

    mpl.show()

def three(step):
    def f1(x):
        return np.sin(x)/x


    def f2(x):
        return 1 - x**2/6


    def f3(x):
        return 1 - x**2/6 + x**4/120


    x = np.linspace(-10., 10., step)
    y1 = np.array(f1(x))
    y2 = np.array(f2(x))
    y3 = np.array(f3(x))

    fig = mpl.figure()
    mpl.plot(x, y1, '-', x, y2, '--', x, y3, ':')

    mpl.show()

def four(step):
    def f(a, b):
        return np.sqrt(a**2 + b**2)

    def g(a):
        return np.sin(a)/a

    fig = mpl.figure()
    ax = Axes3D(fig)

    x = np.linspace(-10., 10., 100)
    y = np.linspace(-10., 10., 100)
    x, y = np.meshgrid(x, y)
    r = np.array(f(x, y))
    z = np.array(g(r))

    ax.plot_wireframe(x, y, z)
    mpl.show()

def five(step):
    def f(a, b):
        return np.sqrt(1 - a**2 - b**2)

    fig = mpl.figure()
    ax = Axes3D(fig)

    x = np.linspace(-1., 1., 100)
    y = np.linspace(-1., 1., 100)
    x, y = np.meshgrid(x, y)
    z = np.array(f(x, y))

    ax.plot_wireframe(x, y, z)
    mpl.show()

'''
def ping_pong():
  # глобальные переменные
  # настройки окна
  WIDTH = 900
  HEIGHT = 300
   
  # настройки ракеток
   
  # ширина ракетки
  PAD_W = 10
  # высота ракетки
  PAD_H = 100
   
  # настройки мяча
   
  # радиус мяча
  BALL_RADIUS = 30
   
  # устанавливаем окно
  root = Tk()
  root.title("PythonicWay Pong")
   
  # область анимации
  c = Canvas(root, width=WIDTH, height=HEIGHT, background="#003300")
  c.pack()
   
  # элементы игрового поля
   
  # левая линия
  c.create_line(PAD_W, 0, PAD_W, HEIGHT, fill="white")
  # правая линия
  c.create_line(WIDTH-PAD_W, 0, WIDTH-PAD_W, HEIGHT, fill="white")
  # центральная линия
  c.create_line(WIDTH/2, 0, WIDTH/2, HEIGHT, fill="white")
   
  # установка игровых объектов
   
  # создаем мяч
  BALL = c.create_oval(WIDTH/2-BALL_RADIUS/2,
                       HEIGHT/2-BALL_RADIUS/2,
                       WIDTH/2+BALL_RADIUS/2,
                       HEIGHT/2+BALL_RADIUS/2, fill="white")
   
  # левая ракетка
  LEFT_PAD = c.create_line(PAD_W/2, 0, PAD_W/2, PAD_H, width=PAD_W, fill="yellow")
   
  # правая ракетка
  RIGHT_PAD = c.create_line(WIDTH-PAD_W/2, 0, WIDTH-PAD_W/2, 
                            PAD_H, width=PAD_W, fill="yellow")
   

  # добавим глобальные переменные для скорости движения мяча
  # по горизонтали
  BALL_X_CHANGE = 20
  # по вертикали
  BALL_Y_CHANGE = 0
   
  def move_ball():
      c.move(BALL, BALL_X_CHANGE, BALL_Y_CHANGE)
   
  def main():
      move_ball()
      # вызываем саму себя каждые 30 миллисекунд
      root.after(30, main)
   
  # запускаем движение
  main()  


  # зададим глобальные переменные скорости движения ракеток
  # скорось с которой будут ездить ракетки
  PAD_SPEED = 20
  # скорость левой платформы
  LEFT_PAD_SPEED = 0
  # скорость правой ракетки
  RIGHT_PAD_SPEED = 0
   
  # функция движения обеих ракеток
  def move_pads():
      # для удобства создадим словарь, где ракетке соответствует ее скорость
      PADS = {LEFT_PAD: LEFT_PAD_SPEED, 
              RIGHT_PAD: RIGHT_PAD_SPEED}
      # перебираем ракетки
      for pad in PADS:
          # двигаем ракетку с заданной скоростью
          c.move(pad, 0, PADS[pad])
          # если ракетка вылезает за игровое поле возвращаем ее на место
          if c.coords(pad)[1] < 0:
              c.move(pad, 0, -c.coords(pad)[1])
          elif c.coords(pad)[3] > HEIGHT:
              c.move(pad, 0, HEIGHT - c.coords(pad)[3])
   
  # Вставляем созданную функцию в main
  def main():
       move_ball()
       move_pads()
       root.after(30, main)
   
  # Установим фокус на Canvas чтобы он реагировал на нажатия клавиш
  c.focus_set()
   
  # Напишем функцию обработки нажатия клавиш
  def movement_handler(event):
      global LEFT_PAD_SPEED, RIGHT_PAD_SPEED
      if event.keysym == "w":
          LEFT_PAD_SPEED = -PAD_SPEED
      elif event.keysym == "s":
          LEFT_PAD_SPEED = PAD_SPEED
      elif event.keysym == "Up":
          RIGHT_PAD_SPEED = -PAD_SPEED
      elif event.keysym == "Down":
          RIGHT_PAD_SPEED = PAD_SPEED
   
  # Привяжем к Canvas эту функцию
  c.bind("<KeyPress>", movement_handler)
   
  # Создадим функцию реагирования на отпускание клавиши
  def stop_pad(event):
    global LEFT_PAD_SPEED, RIGHT_PAD_SPEED
    if event.keysym in "ws":
      LEFT_PAD_SPEED = 0
    elif event.keysym in ("Up", "Down"):
      RIGHT_PAD_SPEED = 0
    
  # Привяжем к Canvas эту функцию
  c.bind("<KeyRelease>", stop_pad)
  # запускаем работу окна
  # импортируем библиотеку random
  import random
   
  # Добавляем глобальные переменные
  # Насколько будет увеличиваться скорость мяча с каждым ударом
  BALL_SPEED_UP = 1.05
  # Максимальная скорость мяча
  BALL_MAX_SPEED = 40
  # Начальная скорость по горизонтали
  BALL_X_SPEED = 20
  # Начальная скорость по вертикали
  BALL_Y_SPEED = 20
  # Добавим глобальную переменную отвечающую за расстояние
  # до правого края игрового поля
  right_line_distance = WIDTH - PAD_W
   
  # функция отскока мяча
  def bounce(action):
    global BALL_X_SPEED, BALL_Y_SPEED
    # ударили ракеткой
    if action == "strike":
      BALL_Y_SPEED = random.randrange(-10, 10)
      if abs(BALL_X_SPEED) < BALL_MAX_SPEED:
        BALL_X_SPEED *= -BALL_SPEED_UP
      else:
        BALL_X_SPEED = -BALL_X_SPEED
    else:
      BALL_Y_SPEED = -BALL_Y_SPEED
    
  # Переписываем функцию движения мяча с учетом наших изменений
  def move_ball():
    # определяем координаты сторон мяча и его центра
    ball_left, ball_top, ball_right, ball_bot = c.coords(BALL)
    ball_center = (ball_top + ball_bot) / 2
  
    # вертикальный отскок
    # Если мы далеко от вертикальных линий - просто двигаем мяч
    if ball_right + BALL_X_SPEED < right_line_distance and \
        ball_left + BALL_X_SPEED > PAD_W:
      c.move(BALL, BALL_X_SPEED, BALL_Y_SPEED)
    # Если мяч касается своей правой или левой стороной границы поля
    elif ball_right == right_line_distance or ball_left == PAD_W:
      # Проверяем правой или левой стороны мы касаемся
      if ball_right > WIDTH / 2:
        # Если правой, то сравниваем позицию центра мяча
        # с позицией правой ракетки.
        # И если мяч в пределах ракетки делаем отскок
          if (c.coords(RIGHT_PAD)[1] < ball_center < c.coords(RIGHT_PAD)[3]):
            bounce("strike")
          else:
            update_score("left")
            spawn_ball()
        else:
          if c.coords(LEFT_PAD)[1] < ball_center < c.coords(LEFT_PAD)[3]:
            bounce("strike")
          else:
            update_score("right")
            spawn_ball()
      # Проверка ситуации, в которой мячик может вылететь за границы игрового поля.
      # В таком случае просто двигаем его к границе поля.
      else:
          if ball_right > WIDTH / 2:
              c.move(BALL, right_line_distance-ball_right, BALL_Y_SPEED)
          else:
              c.move(BALL, -ball_left+PAD_W, BALL_Y_SPEED)
      # горизонтальный отскок
      if ball_top + BALL_Y_SPEED < 0 or ball_bot + BALL_Y_SPEED > HEIGHT:
          bounce("ricochet")

  PLAYER_1_SCORE = 0
  PLAYER_2_SCORE = 0

  p_1_text = c.create_text(WIDTH-WIDTH/6, PAD_H/4,
                           text=PLAYER_1_SCORE,
                           font="Arial 20",
                           fill="white")
   
  p_2_text = c.create_text(WIDTH/6, PAD_H/4,
                            text=PLAYER_2_SCORE,
                            font="Arial 20",
                            fill="white")
  # Добавьте глобальную переменную INITIAL_SPEED
  INITIAL_SPEED = 20
   
  def update_score(player):
      global PLAYER_1_SCORE, PLAYER_2_SCORE
      if player == "right":
          PLAYER_1_SCORE += 1
          c.itemconfig(p_1_text, text=PLAYER_1_SCORE)
      else:
          PLAYER_2_SCORE += 1
          c.itemconfig(p_2_text, text=PLAYER_2_SCORE)
   
  def spawn_ball():
      global BALL_X_SPEED
      # Выставляем мяч по центру
      c.coords(BALL, WIDTH/2-BALL_RADIUS/2,
               HEIGHT/2-BALL_RADIUS/2,
               WIDTH/2+BALL_RADIUS/2,
               HEIGHT/2+BALL_RADIUS/2)
      # Задаем мячу направление в сторону проигравшего игрока,
      # но снижаем скорость до изначальной
      BALL_X_SPEED = -(BALL_X_SPEED * -INITIAL_SPEED) / abs(BALL_X_SPEED)
  root.mainloop()
'''


tasks = [['!'],  # 0 Если будет предвиденная ошибка, тогда будет запускаться эта задача
             ['< 1', '? 0;2', 'X 3', '> 4', '? 3;5', 'X 6', '> 7', '? 8;0', '!'],  # 1 Вычитание 2 чисел
             ['> 1', '? 2;0', 'V 3', '> 4', 'V 5', '!'],  # 2 Уменьшение числа на 2
             ['> 1', '? 2;0', '< 3', 'X 4', '< 5', 'X 6', '!'],  # 3 Увеличение числа на 2
             ['X 1', '> 2', '? 3;1', '> 4', '? 5;3', 'V 6', '> 7', 'V 8', '< 9', '? 10;8', '< 11', '? 12;13', '!', '< 14',
              '? 15;13', '> 0'],  # 4 Умножение числа на 2
             ['? 7;1', '> 2', '? 7;3', '> 4', '? 7;5', '> 6', '? 8;0', '!', '> 9', 'V 7'],  # 5 Делится ли число на 3
             ['X 1', '> 2', '? 3;1', 'V 4', '> 5', '? 9;6', '< 7', '? 8;6', '> 0', '!']  # 6 Сжатие
             ]

task_tape = [
              [0],  # 0
              [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],  # 1
              [0, 1, 1],  # 2
              [0, 1, 1, 1, 1],  # 3
              [1, 1],  # 4
              [1, 1, 1, 1, 1, 1],  # 5
              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 6
              ]

task_position = [0, 6, 0, 0, 0, 0, 0]

cycle = 0
cycle_constraint = 500
last_lines = []
in_work = 1
current_position = 0
current_line = 0
lines = []
tape = []




class Student:
  u'Студент'
  def __init__(self,name, family, father, date_of_birth, adress, phone_number, faculty, graduate, group):
    self.name = name
    self.family = family
    self.father = father
    self.date_of_birth = date_of_birth
    self.adress = adress
    self.phone_number = phone_number
    self.faculty = faculty 
    self.graduate = graduate
    self.group = group
  def __str__(self):
    return ('[Student: %s %s]'%(self.name,self.group))
  def select_group(self, group):
    return self.group == group


def From_one_faculty(faculty):
  for student in Students_list:
    try:
      if student.faculty == faculty:
        print(student)
    except AttributeError:
      print(student, 'error')
  
def Facultets_and_courses():
  print('Список по факультетам')
  facultets = []
  courses = []
  for student in Students_list:
    if student.faculty not in facultets:
      facultets.append(student.faculty)
    if student.graduate not in courses:
      courses.append(student.graduate)
  for faculty in facultets:
    print('Институт: ', faculty)
    for student in Students_list:
      if student.faculty == faculty:
        print(student)
      else:
        pass
    print('\n')

  for graduate in courses:
    print(graduate, ' курс:')
    for student in Students_list:
      if student.graduate == graduate:
        print(student)
      else:
        pass
    print('\n')

def Students_After(Year):
  for student in Students_list:
    if student.date_of_birth.split('.')[2] > Year:
      print(student)

def Students_from_one_group(group_in):
  group = group_in
  
  print(group)
  for student in Students_list:
    try:
      print(student.group, group)
      f = True
      for i, char in enumerate(group):
        try:
          if char != student.group[i]:
            f = False
        except:
          pass
      if f == True:
        print(student)
    except AttributeError:
      print(student, 'error')
  
  '''
  for student in Students_list:
    if student.select_group(group):
      print(student)
  '''
  '''
  for student in Students_list:
    print(group,' равно ли ', student.group)
    try:
      if student.group == group:
        print(student)
    except AttributeError:
      print(student, 'error')
  '''

class Point:

  def __init__(self, x, y, z):
      self.x = x
      self.y = y
      self.z = z

  def get_pos(self,time):
      for second in range(time):
        self.speed = self.speed + (second * self.acceleration)
        self.x = self.x + (self.vector_x*self.speed)
        self.y = self.y + (self.vector_y*self.speed)
        self.z = self.z + (self.vector_z*self.speed)
      return [self.x, self.y, self.z]

  def set_speed(self, speed):
      self.speed = speed

  def set_acceleration(self, acceleration):
      self.acceleration = acceleration

  def vector_init(self, x, y, z):
      self.vector_x = x
      self.vector_y = y
      self.vector_z = z



def intersect_or_no(My_first_point,My_second_point):
  try:
      x1 = My_second_point.x
      xo = My_first_point.x
      yo = My_first_point.y
      y1 = My_second_point.y
      zo = My_first_point.z
      z1 = My_second_point.z
      p = My_first_point.vector_x
      q = My_first_point.vector_y
      r = My_first_point.vector_z
      p1 = My_second_point.vector_x
      q1 = My_second_point.vector_y
      r1 = My_second_point.vector_z
      coord = []
      x = (xo*q*p1-x1*q1*p-yo*p*p1+y1*p*p1)/(q*p1-q1*p)
      y=(yo*p*q1-y1*p1*q-xo*q*q1+x1*q*q1)/(p*q1-p1*q)
      z=(zo*q*r1-z1*q1*r-yo*r*r1+y1*r*r1)/(q*r1-q1*r)
      coord.append(x,y,z)
      return (coord)
  except:
    coord = 'n/a, n/a, n/a'
    return (coord)

def R_between(My_first_point,My_second_point,time):
  return ((My_first_point.get_pos(time)[0] - My_second_point.get_pos(time)[0])**2 + (My_first_point.get_pos(time)[1] - My_second_point.get_pos(time)[1])**2 + (My_first_point.get_pos(time)[2] - My_second_point.get_pos(time)[2])**2 )**(1/2)

def Point_adventure():

  print('Введите начальное положение 1-ой точки')
  My_first_point = Point(int(input('по Х -->')),int(input('по У -->')),int(input('по Z -->')))
  print('Введите вектор направления 1-ой точки')
  My_first_point.vector_init(int(input('по Х -->')), int(input('по У -->')), int(input('по Z -->')))
  print('Введите начальную скорость 1-ой точки')
  My_first_point.set_speed(int(input('В ед. в секунду')))
  print('Введите ускорение 1-ой точки')
  My_first_point.set_acceleration(int(input('В ед. в секунду')))

  print('Введите начальное положение 2-ой точки')
  My_second_point = Point(int(input('по Х -->')),int(input('по У -->')),int(input('по Z -->')))
  print('Введите вектор направления 2-ой точки')
  My_second_point.vector_init(int(input('по Х -->')), int(input('по У -->')), int(input('по Z -->')))
  print('Введите начальную скорость 2-ой точки')
  My_second_point.set_speed(int(input('В ед. в секунду')))
  print('Введите ускорение 2-ой точки')
  My_second_point.set_acceleration(int(input('В ед. в секунду')))

  time = int(input('Введите время (сек.)-->'))
  print('Позиция 1-ой точки в ',My_first_point.get_pos(time))
  print('Позиция 2-ой точки в ',My_second_point.get_pos(time))
  print('Расстояние на данный момент = ', R_between(My_first_point,My_second_point,time))
  print('Точка пересечения --> ', intersect_or_no(My_first_point,My_second_point))







Students_list = []

Student1 = Student('Антон','Саввин','Васильевич','01.03.1998','Общежитие 66','89141000000','ИМИ','2','ПM17')
Students_list.append(Student1)
Student2 = Student('Сахайаана','Винокурова','Леонидовна','01.03.1996','Квартао','89141111111','ИМИ','2','ПM17')
Students_list.append(Student2)
Student3 = Student('Прокопий','Хоютанов','Константинович','01.03.1996','16 квартал','89142222222','ФТИ','4','ГС15')
Students_list.append(Student3)
Student4 = Student('Леонид','Имеев','Владимирович','01.03.1997','Аэропорт','89143333333','ИМИ','2','ПM17')
Students_list.append(Student4)
Student5 = Student('Ньургун','Халыев','Леонидович','01.03.2000','Ваня дьиэтэ','89144444444','ИФКиС','6','ОФ-13')
Students_list.append(Student5)
Student6 = Student('Егор','Красильников','Харлампьевич','01.03.1999','Борисов аттынан','89145555555','ГИ','4','ГР-15')
Students_list.append(Student6)
Student7 = Student('Егор','Мигалкин','Леонидович','01.03.1996','Bakery st, Toronto','103233323000','IoS','1','PF1')
Students_list.append(Student7)
Student8 = Student('Рустам','Готовцев','Егорович','01.03.1999','[UNKNOWN]','89146666666','ГИ','2','ГР17')
Students_list.append(Student8)
Student9 = Student('Станислав','Алексеев','Капитонович','01.03.1999','Ваня дьиэтэ','89147777777','ИМИ','2','ПM17')
Students_list.append(Student9)



def Class_lab():
  print('------------------------------------------------------------------------')
  print('ЗАДАНИЕ 1.А')
  Faculty = input('Введите требуемый факультет -->')
  From_one_faculty(Faculty)
  print('------------------------------------------------------------------------')
  print('ЗАДАНИЕ 1.Б')
  print('Список студентов по факультетам и курсам')
  Facultets_and_courses()
  print('------------------------------------------------------------------------')
  print('ЗАДАНИЕ 1.В')
  Year = input('Введите год -->')
  print('Список студентов родившихся после заданного года:')
  Students_After(Year)

  print('ЗАДАНИЕ 1.Г')
  group = input('Введите группу -->')
  print('Список студентов этой группы:')
  Students_from_one_group(group)


class Quadrangle_point:

        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __add__(self, other):
            return Quadrangle_point(self.x + other.x, self.y + other.y)

        def __sub__(self, other):
            return Quadrangle_point(self.x - other.x, self.y - other.y)

        def __mul__(self, other):
            from numpy import sqrt
            other = self - other
            return sqrt(other.x**2 + other.y**2)

class Quadrangle():

        def __init__(self, a, b, c, d):
            from numpy import sin, cos, deg2rad, sqrt, arccos, degrees
            self.a = a
            self.b = b
            self.c = c
            self.d = d

            self.ab = self.a*self.b
            self.bc = self.b*self.c
            self.cd = self.c*self.d
            self.da = self.d*self.a

            self.d1 = self.a*self.c
            self.d2 = self.b*self.d

            self.a_angle = round(float(degrees(arccos((self.ab**2 + self.da**2 - self.d2**2)/(2*self.ab*self.da)))), 3)
            self.b_angle = round(float(degrees(arccos((self.ab**2 + self.bc**2 - self.d1**2)/(2*self.ab*self.bc)))), 3)
            self.c_angle = round(float(degrees(arccos((self.bc**2 + self.cd**2 - self.d2**2)/(2*self.bc*self.cd)))), 3)
            self.d_angle = round(float(degrees(arccos((self.cd**2 + self.da**2 - self.d1**2)/(2*self.cd*self.da)))), 3)

            if round(self.a_angle + self.b_angle + self.c_angle + self.d_angle) != 360:
                print('НЕ ЧЕТЫРЕХУГОЛЬНИК')
                del self

        def set_a_type(self):
                if self.a_angle == self.b_angle == self.c_angle == self.d_angle and\
                        self.ab == self.bc == self.cd == self.da:
                    self.type = 'Квадрат'
                    return (Quad(self.a, self.b, self.c, self.d))
                elif self.a_angle == self.c_angle != self.b_angle == self.d_angle:
                    if self.ab == self.cd != self.bc == self.da:
                        self.type = 'Прямоугольник'
                        return (Rectangle(self.a, self.b, self.c, self.d))
                    elif self.ab == self.bc == self.cd == self.da:
                        self.type = 'Ромб'
                        return (Diamond(self.a, self.b, self.c, self.d))
                    else:
                        self.type = 'Произвольный'
                        return (Arbitrary_Quadrangle(self.a, self.b, self.c, self.d))

                elif self.a_angle <= 90 <= self.c_angle and self.b_angle <= 90 <= self.d_angle or \
                        self.b_angle <= 90 <= self.a_angle and self.c_angle <= 90 <= self.d_angle or \
                        self.c_angle <= 90 <= self.a_angle and self.d_angle <= 90 <= self.b_angle or \
                        self.a_angle <= 90 <= self.b_angle and self.d_angle <= 90 <= self.c_angle:
                    self.type = 'Трапеция'
                    return (Trapezoid(self.a, self.b, self.c, self.d))
                else:
                    self.type = 'Arbitrary_Quadrangle'
                    return (Arbitrary_Quadrangle(self.a, self.b, self.c, self.d))

                self.p = (self.ab + self.bc + self.cd + self.da)/2
                self.S = round(sqrt((self.p - self.ab)*(self.p - self.bc)*(self.p - self.cd)*(self.p - self.da)), 3)
class Quad():
  def __init__(self, a, b, c, d):
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    print('is a Quad')

class Trapezoid():
  def __init__(self, a, b, c, d):
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    print('is a Trapezoid')

class Diamond():
  def __init__(self, a, b, c, d):
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    print('is a Diamond')

class Rectangle():
  def __init__(self, a, b, c, d):
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    print('is a Rectangle')

class Arbitrary_Quadrangle():
  def __init__(self, a, b, c, d):
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    print('is a Arbitrary_Quadrangle')


def quadrangle_def():
  from numpy import sin, cos, deg2rad, sqrt, arccos, degrees

  figures = []
  c = Quadrangle(Quadrangle_point(0, 0), Quadrangle_point(0, 10), Quadrangle_point(10, 10), Quadrangle_point(10, 0)).set_a_type()
  figures.append( c )
  c = Quadrangle(Quadrangle_point(3, 0), Quadrangle_point(0, 3), Quadrangle_point(0, 10), Quadrangle_point(10, 10)).set_a_type()
  figures.append( c )
  c = Quadrangle(Quadrangle_point(0, 0), Quadrangle_point(-3, 5), Quadrangle_point(0, 10), Quadrangle_point(3, 5)).set_a_type()
  figures.append( c )
  c = Quadrangle(Quadrangle_point(0, 0), Quadrangle_point(10, 0), Quadrangle_point(13, 5), Quadrangle_point(3, 5)).set_a_type()
  figures.append( c )
  c = Quadrangle(Quadrangle_point(0, 0), Quadrangle_point(0, 4), Quadrangle_point(4, 4), Quadrangle_point(4, 0)).set_a_type()
  figures.append( c )
  c = Quadrangle(Quadrangle_point(-5, 6), Quadrangle_point(0, 0), Quadrangle_point(0, -6), Quadrangle_point(-4, 0)).set_a_type()
  figures.append( c )
  c = Quadrangle(Quadrangle_point(-10, 0), Quadrangle_point(-10, 20), Quadrangle_point(10, 20), Quadrangle_point(10, 0)).set_a_type()
  figures.append( c )
  c = Quadrangle(Quadrangle_point(-5, 0), Quadrangle_point(-2, 3), Quadrangle_point(2, 3), Quadrangle_point(5, 0)).set_a_type()
  figures.append( c )
  c = Quadrangle(Quadrangle_point(0, 0), Quadrangle_point(4, 0), Quadrangle_point(0, 4), Quadrangle_point(4, 4)).set_a_type()
  figures.append( c )
  c = Quadrangle(Quadrangle_point(0, 0), Quadrangle_point(3, 0), Quadrangle_point(2, 3), Quadrangle_point(4, 5)).set_a_type()
  figures.append( c )

def Maze():
	
	print('введите размерность')
	N = int(input('Введите N'))
	M = int(input('Введите M'))
	Random_array_top = [[0] * M] * (N)
	Random_array_right = [[0] * M] * (N)
	Random_array_left = [[0] * M] * (N)
	Random_array_down = [[0] * M] * (N)
	from random import randint
	for i in range(N):
		Random_array_top.append([])
		Random_array_down.append([])
		Random_array_right.append([])
		Random_array_left.append([])
		for j in range(M):
				if (i == 0) or (i == (N - 1)):   #границы сверху снизу
					if i == 0:
						Random_array_top[i][j] = 1
					else:
						if i == N - 1:
							Random_array_down[i][j] = 1
					if j == 0:
						Random_array_left[i][j] = 1
					else:
						Random_array_left[i][j] = randint(0, 1)
					if j == M - 1:
						Random_array_right[i][j] = 1
					else:
						Random_array_right[i][j] = randint(0, 1)
				elif j == 0 or j == M - 1:	#границы слева справа
					if j == 0:
						Random_array_left[i][j] = 1
					else:
						if j == M - 1:
							Random_array_right[i][j] = 1
				else:
					Random_array_right[i][j] = randint(0, 1)
					Random_array_top[i][j] = randint(0, 1)
					Random_array_left[i][j] = randint(0, 1)
					Random_array_down[i][j] = randint(0, 1)
	Maze_window = Toplevel(app)
	Maze_window.title("Лабиринт")
	canvas = Canvas(Maze_window, bg = 'white')
	canvas.pack(expand = YES, fill = BOTH)
	Scale_width = int(500/N)
	Scale_height = int(500/M)
	Maze_window.geometry("500x500")
	x = 40
	y = 40
	i = 0
	j = 0
	while i < N:
		while j < M:
			if Random_array_top[i][j]:
				canvas.create_line(x, y, x+Scale_width, y,width=2,fill="black")
				x += Scale_width
			else:
				x += Scale_width
			j += 1
		y += Scale_height
		j = 0
		x = 40
		i += 1 
	x = 40
	y = 40
	i = 0
	j = 0
	while i<N:
		while j < M:
			if Random_array_top[i][j]:
				canvas.create_line(x, y, x, y+Scale_height,width=2,fill="black")
				y += Scale_height
			else:
				y += Scale_height
			j += 1
		x += Scale_width
		j = 0
		i += 1 
	for wall in Random_array_down:
		pass
	for wall in Random_array_right:
		pass
	canvas.pack()


	print(Random_array_right)
	print(Random_array_top)
	print(Random_array_left)
	print(Random_array_down)




functions = {'zad11':zad1_1, 'zad12':zad1_2, 'zad13':zad1_3,
             'zad14':zad1_4, 'zad15':zad1_5, 'zad16':zad1_6,
             'zad17':zad1_7,
             'zad41':zad4_1, 'zad42':zad4_2, 'zad43':zad4_3,
             'zad51':zad5_1, 'zad52':zad5_2, 'zad53':zad5_3,
             'case':case, 'inp':inp,
             'a1':a1, 'a2':a2, 'a3':a3, 'a4':a4, 'a5':a5, 'a6':a6, 'a7':a7,
             'b1':b1, 'b2':b2, 'b3':b3, 'b4':b4, 'b5':b5, 'b6':b6, 'b7':b7, 'b8':b8,'b9':b9, 'b10':b10, 'b11':b11, 'b12':b12, 'b13':b13,
             'c1':c1, 'c2':c2, 'c3':c3, 'c4':c4, 'c5':c5, 'c6':c6,
             'd1':d1, 'd2':d2, 'd3':d3, 'd4':d4, 'd5':d5, 'd6':d6, 'd7':d7, 'd8':d8,'d9':d9, 'd10':d10, 'd11':d11, 'd12':d12,
             #'mp1':mp1, 'mp2':mp2, 'mp3':mp3, 'mp4':mp4, 'mp5':mp5,
             #'post1':post1 
             }
#functions['case'](innp)

app = Tk()
app.title("Практикум на Python")
app.geometry("500x300+200+200")

labelText = StringVar()
labelText.set ("Выберите практическое задание или задание по лекции")

labelText2 = StringVar()
labelText2.set ("Инструкция: \n \n \
Все вводы/выводы данных по заданиям производятся \n \
от данного главного меню, здесь вы можете \n \
увидеть поля ввода и вывода, порядок полей определяется слева направо")

labelText3 = StringVar()
labelText3.set ("\n Disclaimer \n \n \
Leonya`s Programming Practice Work`s Sda4a Program (LPPWSP) \n \
software was made by Leonya. This software \n \
comes with no guarantee. Use at your own risk")

label1 = Label(app, textvariable=labelText, height=0, width=100)
label1.pack()

label2 = Label(app, textvariable=labelText2, height=0, width=100)
label2.pack()

label3 = Label(app, textvariable=labelText3, height=0, width=100)
label3.pack()

labelText4 = StringVar()
labelText3.set ("\n Введите номер лабораторной и задачи \n \n \
в формате zad1 1 или evm1 4")

inputzad = StringVar()
inputzad_entry = Entry(textvariable=inputzad)
inputzad_entry.place(relx=.3, rely=.7, anchor="c")

inputnum = StringVar()
inputnum_entry = Entry(textvariable=inputnum)
inputnum_entry.place(relx=.7, rely=.7, anchor="c")

b = Button(app, text="Выход", width=20, command=app.destroy)
b.pack(side='bottom',padx=0,pady=0)

button1 = Button(app, text="Рассчитать", width=20, command=Input_exc_num)
button1.pack(side='bottom',padx=5,pady=5)

app.bind('<Return>', lambda event: Input_exc_num())

app.mainloop()







