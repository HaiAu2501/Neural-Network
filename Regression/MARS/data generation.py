import csv
import random

N=10000 # No. of data
C=[] #Coefficients of y
C0=100 #Range of coeficients
X0=100 #Range of x
for i in range (0,2*random.randint(3,10)): #Generate function
    C.append(random.uniform(-C0,C0))
def rh(x,t): #Left hinge function
    return max(0,x-t)
def lh(x,t): #Right hinge function
    return max(0,t-x)
def f(x):
    s=0
    for i in range (0,len(C)):
        if(i%2==0):
            s+=rh(x,C[i])
        else:
            s+=lh(x,C[i])
    return s

with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["x", "y"]
    for i in range (0,N): # Generate data
        a=random.uniform(-X0,X0)
        b=f(a)
        writer.writerow([a, b])