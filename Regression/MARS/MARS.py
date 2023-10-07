import csv
from scipy import stats

x=[]
y=[]
xtest=[]
ytest=[]

#Read data
with open('data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        line_count +=1
N=line_count #No. of data
trN=(6*N)/10 #No. of training data, 60%
vN=(3*N)/10 #No. of validating data, 30%
tN=N-trN-vN #No. of testing data, 10%
with open('data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if(line_count<trN+vN):
            x.append(row[0])
            y.append(row[1])
            line_count +=1
        else:
            xtest.append(row[0])
            ytest.append(row[1])
            line_count +=1

def rh(x,t): #Left hinge function
    return max(0,x-t)
def lh(x,t): #Right hinge function
    return max(0,t-x)

def MARS(trainingset,validateset):
    
    