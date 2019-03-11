import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
import random

def display(nplis):
    for i in nplis:
        for j in i:
            if j == 5:
                print('x',end=' ')
            elif j == -5:
                print('o',end=' ')
            else:
                print('b',end=' ')
        print()

def checkWins(nplis):
    for z in nplis:
        if sum(z) == 15:
            print("Sorry U lost!!")
            quit()
        elif sum(z) == -15:
            print("You Win!!")
            quit()
    diag = np.diag(nplis)
    if sum(diag) == 15:
        print("Sorry U lost!!")
        quit()
    elif sum(diag) == -15:
        print("You Win!!")
        quit()
    diag = np.diag(np.rot90(nplis))
    if sum(diag) == 15:
        print("Sorry U lost!!")
        quit()
    elif sum(diag) == -15:
        print("You Win!!")
        quit()
        
def checkSums(nplis):
    for z in range(3):
        if sum(nplis[z]) == 10 or sum(nplis[z]) == -10:
            for i in range(3):
                if nplis[z,i] == 0:
                    return (z,i)
    diag = np.diag(nplis)
    if sum(diag) == 10 or sum(diag) == -10:
        for i in range(3):
            if diag[i] == 0:
                return (i,i)
    diag = np.diag(np.rot90(nplis))
    if sum(diag) == 10 or sum(diag) == -10:
        for i in range(3):
            if diag[i] == 0:
                return (i,2-i)
    for cols in range(3):
        if nplis[0,cols] + nplis[1,cols] + nplis[2,cols] in (-10,10):
            for i in range(3):
                if nplis[i,cols] == 0:
                    return (i,cols)            
    return (-99,-99)

pd.set_option('display.max_columns', None) # to display all the columns

df = pd.read_csv('tic-tac-toe.data.txt')
df.replace('x',5, inplace=True)
df.replace('o',-5, inplace=True)
df.replace('b',0, inplace=True)
df.replace('positive',1, inplace=True)
df.replace('negative',-1, inplace=True)
#print(df.head())

X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()#LinearRegression()#svm.SVR()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

example_measures = np.array([[5,0,0,0,0,0,0,0,0]])

prediction = clf.predict(example_measures)
#print(accuracy, prediction)

lis = [0,0,0,0,0,0,0,0,0]
print('Indexes are from 0-8 x=Computer o=User b=Blank')
for i in range(4):
    flag = 0
    x,y = checkSums(np.reshape(np.array(lis),(3,3)))
    if (x,y) != (-99,-99):
        flag = 1
        ind = 3*x + y
        lis[ind] = 5
    if flag == 0:
        for i in range(9):
            if lis[i] == 0:
                temp = lis[:]
                temp[i] = 5
                if clf.predict([temp,]) > 0:
                    lis[i] = 5
                    flag = 1
                    break
    #print(flag,lis)
    
    while flag == 0:
        place = random.randrange(0,9)
        if lis[place] == 0:
            lis[place] = 5
            flag = 1
    nplis = np.array(lis)
    nplis = np.reshape(lis,(3,3))

    display(nplis)

    checkWins(nplis)
    nplis = nplis.T
    checkWins(nplis)
    while True:
        user_move = int(input('Your Turn'))
        if lis[user_move] == 0:
            lis[user_move] = -5
            break
        else:
            print("Already filled")
    nplis = np.array(lis)
    nplis = np.reshape(lis,(3,3)) 
    checkWins(nplis)
    nplis = nplis.T
    checkWins(nplis)
print('Draw!!')
