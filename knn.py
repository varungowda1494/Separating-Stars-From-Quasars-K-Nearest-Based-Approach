from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from csv import reader
from math import *
import matplotlib.pyplot as plt


def strfloat(dataset, column):
    for row in dataset:
        row[column] = float(row[column])
        
def distance(row1, row2):
    dist = 0.0
    for i in range(len(row1)-1):
        dist += (row1[i] - row2[i])**2
    return sqrt(dist)
 
def neighbors(train, test_row, n):
    distances = []
    for train_row in train:
        dist = distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(n):
        neighbors.append(distances[i][0])
    return neighbors
 
def predict(train, test_rows, n):
    predictions=[]
    for i in test_rows:
        nbrs = neighbors(train,i, n)
        matches = [row[-1] for row in nbrs]
        predicted = max(set(matches), key=matches.count)
        predictions.append(predicted)
    return predictions

def acc(test_data,predictions):
    correct = 0
    for i in range(len(test_data)):
        if test_data[i][-1] == predictions[i]:
            correct += 1
    accuracy = float(correct)/len(test_data) * 100   
    return accuracy

bc = []
filename='MiniProject/may-this-be-the-last-time/catalog1/cat1.1.1.csv'
with open(filename, 'r') as opfile:
    total = reader(opfile)
    for row in total:
        bc.append(row)

for i in range(len(bc[0])-1):
    strfloat(bc, i)

check_data,test_data=train_test_split(bc,test_size=0.2,random_state = 30)
predictions=predict(check_data,test_data,15)
print(acc(test_data,predictions))

#To predict : predict(check_data,required_values,15)

#plotting k and random values v/s accuracy
k=[]
acc=[]
for i in range(1,50):
    k.append(i)
    check_data,test_data=train_test_split(bc,test_size=0.2,random_state = i)
    predictions=predict_classification(check_data,test_data,15)
    acc.append(accuracy(test_data,predictions))
plt.plot(k,acc)
plt.show()

#confustion matrix and classification report
actual=[]
for i in range(len(test_data)):
    actual.append(test_data[i][-1])
print('confusion matrix')
print(confusion_matrix(actual,predictions))
print('classification_report ')
print(classification_report(actual,predictions))

#plotting the distribution of data
x1=[]
y1=[]
x0=[]
y0=[]
for i in bc:
    if(i[21]=='1'):
        x1.append(i[3])                      # to do same for all other features vs nuv-z feature
        y1.append(i[4])
    else:
        x0.append(i[3])
        y0.append(i[4])
plt.scatter(x1,y1,color='red')
plt.scatter(x0,y0,color='blue')
plt.xlabel('nuv-i')
plt.ylabel('nuv-z')
plt.show()
