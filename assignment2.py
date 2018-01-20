'''
Created on Oct 12, 2017
@author: sagar
'''
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_sc
import csv
'''Read the sentence file and make list of sentence and annotator 1 response
'''
with open('DialogueActs_Homework2.csv', 'r') as csvfile:
costructList = csv.reader(csvfile, delimiter=',', quotechar='|')
sentence=[]
classify=[]
for row in costructList:
sentence.append(row[0])
classify.append(row[1])
sentence=sentence[1:]
classify=classify[1:]
''' for each values in classify, encode with one hot values'''
values=array(classify)
print(values)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
def findMaxWidth(list):
max=0
for i in list:
if len(i)>max:
max=len(i)
return max
‘’’ Pos Tagging for each sentence’’’
list=[]
featureSet=[]
for j in sentence:
t=nltk.word_tokenize(j)
tags=[i[1]for i in nltk.pos_tag(t)]
tagsForEachSentence=[dict[i[1]]/45 for i in nltk.pos_tag(t)]
list.append(tagsForEachSentence)
featureSet.append(tags)with open("featureSetPosTags1.csv",'w') as resultFile:
wr = csv.writer(resultFile, dialect='excel')
for rows in featureSet:
wr.writerow(rows)
x=np.zeros((100,16))
for i in range(100):
for j in range(len(list[i])):
x[i][j]=list[i][j]
print(x)
x_train=x[1:80]
x_test=x[81:100]
y_train=onehot_encoded[1:80]
y_test=onehot_encoded[81:100]
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(x_train, y_train)
# Make predictions using the testing set
y_pred = regr.predict(x_test)
print(y_pred)
temp=[np.argmax(i) for i in y_pred]
# The mean squared error
print("Mean squared error: %.2f"
% mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
#accuracy
temp2=[i[0] for i in integer_encoded[81:]]
print(temp2)
sameCount=0
for i in range(len(temp)):
if temp[i]==temp2[i]:
sameCount=count+1
print("Accuracy="+str(100*sameCount/20)+"%"
