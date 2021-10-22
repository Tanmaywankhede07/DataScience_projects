import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('creditcard.csv')

#The .shape attribute of pandas.DataFrame stores the number of rows and columns
# as a tuple (number of rows, number of columns).
print(dataset.shape) #output will be (284807,31) i.e(rows,columns)

#The .head() function is used to get the first n rows.
#try print(dataset.head(5))
dataset.head()

#.info() display information such as the number of rows and columns, the total memory usage, 
#the data type of each column, and the number of non-NaN elements.
dataset.info()

#The describe() method is used for calculating some statistical data like percentile,
#mean and std of the numerical values of the Series or DataFrame.
#Try print(dataset.describe())
dataset.describe()

print('------------------------------------------------------')
class_names = {0:'Not Fraud', 1:'Fraud'}
print(dataset.Class.value_counts().rename(index = class_names))

print('------------------------------------------------------')
splitdata = dataset.iloc[:,1:30].columns
target = dataset.iloc[:1,30:].columns
print(splitdata)
print(target)
data_split = dataset[splitdata]
data_target = dataset[target]

#Train and test our model
#train_size = 0.70 means using 70% of dataset for training the model and 30% for test
X_train, X_test, y_train , y_test = train_test_split(data_split, data_target, train_size = 0.70,test_size=0.30,random_state=1)
print('------------------------------------------------------') 
print("Length of X_train is: {X_train}".format(X_train = len(X_train)))
print("Length of X_test is: {X_test}".format(X_test = len(X_test)))
print("Length of y_train is: {y_train}".format(y_train = len(y_train)))
print("Length of y_test is: {y_test}".format(y_test = len(y_test)))


model = LogisticRegression()
model.fit(X_train, y_train.values.ravel())
pred = model.predict(X_test)

class_names = ['not_fraud', 'fraud']
matrix = confusion_matrix(y_test, pred)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
#Name of Graph
plt.title("Confusion Matrix"), plt.tight_layout()
#Name of Axis
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
#To show the Predicted Graph
plt.show()
