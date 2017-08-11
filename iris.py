import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('IRIS.csv')

w1 = 1/len(df.columns)
w2 = 1/len(df.columns)
w3 = 1/len(df.columns)
w4 = 1/len(df.columns)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0,2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.legend(loc='upper left')
plt.show()

