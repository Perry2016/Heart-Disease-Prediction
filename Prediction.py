import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_heart = pd.read_csv("heart.csv")

df_heart.head()

df_heart.target.value_counts()

plt.scatter(x=df_heart.age[df_heart.target==1],y=df_heart.thalach[df_heart.target==1],c="red")
plt.scatter(x=df_heart.age[df_heart.target==0],y=df_heart.thalach[df_heart.target==0],marker='^')
plt.legend(["Disease", "No Disease"])
plt.xlabel("Age")
plt.ylabel("Heart Rate")
plt.show()

X = df_heart.drop(['target'], axis = 1)
y = df_heart.target.values
#y = y.reshape(-1,1)
print("TensorFlow X Shape", X.shape)
print("TensorFlow y Shape", y.shape)

# Split the dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ML Model

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("SK learn prediction accuracy {:.2f}%".format(lr.score(X_test, y_test)*100))
