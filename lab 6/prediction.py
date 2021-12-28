import pandas as pd
import  numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('result.csv')


data['isWin'] = data['isWin'].astype(int)
data['time'] = data['time'].astype(float)

print(data.corr())

win = data['isWin']
features = data.drop(['isWin', 'algorithm'], axis=1)

# по скору и времени ---> шанс
X_train, X_test, Y_train, Y_test = train_test_split(features, win, test_size=0.1, random_state=10)

reg = LinearRegression()
reg.fit(X_train, Y_train)
print()
print('Score', reg.score(X_train,Y_train)) # точность модели
print('Intercept', reg.intercept_) # уровень наклона линии на графике
print('Coef', reg.coef_) # параметр при наклоне
print(reg.predict(X_test)) # prediction на 5 шагов вперед (очки, время)




# score
model = LinearRegression()

x = pd.DataFrame(data['score'])
y = pd.DataFrame(data['isWin'])

model.fit(x,y)
plt.scatter(data['score'], data['isWin'])
plt.plot(x, model.predict(x), color = 'green')
plt.xlabel('score')
plt.ylabel('win')
plt.show()



X_test = pd.DataFrame(data['score'].head(5))
print(model.predict(X_test))


# time
model2 = LinearRegression()

x = pd.DataFrame(data['time'])
y = pd.DataFrame(data['isWin'])

model2.fit(x,y)
plt.scatter(data['time'], data['isWin'])
plt.plot(x, model.predict(x), color = 'green')
plt.xlabel('time')
plt.ylabel('win')
plt.show()

print(f"Score: {model.score(x,y)}")

X_test = pd.DataFrame(data['time'].head(5))
print(model2.predict(X_test))