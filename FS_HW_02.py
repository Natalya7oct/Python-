#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


import matplotlib.pyplot as plt


# In[3]:


from sklearn.datasets import load_boston
import pandas as pd
data=load_boston()


# In[4]:


df = pd.DataFrame(data.data, columns=data.feature_names)
df['MEDV'] = data.target
df.head()


# In[5]:


print(load_boston()['DESCR'])


#         - CRIM уровень преступности на душу населения по городам
#         - ZN доля жилой земли для больших домов
#         - INDUS доля земли, занятой производством, на город??
#         - CHAS наличие реки (1) или отсутствие (0)
#         - NOX концентрация оксидов азота
#         - RM среднее количество комнат на жилое помещение
#         - AGE доля занимаемых владельцами квартир, построенных до 1940 года???
#         - DIS взвешенное расстояние до пяти бостонских центров занятости
#         - RAD индекс доступности к радиальным магистралям
#         - TAX Налоговая ставка на имущество на 10 000 долларов США
#         - PTRATIO Соотношение учеников и учителей городам
#         - B доля чернокожих по городам.
#         - LSTAT процент малообеспеченного населения
#         - MEDV Медианная стоимость домов, занимаемых владельцами, в $ 1000

# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.describe(include='all')


# In[9]:


df.isnull().sum()


# Пропусков нет, уже хорошо.

# In[10]:


plt=sns.pairplot(df)


# Распределение MEDV похоже на нормальное, но правый хвост задернут. Это проблема для линейной регрессии?

# Переменные  MEDV и LSTAT  не имеют выбросов судя по гистограммам.

# Посмотрим на корреляцию некоторых, наиболее интересных переменных.

# In[44]:


cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'AGE','MEDV']
hm = sns.heatmap(df[cols].corr(),
                 cbar=True,
                 annot=True)


# Самое сильное влияние оказывает переменная LSTAT - процент малообеспеченного населения. Это и не удивительно, ведь, действительно, чем больше в районе малообеспеченных граждан, тем дешевле квартиры в нем.

# LSTAT скоррелировано с остальными переменными в значительной степени, 
# поэтому нет смысла добавлять в модель что-то еще кроме LSTAT.

# In[17]:


sns.regplot(x="LSTAT", y="MEDV", data=df) 


# Видим, что действительно имеется линейная зависимость между переменными LSTAT и MEDV. Но после линейной зависимости попробуем построить полиноминальную регрессию.

# Постоим линейную модель по переменной LSTAT.

# In[13]:


X = df[['LSTAT']].values
y = df['MEDV'].values


# In[24]:


from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y) #обучаем модель
y_pred = slr.predict(X) #получаем предсказанные значения
print('Slope: {:.2f}'.format(slr.coef_[0])) #выводим угол наклона прямой
print('Intercept: {:.2f}'.format(slr.intercept_)) #выводим свободную переменную


# Проверим качество модели.

# In[26]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0) #разделили на тестовые данные и обучающие (3:7)


# In[20]:


slr = LinearRegression()

slr.fit(X_train, y_train) #обучили обучающую выборку (а что тут еще скажешь, такая вот тавтология)
y_train_pred = slr.predict(X_train) #получаем предсказанные значения для обучающей выборки
y_test_pred = slr.predict(X_test) #получаем предсказанные значения для тестовой выборки


# Посчитаем средне-квадратичное отклонение и коэффициент детерминации

# In[28]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

print('MSE train: {:.3f}, test: {:.3f}'.format(
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: {:.3f}, test: {:.3f}'.format(
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# Средне-квадратичное отклонение на тестовой и обучающей выборках отличаются незначительно,
# но само по себе оно очень большое и сравнимо со максимальным значением предсказываемой переменной.
# Коэффициенты детерминации совпадают, но совсем маленькие, чуть-чуть больше 50%.
# Это говорит о том, что наша модель предсказывает маленький процент дисперсии, да и то плохой=))

# Построим график остатков Residuals plot. C его помощью мы можем увидеть нелинейность и выбросы, проверить случайность распределения ошибки.
# 
# 

# In[32]:


plt.scatter(y_train_pred,  y_train_pred - y_train,
            c='blue', marker='o', label='Training data') #остатки обучающей выборки
plt.scatter(y_test_pred,  y_test_pred - y_test,
            c='lightgreen', marker='s', label='Test data') #остатки тестовой выборки
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='red')
plt.xlim([-10, 50])
plt.tight_layout()


# In[ ]:


В остатках проглядывается некоторая закономерность, они не распределены нормально.


# In[ ]:


Попробуем полиноминальное распределение.


# In[47]:


from sklearn.preprocessing import PolynomialFeatures


# In[48]:


X = df[['LSTAT']].values
y = df['MEDV'].values


# In[49]:


regr = LinearRegression()


# In[50]:


quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)


# In[56]:


X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))
linear_ms = mean_squared_error(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))
quadratic_ms = mean_squared_error(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))
cubic_ms = mean_squared_error(y, regr.predict(X_cubic))


# In[72]:


plt.scatter(X, y, label='training points', color='lightgray')

plt.plot(X_fit, y_lin_fit, 
         label='linear (d=1), $R^2={:.2f}$'.format(linear_r2)+'$, MS={:.2f}$'.format(linear_ms),
         color='blue', 
         lw=2, 
         linestyle=':')

plt.plot(X_fit, y_quad_fit, 
         label='quadratic (d=2), $R^2={:.2f}$'.format(quadratic_r2)+'$, MS={:.2f}$'.format(quadratic_ms),
         color='red', 
         lw=2,
         linestyle='-')

plt.plot(X_fit, y_cubic_fit, 
         label='cubic (d=3), $R^2={:.2f}$'.format(cubic_r2)+'$, MS={:.2f}$'.format(cubic_ms),
         color='green', 
         lw=2, 
         linestyle='--')

plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper right')


# In[ ]:


Судя по графику, кубическая функция аппроксимирует данные лучше всего.


# In[ ]:




