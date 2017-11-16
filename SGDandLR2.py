# coding=utf-8

# 从sklearn.datasets导入波士顿房价数据读取器。
from sklearn.datasets import load_boston
# 从读取房价数据存储在变量boston中。
boston = load_boston()
# 输出数据描述。
#print boston.DESCR

# 从sklearn.cross_validation导入数据分割器。
from sklearn.model_selection import train_test_split

# 导入numpy并重命名为np。
import numpy as np

X = boston.data
y = boston.target

# 随机采样25%的数据构建测试样本，其余作为训练样本。
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)

# 分析回归目标值的差异。
#print "The max target value is", np.max(boston.target)
#print "The min target value is", np.min(boston.target)
#print "The average target value is", np.mean(boston.target)

# 从sklearn.preprocessing导入数据标准化模块。
from sklearn.preprocessing import StandardScaler

# 分别初始化对特征和目标值的标准化器。
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理。
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

#避免警告，但理由还不清楚，之后学习
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)



# 从sklearn.linear_model导入LinearRegression。
from sklearn.linear_model import LinearRegression

# 使用默认配置初始化线性回归器LinearRegression。
lr = LinearRegression()
# 使用训练数据进行参数估计。
lr.fit(X_train, y_train)
# 对测试数据进行回归预测。
lr_y_predict = lr.predict(X_test)

# 从sklearn.linear_model导入SGDRegressor。
from sklearn.linear_model import SGDRegressor

# 使用默认配置初始化线性回归器SGDRegressor。
sgdr = SGDRegressor()
# 使用训练数据进行参数估计。这里使用ravel（）方法避免警告，理由不清楚，还待学习
sgdr.fit(X_train, y_train.ravel())
# 对测试数据进行回归预测。
sgdr_y_predict = sgdr.predict(X_test)

# 使用LinearRegression模型自带的评估模块，并输出评估结果。
print 'The value of default measurement of LinearRegression is', lr.score(X_test, y_test)

# 从sklearn.metrics依次导入r2_score、mean_squared_error以及mean_absoluate_error用于回归性能的评估。
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 使用r2_score模块，并输出评估结果。
print 'The value of R-squared of LinearRegression is', r2_score(y_test, lr_y_predict)

# 使用mean_squared_error模块，并输出评估结果。
print 'The mean squared error of LinearRegression is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict))

# 使用mean_absolute_error模块，并输出评估结果。
print 'The mean absoluate error of LinearRegression is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict))

# 使用SGDRegressor模型自带的评估模块，并输出评估结果。
print 'The value of default measurement of SGDRegressor is', sgdr.score(X_test, y_test)

# 使用r2_score模块，并输出评估结果。
print 'The value of R-squared of SGDRegressor is', r2_score(y_test, sgdr_y_predict)

# 使用mean_squared_error模块，并输出评估结果。
print 'The mean squared error of SGDRegressor is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict))

# 使用mean_absolute_error模块，并输出评估结果。
print 'The mean absoluate error of SGDRegressor is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict))

