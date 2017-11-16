# coding=utf-8

#数据导入和预处理
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#观察数据
#print titanic.head(), titanic.info()
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
#探查x中的元素的情况以制定数据预处理的方法
#print X.info()
X['age'].fillna(value=X['age'].mean(), inplace=True)
#print X.info()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
#特征转换
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

#print vec.feature_names_


# 从sklearn.tree中导入决策树分类器。
from sklearn.tree import DecisionTreeClassifier
# 使用默认配置初始化决策树分类器。
dtc = DecisionTreeClassifier()
# 使用分割到的训练数据进行模型学习。
dtc.fit(X_train, y_train)
# 用训练好的决策树模型对测试特征数据进行预测。
y_predict = dtc.predict(X_test)

# 从sklearn.metrics导入classification_report。
from sklearn.metrics import classification_report
# 输出预测准确性。
print dtc.score(X_test, y_test)
# 输出更加详细的分类性能。
print classification_report(y_predict, y_test, target_names = ['died', 'survived'])
