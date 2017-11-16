# coding=utf-8

from sklearn.datasets import load_iris
iris = load_iris()
# print iris.data.shape

#查看数据说明
#print iris.DESCR

#数据分割
from sklearn.model_selection import train_test_split
#将数据集分为训练集和测试集，test_size是测试集占训练集的比例
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

#数据标准化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#使用k邻近法
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)

# 依然使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析。
from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names = iris.target_names)
