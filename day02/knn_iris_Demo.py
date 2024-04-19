from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def knn_iris_demo():
    """
    用KNN算法对鸢尾花进行分类,并测试测试模型
    :return:
    """
    # 1.获取数据
    iris = load_iris()
    # 2.划分数据类
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)
    # 3.特征工程 : 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4.KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=5)
    estimator.fit(x_train, y_train)

    # 5.模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict\n", y_predict)
    # 方法一:直接比对真实值和预测值
    print("直接比对:\n", y_test == y_predict)
    # 方法二:计算准确率
    accuracy = estimator.score(x_test, y_test)
    print("Accuracy:\n", accuracy)
    return None


def knn_iris_gscv():
    """
    用KNN算法对鸢尾花进行分类,并测试测试模型
    :return:
    """
    # 1.获取数据
    iris = load_iris()
    # 2.划分数据类
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)
    # 3.特征工程 : 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4.KNN算法预估器
    estimator = KNeighborsClassifier()
    # 加入网格收缩与交叉验证模型优化
    param_grid = [{'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 17, 19, 21]}]
    estimator = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1, cv=10)
    estimator.fit(x_train, y_train)

    # 5.模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict\n", y_predict)
    # 方法一:直接比对真实值和预测值
    print("直接比对:\n", y_test == y_predict)
    # 方法二:计算准确率
    accuracy = estimator.score(x_test, y_test)
    print("Accuracy:\n", accuracy)

    # 最佳参数
    print("最佳参数:\n", estimator.best_params_)
    # 最佳结果
    print("最佳结果:\n", estimator.best_score_)
    # 最佳估计器
    print("最佳估计器:\n", estimator.best_estimator_)
    # 交叉验证结果
    print("交叉验证结果:\n", estimator.cv_results_)
    return None


if __name__ == '__main__':
    # 代码1: 使用KNN对鸢尾花分类
    # knn_iris_demo()

    # 代码2: 使用KNN对鸢尾花分类,并进行调优
    knn_iris_gscv()
