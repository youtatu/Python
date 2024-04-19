from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import jieba


def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取数据集
    iris = load_iris()
    print("鸢尾花数据集:\n", iris)
    print("鸢尾花数据描述:\n", iris["DESCR"])
    # print("鸢尾花数据描述:\n", iris.DESCR) 也可以
    print("查看特征值的名字:\n", iris.feature_names)
    print("查看鸢尾花标签名:\n", iris.target_names)
    print("查看特征值:\n", iris.data)
    # 想看返回什么形状
    print("查看返回形状:\n", iris.data, "\n", iris.data.shape)
    print("查看目标值:\n", iris.target)

    """
    数据集划分
    """
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集特征值:\n", x_train, "\n", x_train.shape)
    return None


def dict_demo():
    """
    字典特征提取
    :return:
    """
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 80},
            {'city': '深圳', 'temperature': 60}, ]

    # 1. 实例化一个转换器类
    transfer = DictVectorizer(sparse=False)
    # 2. 调用fit_transform()
    data_new = transfer.fit_transform(data)
    data_new2 = transfer.inverse_transform(data_new)
    data_new3 = transfer.get_feature_names_out()

    print(data_new)
    print(data_new2)
    print(data_new3)

    return None


def count_demo():
    """
    英文文本特征提取: CountVectorizer
    :return:
    """
    data = ["life is short, i like like python", "life is too long, i dislike python"]
    # 1. 实例化一个转换器
    transfer = CountVectorizer()
    # 可以设置停用词, 网上有专业的停用词表
    # transfer = CountVectorizer(stop_words=["is", "like"])
    # 2. 调用fit_transform
    new_data = transfer.fit_transform(data)
    print("new_data:\n", new_data.toarray())
    new_data2 = transfer.inverse_transform(new_data)
    print("原数据:\n", new_data2)
    new_data3 = transfer.get_feature_names_out()
    print("特征名字:\n", new_data3)

    return None


def count_chinese_demo():
    """
    中文文本特征提取: CountVectorizer
    :return:
    """
    data = ["我 喜欢 他的 小狗,我 会 追着 他 跑", "他的 小狗 不喜欢 我,而且 追着 我 咬"]
    # 1. 实例化一个转换器
    transfer = CountVectorizer()
    # 2. 调用fit_transform
    new_data = transfer.fit_transform(data)
    print("new_data:\n", new_data.toarray())
    new_data2 = transfer.inverse_transform(new_data)
    print("原数据:\n", new_data2)
    new_data3 = transfer.get_feature_names_out()
    print("特征名字:\n", new_data3)
    return None


def count_chinese_demo2():
    """
    中文文本特征抽取:自动分词
    :param: 需要处理的中文文本, 字符串格式
    :return:分词结果, 列表格式
    """
    data = ["今天很残酷,明天更残酷,后天很美好,但是绝大部分是死在明天晚上,所以每个人不要放弃今天.",
            "我们看到的从很远星系来的光是在几百万年之前发出的,这样当我们看到宇宙时,我们是在看他的过去.",
            "如果只用一种方式了解某样食物,你就不会真正了解他.了解事物的真正含义的秘密取决于如何将其与我们所了解的事物相联系."]
    # 将中文文本分词
    data_cut = []
    for text in data:
        data_cut.append(cut_word(text))

    # 接下来与前面一样
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data_cut)
    print("data_new: \n", data_new.toarray())
    data_new2 = transfer.inverse_transform(data_new)
    print("原数据: \n", data_new2)
    data_new3 = transfer.get_feature_names_out(data_cut)
    print("特征名字: \n", data_new3)


def tfidf_demo():
    """
    使用TF-idf方法进行文本特征抽取
    :return:
    """
    data = ["今天很残酷,明天更残酷,后天很美好,但是绝大部分是死在明天晚上,所以每个人不要放弃今天.",
            "我们看到的从很远星系来的光是在几百万年之前发出的,这样当我们看到宇宙时,我们是在看他的过去.",
            "如果只用一种方式了解某样食物,你就不会真正了解他.了解事物的真正含义的秘密取决于如何将其与我们所了解的事物相联系."]
    # 将中文文本分词
    data_cut = []
    for text in data:
        data_cut.append(cut_word(text))

    # 接下来与前面一样
    transfer = TfidfVectorizer()
    data_new = transfer.fit_transform(data_cut)
    print("data_new: \n", data_new.toarray())
    data_new2 = transfer.inverse_transform(data_new)
    print("原数据: \n", data_new2)
    data_new3 = transfer.get_feature_names_out(data_cut)
    print("特征名字: \n", data_new3)


def cut_word(text):
    """
    分词函数
    进行中文分词
    :param text: 中文文本
    :return: 分词结果
    """
    cut_result = " ".join(list(jieba.cut(text)))
    return cut_result


def minmax_demo():
    """
    归一化
    :return:
    """
    # 1. 获取数据
    data = pd.read_csv("dating.txt")
    # print("data:\n", data)
    data = data.iloc[:, :3]  # 输出前三列
    # print("data:\n", data)

    # 2. 实例化一个转换器
    # transfer = MinMaxScaler()
    # 这里也可以设置范围
    transfer = MinMaxScaler(feature_range=(2, 3))

    # 3. 调用转换器
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None


def stand_demo():
    """
    标准化
    :return:
    """
    # 导入数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    # 创建实例对象
    transfer = StandardScaler()

    data_new = transfer.fit_transform(data)
    print("标准化结果 \n", data_new)
    print("特征平均值:\n", transfer.mean_)
    print("特征方差:\n", transfer.var_)

    return None


def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1. 获取数据
    data = pd.read_csv("factor_returns.csv")
    # 截取所有行,列从第一行开始到倒数第二列
    data = data.iloc[:, 1:-2]

    # 2. 实例化   阈值默认为0
    # transfer = VarianceThreshold()
    # 设置的阈值为5，表示特征的方差低于5的将会被移除
    transfer = VarianceThreshold(threshold=5)

    # 3. 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new \n", data_new, "\n", data_new.shape)

    # 使用pearsonr公式计算相关系数

    r1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
    print("pe_ratio与pb_ratio相关性: \n", r1)
    r2 = pearsonr(data["revenue"], data["total_expense"])
    print("revenue与total_expense相关性: \n", r2)
    # 绘制一下图形
    plt.figure(figsize=(20, 8), dpi=100)
    plt.scatter(data['revenue'], data['total_expense'])
    plt.show()


def pca_demo():
    """
    PCA降维
    :return:
    """
    data = [[2, 8, 5, 4], [6, 3, 0, 8], [5, 4, 9, 1]]

    # 1. 实例化一个转换器
    # 将4个特征降成2个
    # transfer = PCA(n_components=2)
    # 如果是小数,就是保留百分之多少的信息
    transfer = PCA(n_components=0.95)
    # 2. 调用fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None
   

if __name__ == '__main__':
    # 代码1:sklearn数据集的使用
    # datasets_demo()

    # 代码2: 字典特征提取
    # dict_demo()

    # 代码3: 英文文本特征提取:CountVectorizer
    # count_demo()

    # 代码4: 中文文本特征提取,需要分词(免费的有结巴分词)
    # count_chinese_demo()

    # 代码5: 中文文本特征抽取,自动分词
    # count_chinese_demo2()

    # 代码6: 使用TF-idf方法进行文本特征抽取
    # tfidf_demo()

    # 代码7: minmax归一化
    # minmax_demo()

    # 代码8: 标准化
    # stand_demo()

    # 代码9: 过滤低方差特征
    # variance_demo()

    # 代码10. PCA降维
    pca_demo()
