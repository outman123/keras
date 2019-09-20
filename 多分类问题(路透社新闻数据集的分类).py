'''
@version: python3.6
@author: Administrator
@file: 多分类问题(路透社新闻数据集的分类).py
@time: 2019/09/20
'''

from tensorflow import keras
import numpy as np
import matplotlib.pylab as plt

#1、加载路透社数据集，每个样本都是一个整数列表，10000表示每个新闻中的单词只取前10000个，即列表长度最大为10000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.reuters.load_data(num_words=10000)
print(train_data[10])

#2、准备数据
# one-hot编码，将整数序列编码为二进制矩阵，方便识别分类
def vectories_sequences(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))  # 构造形状为(len(sequences),dim)的矩阵
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0  # 将resuls[i]指定的索引设置为1，sequence实际上是一系列整数
    return results

# 将训练数据向量化
x_train = vectories_sequences(train_data)
x_test=vectories_sequences(test_data)
#将标签向量化，使用内置方法（其实样本向量化也可以使用内置函数，无需自定义）
one_hot_train_labels=keras.utils.to_categorical(train_labels)
one_hot_test_labels=keras.utils.to_categorical(test_labels)

#3、构建网络，不同于二分类，类别从2增加到46，输出维度增大，所以中间隐藏层节点数也需要增大
model=keras.models.Sequential()
model.add(keras.layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(keras.layers.Dense(64,activation='relu'))#不需要指定该层的输入形状，该框架会自动识别匹配层之间的连接
model.add(keras.layers.Dense(46,activation='softmax')) # relu函数将所有负值归0，使用softmax，将输出46种不同输出类别的概率分布。

#编译模型，由于多分类，使用categorical_crossentropy损失函数，衡量预测输出和真实标签的分布之间的距离，并使之最小化
model.compile(optimizer=keras.optimizers.RMSprop(lr=0.01),loss=keras.losses.categorical_crossentropy,metrics=[keras.metrics.categorical_accuracy])

#4、验证模型
#从训练集中拿出前1000作为验证集
x_val=x_train[:1000]
x_train=x_train[1000:]
y_val=one_hot_train_labels[:1000]
y_train=one_hot_train_labels[1000:]
#训练模型,一个批次512个样本，训练集一共使用9个轮次，因为绘制出训练和验证的损失曲线后，会发现eopchs过大会过拟合,设置9较合适
history=model.fit(x_train,y_train,epochs=9,batch_size=512,validation_data=(x_val,y_val))
#绘制训练精度和验证精度

#print(history.history.keys())
# 获取训练和验证过程的值数据，即y轴数据
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
# 获取x轴数据
epoch = range(1, len(acc) + 1)
# 画图
plt.plot(epoch, acc, 'bo', label='训练过程精确度')
plt.plot(epoch, val_acc, 'b', label='验证过程精确度')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False
plt.title('训练和验证过程的精确值变化')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

#5、利用模型预测结果
#predict()函数返回测试数据在46种类别上的概率分布,概率最大的元素就是预测类别。注意该数据集的标签是整数，只是真正类别的索引
predictions=model.predict(x_test)
print(np.argmax(predictions[0]))