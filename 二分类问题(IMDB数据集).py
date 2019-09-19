'''
@version: python3.6
@author: Administrator
@file: 二分类问题(IMDB数据集).py
@time: 2019/09/19
'''


import tensorflow as tf
from tensorflow import keras#将tensorflow作为keras的后端
import numpy as np
import matplotlib.pylab as plt
#1、获取和准备数据

#one-hot编码,将整数序列编码为二进制矩阵，方便识别分类
def vectories_sequences(sequences,dim=10000):
    results=np.zeros((len(sequences),dim))#构造形状为(len(sequences),dim)的矩阵
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.0#将resuls[i]指定的索引设置为1
    return results

#获取imdb评论数据集，最终实现判断评论是好的的概率的功能
(train_data,train_labels),(test_data,test_labels) = keras.datasets.imdb.load_data(num_words=10000)
print(train_data[0])
print(train_labels)
#将训练数据向量化
x_train=vectories_sequences(train_data)
#x_test=vectories_sequences(test_data)
#标签向量化
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')

#2、构建网络

model=keras.models.Sequential()#层的线性堆叠
model.add(keras.layers.Dense(16,activation='relu',input_shape=(10000,)))#该层神经元有16个，激活函数为relu,输入张量的行数是10000
model.add(keras.layers.Dense(16,activation='relu'))#不需要指定该层的输入形状，该框架会自动识别匹配层之间的连接
model.add(keras.layers.Dense(1,activation='sigmoid'))#relu函数将所有负值归0，sigmoid函数将任意值压缩到[0,1]之间，所以输出可以看作概率值
#编译模型，指定优化器，损失函数和衡量指标
model.compile(optimizer=keras.optimizers.RMSprop,loss=keras.losses.binary_crossentropy(),metrics=[keras.metrics.binary_accuracy()])

#3、验证模型
#留出验证集
x_val=x_train[:10000]
x_train=x_train[10000:]
y_val=y_train[:10000]
y_train=y_train[10000:]
#训练模型,一个批次512个样本，训练集一共使用20个轮次
history=model.fit(x_train,y_train,epochs=10,batch_size=512,validation_data=(x_val,y_val))
#绘制训练损失和验证集的损失

#history.history.keys()#查看训练过程中的返回数据
history_dict=history.history
#获取训练和验证过程的损失值数据——y轴数据
loss=history_dict['loss']
val_loss=history_dict['val_loss']
#获取x轴数据
epoch=range(1,len(loss)+1)
#画图
plt.plot(epoch,loss,'bo',label='training loss')
plt.plot(epoch,val_loss,'b',label='validation loss')
plt.title('训练和验证过程的损失值变化')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()