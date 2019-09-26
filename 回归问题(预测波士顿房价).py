'''
@version: python3.6
@author: Administrator
@file: 回归问题(预测波士顿房价).py
@time: 2019/09/24
'''

#已知上世纪70年代的波士顿郊区的样本数据（包括犯罪率、房产税率等特征）和房屋价格，但数据量较少，只有506个，分成404训练样本和102个测试样本

from tensorflow import keras
import numpy as np
import matplotlib.pylab as plt
#1、获取数据
(train_data,train_targets),(test_data,test_targets) = keras.datasets.boston_housing.load_data()
print(train_data[0])#查看数据
print(train_targets[0])
print(len(train_data))
print(train_data.shape[0],train_data.shape[1])

#2、准备数据
#训练集和测试集的特征标准化，由于数据的特征都有不同的取值范围，所以进行特征缩放来加快模型训练
train_data-=train_data.mean(axis=0)
train_data/=train_data.std(axis=0)
test_data-=train_data.mean(axis=0)
test_data/=train_data.std(axis=0)
print(train_data[0:4])
#3、构建网络
#由于后面需要重复构建网络，所以为了复用，写一个函数来实现
def build_model():
    model=keras.models.Sequential()
    model.add(keras.layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',
                  metrics=['mae'])#回归问题使用均方误差来作为损失函数，平均绝对误差来作为训练过程的指标
    return model

#4、验证模型
#由于数据量较少，普通验证可能会有较大波动（方差较大）,所以我们可以使用k折交叉验证（与dropout很像）
k=4
num_val_samples=len(train_data)//k
num_epoch=100
all_mae_history=[]#保存所有循环中模型的平均绝对误差
for i in range(k):
    #从训练集中获取100个数据和标签作为验证集数据和样本
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets=train_targets[i*num_val_samples:(i+1)*num_val_samples]
    #除去验证集的剩下的部分作为新的训练集
    partial_train_data=np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets=np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)
    #构建模型，并且训练
    model=build_model()
    history=model.fit(partial_train_data,partial_train_targets,batch_size=1,epochs=250,verbose=0,validation_data=(val_data,val_targets))
    #获取训练过程的作为衡量的平均绝对误差
    print(history.history.keys())
    mae_history=history.history['val_mean_absolute_error']
    all_mae_history.append(mae_history)
#计算每一次循环的每一次epoch的平均绝对误差
average_mae_history=[np.mean([x[i] for x in all_mae_history]) for i in range(num_epoch)]
plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel('epoch')
plt.ylabel('validation MEA')
plt.show()


