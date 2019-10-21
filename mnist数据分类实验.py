'''
@version: python3.6
@author: Administrator
@file: mnist数据分类实验.py
@time: 2019/10/15
'''


from tensorflow import keras
import numpy as np
import matplotlib.pylab as plt
#1、获取数据
(train_data,train_label),(test_data,test_label)= keras.datasets.mnist.load_data()
print(train_data.shape,test_data.shape)
print(train_label)
#2、数据预处理
train_data=train_data.reshape((60000,28,28,1))
train_data=train_data.astype('float32')/255
test_data=test_data.reshape((10000,28,28,1))
test_data=test_data.astype('float32')/255
train_label=keras.utils.to_categorical(train_label)
test_label=keras.utils.to_categorical(test_label)

#分割出验证集**
val_data=train_data[50000:]
val_label=train_label[50000:]
train_data=train_data[:50000]
train_label=train_label[:50000]
print(train_data.shape)
#3、构建网络
model=keras.models.Sequential()
# #线性回归
# model.add(keras.layers.Dense(10,input_shape=(28*28,)))
# #逻辑回归
# model.add(keras.layers.Dense(10,activation='softmax',input_shape=(28*28,)))
# 卷积神经网络
model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D((2,2)))
model.add(keras.layers.Dropout(0.25))#降低过拟合
model.add(keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D((2,2)))
model.add(keras.layers.Dropout(0.25))#降低过拟合
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(81,activation='relu'))
model.add(keras.layers.Dropout(0.5))#降低过拟合
model.add(keras.layers.Dense(10,activation='softmax'))
print(model.summary())

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

#4、验证模型
history=model.fit(train_data,train_label,batch_size=500,epochs=6,validation_data=(val_data,val_label))

#5、作图调整拟合和预测
print(history.history.keys())
val_loss=history.history['val_loss']
val_acc=history.history['val_acc']
loss=history.history['loss']
acc=history.history['acc']

plt.plot(range(0,len(acc)),acc,'g',label='train acc')
plt.plot(range(0,len(val_acc)),val_acc,'r',label='val acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()

#测试集的损失和准确率
test_loss,test_acc=model.evaluate(test_data,test_label)
print('test_loss: ',test_loss,'test_acc: ',test_acc)