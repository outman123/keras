'''
@version: python3.6
@author: Administrator
@file: 使用预训练的卷积神经网络.py
@time: 2019/10/14
'''

from tensorflow import keras
import os
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pylab as plt
'''
预训练网络是一个已经在大型数据集上训练并保存好的网络，如果原始数据集足够大且通用，那么预训练网络学到的特征结构
可以作为视觉问题的通用模型。例如由于特征的在不同问题上的可移植性，在imageNet数据集（140万样本，1000个种类）上
训练好的大型神经网络在猫狗分类问题中也有良好表现。

可以使用VGG,ResNet,MobileNet,Inception,Xception等，这里使用VGG16架构。

使用预训练模型有两种方法：
1、特征提取：取出之前训练好的网络的卷积基，在上面运行新样本数据，然后再输出上面训练一个新的分类器
2、模型微调

'''
#特征提取，一般仅重复使用卷积神经网络层中的卷积基，而避免重复使用密集连接分类器（问题类别不同；丢失空间位置信息）
#其中又可以分成2.1的不使用数据增强的快速特征提取：每个输入图像只运行一次卷积基，计算力小，但过拟合严重，准确率有限
#             2.2的使用数据增强的特征提取：主要是扩展conv_base，在输入数据上端到端的运行模型，计算代价很高
#1、将VGG16卷积基实例化
#weights指定权重检查点，include_top表示是否包括密集连接分类器，input_shape表示输入的图像张量形状
conv_base=keras.applications.VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
print(conv_base.summary())


#2.1、使用预训练的卷积基提取特征：利用实例预测得到一系列特征，再将这些特征经过自己定义的分类器
#找到对应文件目录下为了加载不同样本
base_dir="D:/用户目录/我的文档/keras实践/dogs_cats/"
train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir,'test')
datagen=ImageDataGenerator(1./255)
#定义函数来返回特征和标签，参数是文件路径和样本数量
def extract_feature(directory,count):
    feature=np.zeros(shape=(count,4,4,512))#存储所有的输入图片的特征
    labels=np.zeros(shape=(count))#存储所有的输入图片
    batch_size = 20#每一个批次包含20个样本
    generator=datagen.flow_from_directory(directory,target_size=(150,150),batch_size=batch_size,class_mode='binary')
    i=0#记录当期batch的序号
    for input_batch,label_batch in generator:#从生成器中取出预处理后的每个批次图片样本和标签，经过VGG16卷积基获得特征存储到feature中
        feature_batch=conv_base.predict(input_batch)
        feature[i*batch_size:(i+1)*batch_size]=feature_batch
        labels[i*batch_size:(i+1)*batch_size]=label_batch
        i=i+1
        if i*batch_size==2000:
            break
    return feature,labels
#将训练集，验证集和测试集的样本数据都提取特征，作为后面自己构建的密集连接分类网络的输入
train_feature,train_label=extract_feature(train_dir,2000)
validation_feature,validation_label=extract_feature(validation_dir,1000)
test_feature,test_label=extract_feature(train_dir,1000)
print(train_feature.shape,train_label.shape)
#将特征的形状展平为4*4*512，再输入到密集层中
train_feature.reshape(shape=(2000,4*4*512))
validation_feature.reshape(shape=(1000,4*4*512))
test_feature.reshape(shape=(1000,4*4*512))

#构建并训练密集连接分类器
model=keras.models.Sequential()
model.add(keras.layers.Dense(256,activation='relu',input_dim=4*4*512))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1,activation='sigmoid'))
#编译自定义分类器
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
#训练自定义分类器
history=model.fit(train_feature,train_label,batch_size=20,epochs=30,validation_data=(validation_feature,validation_label))

#绘制准确率曲线
print(history.history.keys())
val_loss=history.history['val_loss']
val_acc=history.history['val_acc']
loss=history.history['loss']
acc=history.history['acc']

plt.plot(range(1,len(acc)+1),acc,'g',label='train acc')
plt.plot(range(1,len(val_acc)+1),val_acc,'r',label='val acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()

# #2.2使用数据增强的特征提取：将实例作为自己定义的序列网络的一部分，后面跟上一个全连接层，图片经过数据增强后输入自己定义的网络
#
# #数据处理：数据增强
# train_datagen=ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,
#                                  height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,
#                                  horizontal_flip=True,fill_mode='nearest')
# #验证和测试集不需要数据增强
# validation_datagen=ImageDataGenerator(rescale=1./255)
# #生成预处理后样本图片
# train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
# validation_generator=validation_datagen.flow_from_directory(validation_dir,target_size=(150,150),batch_size=20,class_mode='binary')
#
# #构建网络
# #在卷积基上添加一个密集连接分类器
# model=keras.models.Sequential()
# model.add(conv_base)
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(256,activation='relu'))
# model.add(keras.layers.Dense(1,activation='sigmoid'))
# print(model.summary())
#
# #注意在编译和训练模型前，必须要冻结卷积基，否则卷积基里面训练好的权重会被改变，达不到原来的表示效果
# conv_base.trainable=False#此时只有conv_base之后的Dense层的权重会被训练
# #编译模型
# model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
# #训练模型
# history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=30,validation_data=validation_generator,validation_steps=50)
# #准确率能达到96%左右，比不使用数据增强的特征提取提高6%左右


#在使用数据增强的特征提取上训练好的模型的基础上，我们还可以继续使用模型微调的方法来进一步优化模型，提高准确率
'''
微调模型:对于上面的方式，可以解冻部分顶层的几层网络，将这几层和自己定义的新层一起训练
该例题的步骤：  在已经训练好的基网络上添加自定义网络；
              冻结基网络；
              训练所添加的部分；
              解冻基网络的部分层；
              联合训练解冻的这些层新添加的层；
              
前三步在上面的2.2数据增强的特征提取部分已完成，接下来实现后两步
'''
#




