'''
@version: python3.6
@author: Administrator
@file: 从头开始训练一个卷积神经网络.py
@time: 2019/09/29
'''

#数据来自http://www.kaggle.com/c/dogs-vs-cats/data，该数据集包含25000张猫狗图片（各12500张）

import os,shutil
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pylab as plt
#1、基于原数据集中创建一个小型的新数据集，包含三个子集，每个类别1000张的训练集，每个类别500张的验证集，每个类别500张测试集

# #为新的数据集创建训练集文件夹，验证集文件夹，测试集文件夹
# orginal_dataset_dir="D:/用户目录/我的文档/keras实践/dog_cat_train/"
base_dir="D:/用户目录/我的文档/keras实践/dogs_cats/"
# os.makedirs(base_dir)
train_dir=os.path.join(base_dir,'train')
# os.makedirs(train_dir)
validation_dir=os.path.join(base_dir,'validation')
# os.makedirs(validation_dir)
test_dir=os.path.join(base_dir,'test')
# os.makedirs(test_dir)
# #创建训练集的猫和狗的文件夹
train_cats_dir=os.path.join(train_dir,'cats')
# os.makedirs(train_cats_dir)
train_dogs_dir=os.path.join(train_dir,'dogs')
# os.makedirs(train_dogs_dir)
# #创建验证集的猫和狗的文件夹
validation_cats_dir=os.path.join(validation_dir,'cats')
# os.makedirs(validation_cats_dir)
validation_dogs_dir=os.path.join(validation_dir,'dogs')
# os.makedirs(validation_dogs_dir)
# #创建测试集的猫和狗的文件夹
test_cats_dir=os.path.join(test_dir,'cats')
# os.makedirs(test_cats_dir)
test_dogs_dir=os.path.join(test_dir,'dogs')
# os.makedirs(test_dogs_dir)
# #从原数据集中找1000张猫图片复制粘贴到新的训练集的猫的路径下
# fnames=['cat.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src=os.path.join(orginal_dataset_dir,fname)
#     dst=os.path.join(train_cats_dir,fname)
#     shutil.copyfile(src,dst)
# #从原数据集中找500张猫图片复制粘贴到新的验证集的猫的路径下
# fnames=['cat.{}.jpg'.format(i) for i in range(1000,1500)]
# for fname in fnames:
#     src=os.path.join(orginal_dataset_dir,fname)
#     dst=os.path.join(validation_cats_dir,fname)
#     shutil.copyfile(src,dst)
# #从原数据集中找500张猫图片复制粘贴到新的测试集的猫的路径下
# fnames=['cat.{}.jpg'.format(i) for i in range(1500,2000)]
# for fname in fnames:
#     src=os.path.join(orginal_dataset_dir,fname)
#     dst=os.path.join(test_cats_dir,fname)
#     shutil.copyfile(src,dst)
#
#
# #从原数据集中找1000张狗图片复制粘贴到新的训练集的狗的路径下
# fnames=['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src=os.path.join(orginal_dataset_dir,fname)
#     dst=os.path.join(train_dogs_dir,fname)
#     shutil.copyfile(src,dst)
# #从原数据集中找500张狗图片复制粘贴到新的验证集的狗的路径下
# fnames=['dog.{}.jpg'.format(i) for i in range(1000,1500)]
# for fname in fnames:
#     src=os.path.join(orginal_dataset_dir,fname)
#     dst=os.path.join(validation_dogs_dir,fname)
#     shutil.copyfile(src,dst)
# #从原数据集中找500张狗图片复制粘贴到新的测试集的狗的路径下
# fnames=['dog.{}.jpg'.format(i) for i in range(1500,2000)]
# for fname in fnames:
#     src=os.path.join(orginal_dataset_dir,fname)
#     dst=os.path.join(test_dogs_dir,fname)
#     shutil.copyfile(src,dst)
#
#
# #查看新的数据集
# print('train_cat:',len(os.listdir(train_cats_dir)))
# print('val_cat:',len(os.listdir(validation_cats_dir)))
# print('test_cat:',len(os.listdir(test_cats_dir)))
# print('train_dog:',len(os.listdir(train_dogs_dir)))
# print('val_dog:',len(os.listdir(validation_dogs_dir)))
# print('test_dog:',len(os.listdir(test_dogs_dir)))

#2、构建网络
model=keras.models.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(keras.layers.MaxPool2D((2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D((2,2)))
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D((2,2)))
model.add(keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(keras.layers.MaxPool2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))#进一步降低过拟合
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))
model.summary()
#配置模型
model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])

#3.1、数据预处理（重点）：读取图片文件；将JPEG文件解码为RBG像素网格；将像素网格转化成浮点数张量；将像素值缩放到（0，1）之间，即标准化
# #在keras中提供这样的图像处理辅助工具，其中ImageDataGenerator可以快速创建python生成器，将图像文件转换为处理好的批量张量
# train_datagen=ImageDataGenerator(rescale=1./255)
# test_datagen=ImageDataGenerator(rescale=1./255)
# #图片大小调整为150x150，标签使用二进制表示
# train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
# validation_generator=test_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=20,class_mode='binary')
# #使用批量生成器拟合模型
# history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=20,validation_data=validation_generator,validation_steps=50)
# model.save('dogs_cats_small_1.h5')


#数据样本较少以及模型较复杂都会导致过拟合，所以为了降低过拟合，我们使用数据增强来增加样本，dropout或正则化来减少网络中的输入和参数

#3.2、数据增强。直观上就是将原图片进行旋转、平移，错切，缩放等变换来产生新的样本图片
#rotation_range是随机旋转的角度值，width_shift_range是水平上平移的范围，shear_range是随机错切变换的角度，zoom_range是随机缩放的范围,horizontal_flip是将一半图片水平翻转
train_datagen=ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
#注意验证集不参与训练，所以不需要进行数据增强
test_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=32,class_mode='binary')
validation_generator=test_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=32,class_mode='binary')
#训练数据并保存
history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=20,validation_data=validation_generator,validation_steps=50)
model.save('dogs_cats_small_2.h5')

#获取训练和验证中的数据，用来绘图
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epoch=range(1,len(acc)+1)

plt.plot(epoch,acc,'bo',label='training acc')
plt.plot(epoch,val_acc,'b-',label='validation acc')
plt.title("accuracy")
plt.legend()
plt.show()
plt.figure()
plt.plot(epoch,loss,'bo',label='training loss')
plt.plot(epoch,val_loss,'b-',label='validation loss')
plt.title("loss")
plt.legend()
plt.show()

'''
最终通过比较dogs_cats_small_1.h5模型和dogs_cats_small_2.h5模型的训练和验证的精度、损失曲线，发现精度会提高15%左右，
从而到达82%左右，还可以通过调节超参数和卷积层数和过滤器来达到更高精度，但是只靠自己从头开始训练自己的网络，再进一步提高
精度就较难，因此我们需要使用预训练的模型，该部分在下一个文件
'''
