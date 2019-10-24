'''
@version: python3.6
@author: Administrator
@file: CAPTCHA验证码识别.py
@time: 2019/10/21
'''

'''

目的：本实验主要是使用CNN来训练识别含有四个数字的验证码（属于分类问题），当然字符类型和数量的改动都可以自己在代码中设置。
数据集：captcha是用Python写的生成验证码（用于区分人和机器的主要方法）的库，支持图片验证码和语音验证码。
       我们使用captcha库来生成数据，但是数据的使用和加载主要有两种方式：
       1、验证码图片写在本地文件夹下，然后再来读取图片数据。需要自己提前生成图片保存好，且数据量会有所限制，但运行较快，多次调试使用
       2、自动生成在内存中。需要自定义生成器，训练、验证和测试都使用keras的生成器方式，数据无限但运行较慢
网络架构：卷积基有三个部分，每个部分包含两个卷积层，一个池化层；
         通过卷积基获得特征图后进行flatten扁平化和dropout降低过拟合；
         dropout后进入平行的四个分支密集层，每个密集层识别出图中一个字符的类别。
         最终将这四个密集层的输出合并在一起作为整个网络的输出。
         
最终准确率可以达到90%以上，当然除了使用卷积神经网络进行训练，使用LSTM可以获得更好的效果，之后会进行相应的实现
'''
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
import random
import string
import os
import pickle
from keras_preprocessing import image
from tensorflow import keras
import pydot
import graphviz
import pydot_ng
#选取字符串中的数字作为验证码内
number=string.digits
number_class=len(number)
number_len=4



# #1、产生图片写在本地进行训练


# #定义函数生成字符对应的验证码图片
# def gen_captcha_text_and_image():
#     image=ImageCaptcha(width=170,height=80)
#     #随机生成4数字
#     captcha_text=[random.choice(number) for j in range(number_len)]
#     captcha_text=''.join(captcha_text)
#     #生成验证码
#     captcha=image.generate(captcha_text)
#     #写在本地作为数据集
#     # image.write(captcha_text,'captcha/'+captcha_text+'.jpg')
#     captcha_image=Image.open(captcha)
#     #将图片转化成数组
#     captcha_image=np.array(captcha_image)
#     return captcha_text,captcha_image
# # 生成图片
# # for i in range(10000):
# #     print(i)
# #     gen_captcha_text_and_image()
#
# #字符串向one-hot编码转化
# def captcha_to_vec(captche):
#     vector=np.zeros(shape=(number_len*number_class))
#     for i,ch in enumerate(captche):
#         index=i*10+number.find(ch)
#         vector[index]=1#对应0-9下标设为1
#     return vector
# #one-hot编码向验证码字符串转化
# def vec_to_captch(vector):
#     captcha=[]
#     vector[vector<0.5]=0#把概率小于0.5的标记为错误
#     char_pos=vector.nonzero()[0]
#     for i,ch in enumerate(char_pos):
#         captcha.append(number[ch%number_class])
#     return ''.join(captcha)
#
#
# #获取之前生成的图片数据
# image_list=[]
# path='D:/用户目录/我的文档/keras实践/captcha/'
# for item in os.listdir(path):
#     image_list.append(item)
#
# #将样本和标签张量化
# X=np.zeros((len(image_list),80,170,3),dtype=np.uint8)
# y=np.zeros((len(image_list),number_class*number_len),dtype=np.uint8)
# for i,img in enumerate(image_list):
#     raw_img=image.load_img(path+img,target_size=(80,170))
#     X[i]=image.img_to_array(raw_img)
#     y[i]=captcha_to_vec(img.split('.')[0])#将标签进行one-hot编码
# #缩放以加快收敛
# X=X.astype('float32')/255
# #查看张量后数据阶数、形状和数据类型
# print(X.shape,y.shape)
#
#
# #构建网络
# input_tensor=keras.Input(shape=(80,170,3))
# x=input_tensor
# for i in range(3):
#     x=keras.layers.Conv2D(64*(i+1),(5,5),padding='same',activation='relu')(x)
#     x=keras.layers.Conv2D(64*(i+1),(5,5),padding='same',activation='relu')(x)
#     x=keras.layers.MaxPool2D((2,2))(x)
# #提取出特征后扁平化
# x=keras.layers.Flatten()(x)
# x=keras.layers.Dropout(0.25)(x)
# #连接4个分类器，每个分类器有10个节点，分别输出10个字符的概率
# x=[keras.layers.Dense(number_class,activation='softmax')(x) for i in range(number_len)]
# #将四个分类器的输出拼在一起
# output=keras.layers.concatenate(x)
# #构建模型
# model=keras.Model(inputs=input_tensor,outputs=output)
#
# #可以生成网络架构图，但要配置安装pydot,graphviz，pydot_ng库
# keras.utils.plot_model(model,show_shapes=True,to_file='model.png')
#
#
# opt=keras.optimizers.Adadelta(lr=0.1)
# model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
# #每次epoch都保存一下权重，用于继续训练
# checkpointer = keras.callbacks.ModelCheckpoint(filepath="output/weights.{epoch:02d}--{val_loss:.2f}-{val_acc:.4f}.hdf5",
#                                verbose=2, save_weights_only=True)
# #开始训练，validation_split代表10%的数据不参与训练，用于做验证急
# #我之前训练了50个epochs以上，这里根据自己的情况进行选择。如果输出的val_acc已经达到你满意的数值，可以终止训练
# model.fit(X, y,batch_size=256,epochs=100,callbacks=[checkpointer], validation_split=0.1)
# #保存训练好的模型
# model.save_weights('captch_model_weight-1.h5')
# model.save('captch_model-1.h5')
#
# #验证模型
# def testCaptcha(index):
#     raw_img = X[index]
#     true_label = y[index]
#     #将获取的验证码图片张量后用模型预测其四个数字
#     X_test = np.zeros((1, 80, 170, 3), dtype=np.float32)
#     X_test[0] = image.img_to_array(raw_img)
#     result = model.predict(X_test)
#     #将预测结果从one-hot编码转化为字符串
#     vex_test = vec_to_captch(result[0])
#     true_test = vec_to_captch(true_label)
#     plt.imshow(raw_img)
#     plt.show()
#     print("原始：", true_test, "预测", vex_test)
#
# # 选3张验证码进行验证
# for i in range(3):
#     testCaptcha(i)




#分割线————————————————————————————————————————————————————————————————————————————————————————————————————————————





# 2、使用生成器产生数据来训练

#先定义一个生成器每次生成32个样本和标签
def gen(batch_size = 32):
    X = np.zeros((batch_size,80,170,3),dtype=np.uint8)#定义样本张量为（32,80,170,3）
    y = np.zeros((batch_size,number_len*number_class),dtype=np.uint8) #定义标签张量为（32,40）
    generator = ImageCaptcha(height=80,width=170)
    while True:#生成器不断生成样本和标签
        for i in range(batch_size):
            random_str = ''.join([random.choice(number) for j in range(number_len)])#随机生成4个数字
            X[i] = np.array(generator.generate_image(random_str))#生成验证码图片
            for j, ch in enumerate(random_str):#one-hot编码
                y[i,j*number_class+number.find(ch)] = 1
        yield X,y#返回迭代器内容

def decode(vector):#将onehot解码为原字符串
    captcha = []
    vector[vector<0.5]=0#把概率小于0.5的标记为错误
    char_pos=vector.nonzero()[0]
    for i,ch in enumerate(char_pos):
        captcha.append(number[ch%number_class])
    return ''.join(captcha)


#api函数式构建网络
input_tensor=keras.Input(shape=(80,170,3))
x=input_tensor
for i in range(3):
    x=keras.layers.Conv2D(64*(i+1),(5,5),padding='same',activation='relu')(x)
    x=keras.layers.Conv2D(64*(i+1),(5,5),padding='same',activation='relu')(x)
    x=keras.layers.MaxPool2D((2,2))(x)
#提取出特征后扁平化
x=keras.layers.Flatten()(x)
x=keras.layers.Dropout(0.25)(x)
#连接4个分类器，每个分类器有10个节点，分别输出10个字符的概率
x=[keras.layers.Dense(number_class,activation='softmax')(x) for i in range(number_len)]
#将四个分类器的输出拼在一起
output=keras.layers.concatenate(x)
#构建模型
model=keras.Model(inputs=input_tensor,outputs=output)
model.summary()
#生成网络结构图用来查看
keras.utils.plot_model(model,show_shapes=True,to_file='model.png')

#编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
#训练模型
checkpointer = keras.callbacks.ModelCheckpoint(filepath="output/weights.{epoch:02d}--{val_loss:.2f}-{val_acc:.4f}.hdf5",
                               verbose=2, save_weights_only=True)
history=model.fit_generator(gen(), steps_per_epoch=10000, epochs=10,callbacks=[checkpointer],validation_data=gen(), validation_steps=1000)
model.save('captch_model-1.h5')


#验证
X, y = next(gen(1))
y_pred = model.predict(X)
plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
plt.imshow(X[0], cmap='gray')