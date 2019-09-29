'''
@version: python3.6
@author: Administrator
@file: kaggle_house_prices.py
@time: 2019/09/25
'''


#具体题目和讲解请见于该地址：http://zh.d2l.ai/chapter_deep-learning-basics/kaggle-house-price.html


from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
#1.获取数据并查看
train_data=pd.read_csv('D:/用户目录/我的文档/keras实践/house_prices/train.csv')#返回的是dataframe形式
test_data=pd.read_csv('D:/用户目录/我的文档/keras实践/house_prices/test.csv')
print(train_data.shape,test_data.shape)
# data=train_data.iloc[0:4,[0,1,2,3,-1]]
print(train_data.head())#输出前5行数据
#2、数据处理
#进行数据预处理的时候更加方便。等所有的需要的预处理进行完之后，我们再把他们分隔开。
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))#测试数据接在训练数据后面
numeric=all_features.dtypes[all_features.dtypes!='object'].index#取出dataframe所有非字符串列的下标

#连续数值的特征做标准化,缺省补0，数乘运算使用apply
all_features[numeric]=all_features[numeric].apply(lambda x:(x-x.mean())/x.std())
all_features[numeric]=all_features[numeric].fillna(0)

#get_dummies 是利用pandas实现one hot encode(特征数增加，即列数增加)
all_features=pd.get_dummies(all_features,dummy_na=True)
#将训练集和测试集分开，并且向量化
n_train=train_data.shape[0]
train_features=np.array(all_features[:n_train].values)
test_features=np.array(all_features[n_train:].values)
train_labels=np.array(train_data.SalePrice.values).reshape((-1,1))
print("处理后数据:",train_features.shape)
exam=train_features[0:4]
print(exam)

#3、构建网络
def rmse(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1))
def build_model():
    model=keras.models.Sequential()
    model.add(keras.layers.Dense(16,activation='relu',input_shape=(train_features.shape[1],)))
    model.add(keras.layers.Dense(16,activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam',loss=rmse,metrics=['mae'])
    return model

#4、k折交叉验证
k=5
num_val_sample=len(train_features)//k
num_epoch=200
train_mae_history=[]#记录训练和验证过程中所有的误差
val_mae_history=[]
print(num_val_sample)
# val_data=train_features[:num_val_sample]
# val_labels=train_labels[:num_val_sample]
for i in range(k):
    #获取验证集
    val_data=train_features[i*num_val_sample:(i+1)*num_val_sample]
    val_labels=train_labels[i*num_val_sample:(i+1)*num_val_sample]
    #剩下的部分作为训练集
    partial_train_data=np.concatenate([train_features[:i*num_val_sample],train_features[(i+1)*num_val_sample:]],axis=0)
    partial_train_labels=np.concatenate([train_labels[:i*num_val_sample],train_labels[(i+1)*num_val_sample:]],axis=0)
    #训练模型
    model=build_model()
    history=model.fit(partial_train_data,partial_train_labels,validation_data=(val_data,val_labels),epochs=num_epoch,batch_size=32,verbose=0)
    print(history.history.keys())
    #记录训练和验证过程中的误差
    val_mae=history.history['val_mean_absolute_error']
    val_mae_history.append(val_mae)
    train_mae = history.history['mean_absolute_error']
    train_mae_history.append(train_mae)
    print('%d-fold validation: avg train rmse %f, avg valid rmse %f'
          % (k,np.array(train_mae).mean(),np.array(val_mae).mean() ))

#用图像展现训练和验证的误差，以便查看是否过拟合，借此调整模型
val_average_mae=[np.mean([x[i] for x in val_mae_history]) for i in range(num_epoch)]
train_average_mae=[np.mean([x[i] for x in train_mae_history]) for i in range(num_epoch)]
plt.plot(range(1,len(val_average_mae)+1),val_average_mae,'b',label='val_mae')
plt.plot(range(1,len(train_average_mae)+1),train_average_mae,'r',label='train_mae')
plt.xlabel('epoch')
plt.ylabel('mae')
plt.show()
