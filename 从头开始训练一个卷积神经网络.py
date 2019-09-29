'''
@version: python3.6
@author: Administrator
@file: 从头开始训练一个卷积神经网络.py
@time: 2019/09/29
'''

#数据来自http://www.kaggle.com/c/dogs-vs-cats/data，该数据集包含25000张猫狗图片（各12500张）


#从原数据集中创建一个小型的新数据集，包含三个子集，每个类别1000张的训练集，每个类别500张的验证集，每个类别500张测试集
import os,shutil
#为新的数据集创建训练集文件夹，验证集文件夹，测试集文件夹
orginal_dataset_dir="D:/用户目录/我的文档/keras实践/dog_cat_train/"
base_dir="D:/用户目录/我的文档/keras实践/dogs_cats/"
os.makedirs(base_dir)
train_dir=os.path.join(base_dir,'train')
os.makedirs(train_dir)
validation_dir=os.path.join(base_dir,'validation')
os.makedirs(validation_dir)
test_dir=os.path.join(base_dir,'test')
os.makedirs(test_dir)
#创建训练集的猫和狗的文件夹
train_cats_dir=os.path.join(train_dir,'cats')
os.makedirs(train_cats_dir)
train_dogs_dir=os.path.join(train_dir,'dogs')
os.makedirs(train_dogs_dir)
#创建验证集的猫和狗的文件夹
validation_cats_dir=os.path.join(validation_dir,'cats')
os.makedirs(validation_cats_dir)
validation_dogs_dir=os.path.join(validation_dir,'dogs')
os.makedirs(validation_dogs_dir)
#创建测试集的猫和狗的文件夹
test_cats_dir=os.path.join(test_dir,'cats')
os.makedirs(test_cats_dir)
test_dogs_dir=os.path.join(test_dir,'dogs')
os.makedirs(test_dogs_dir)
#从原数据集中找1000张猫图片复制粘贴到新的训练集的猫的路径下
fnames=['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src=os.path.join(orginal_dataset_dir,fname)
    dst=os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)
#从原数据集中找500张猫图片复制粘贴到新的验证集的猫的路径下
fnames=['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src=os.path.join(orginal_dataset_dir,fname)
    dst=os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src,dst)
#从原数据集中找500张猫图片复制粘贴到新的测试集的猫的路径下
fnames=['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src=os.path.join(orginal_dataset_dir,fname)
    dst=os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)


#从原数据集中找1000张狗图片复制粘贴到新的训练集的狗的路径下
fnames=['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src=os.path.join(orginal_dataset_dir,fname)
    dst=os.path.join(train_dogs_dir,fname)
    shutil.copyfile(src,dst)
#从原数据集中找500张狗图片复制粘贴到新的验证集的狗的路径下
fnames=['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src=os.path.join(orginal_dataset_dir,fname)
    dst=os.path.join(validation_dogs_dir,fname)
    shutil.copyfile(src,dst)
#从原数据集中找500张狗图片复制粘贴到新的测试集的狗的路径下
fnames=['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src=os.path.join(orginal_dataset_dir,fname)
    dst=os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src,dst)


#查看新的数据集
print('train_cat:',len(os.listdir(train_cats_dir)))
print('val_cat:',len(os.listdir(validation_cats_dir)))
print('test_cat:',len(os.listdir(test_cats_dir)))
print('train_dog:',len(os.listdir(train_dogs_dir)))
print('val_dog:',len(os.listdir(validation_dogs_dir)))
print('test_dog:',len(os.listdir(test_dogs_dir)))


