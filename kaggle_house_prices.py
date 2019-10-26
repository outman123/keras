'''
@version: python3.6
@author: Administrator
@file: kaggle_house_prices.py
@time: 2019/09/25
'''


#具体题目和讲解请见于该地址：http://zh.d2l.ai/chapter_deep-learning-basics/kaggle-house-price.html


import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import skew
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#1.获取数据并查看
train_data=pd.read_csv('D:/用户目录/我的文档/keras实践/house_prices/train.csv')#返回的是dataframe类型
test_data=pd.read_csv('D:/用户目录/我的文档/keras实践/house_prices/test.csv')
print(train_data.shape,test_data.shape)
# data=train_data.iloc[0:4,[0,1,2,3,-1]]
print(train_data.head())#输出前5行数据
print(test_data.head())
#2、去除异常数据

# 使用下列两种方式都可以查看与SalePrice最相关的10个属性
# #作图来显示相关性
# corrmat=train_data.corr()
# plt.figure(figsize=(12,9))
# cols=corrmat.nlargest(10,'SalePrice')['SalePrice'].index
# cm=np.corrcoef(train_data[cols].values.T)
# sns.set(font_scale=1.25)
# hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f', annot_kws={'size': 10},xticklabels=cols.values,yticklabels=cols.values)
# plt.show()
# 或者不作图，直接输出相关性大于0.5的属性数据
Corr=train_data.corr()
print(Corr[Corr['SalePrice']>0.5])

#已经知道有哪些特征与SalePrice比较相关，加下来绘制散点图来具体看每个属性与SalePrice的关系
# sns.pairplot(x_vars=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt'],y_vars=['SalePrice'],data=train_data,dropna=True)
# plt.show()

#根据散点图的显示来去除异常值
train_data.drop(train_data[(train_data['OverallQual']<5) & (train_data['SalePrice']>200000)].index,inplace=True)
train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<200000)].index,inplace=True)
train_data.drop(train_data[(train_data['YearBuilt']<1900) & (train_data['SalePrice']>400000)].index,inplace=True)
train_data.drop(train_data[(train_data['FullBath']<1) & (train_data['SalePrice']>300000)].index,inplace=True)
train_data.drop(train_data[(train_data['TotalBsmtSF']>6000) & (train_data['SalePrice']<200000)].index,inplace=True)
train_data.reset_index(drop=True, inplace=True)
print(train_data.shape)


#3、缺省值处理
'''
对于缺失数据的处理，通常会有以下几种做法：
如果缺失的数据过多，可以考虑删除该列特征
用平均值、中值、分位数、众数、随机值等替代。但是效果一般，因为等于人为增加了噪声
用插值法进行拟合
用其他变量做预测模型来算出缺失变量。效果比方法1略好。有一个根本缺陷，如果其他变量和缺失变量无关，则预测的结果无意义
最精确的做法，把变量映射到高维空间。比如性别，有男、女、缺失三种情况，则映射成3个变量：是否男、是否女、是否缺失。缺点就是计算量会加大
'''

#将数据集连接组合在一起。等所有的需要的预处理进行完之后，再把他们分隔开。
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))#测试数据接在训练数据后面。训练集和测试集第一列是序号，无用舍去，测试集没有标签
numeric=all_features.dtypes[all_features.dtypes!='object'].index#取出dataframe所有非字符串类型的列名（特征名）
print(all_features.shape)
print(all_features.head())
# print(numeric)

#倒叙统计每种属性的缺省总数和占总数比重
count=all_features.isnull().sum().sort_values(ascending=False)
ratio=count/len(all_features)
nulldata=pd.concat([count,ratio],axis=1,keys=['count','ratio'])
print(nulldata)

# 有的房屋确实不存在某种的特征，所以类别特征的缺失值以一种新类别插补，数值特征以0值插补。
# 剩余的那一部分缺失的特征值采用众数插补。

# # 查看缺失值情况
def missing_values(alldata):
    alldata_na = pd.DataFrame(alldata.isnull().sum(), columns={'missingNum'})
    alldata_na['missingRatio'] = alldata_na['missingNum'] / len(alldata) * 100
    alldata_na['existNum'] = len(alldata) - alldata_na['missingNum']

    alldata_na['train_notna'] = len(train_data) - train_data.isnull().sum()
    alldata_na['test_notna'] = alldata_na['existNum'] - alldata_na['train_notna']
    alldata_na['dtype'] = alldata.dtypes

    alldata_na = alldata_na[alldata_na['missingNum'] > 0].reset_index().sort_values(by=['missingNum', 'index'],
                                                                                    ascending=[False, True])
    alldata_na.set_index('index', inplace=True)
    return alldata_na

all_features_na = missing_values(all_features)


# poolqcna = all_features[(all_features['PoolQC'].isnull()) & (all_features['PoolArea'] != 0)][['PoolQC', 'PoolArea']]
# areamean = all_features.groupby('PoolQC')['PoolArea'].mean()
# for i in poolqcna.index:
#     v = all_features.loc[i, ['PoolArea']].values
#     print(type(np.abs(v - areamean)))
#     all_features.loc[i, ['PoolQC']] = np.abs(v - areamean).astype('float64').argmin()

all_features['PoolQC'] = all_features["PoolQC"].fillna("None")
all_features['PoolArea'] = all_features["PoolArea"].fillna(0)

all_features[['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']] = all_features[
    ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']].fillna('None')
all_features[['GarageCars', 'GarageArea']] = all_features[['GarageCars', 'GarageArea']].fillna(0)
all_features['Electrical'] = all_features['Electrical'].fillna(all_features['Electrical'].mode()[0])

# 注意此处'GarageYrBlt'尚未填充


a = pd.Series(all_features.columns)
BsmtList = a[a.str.contains('Bsmt')].values

condition = (all_features['BsmtExposure'].isnull()) & (all_features['BsmtCond'].notnull())  # 3个
all_features.ix[(condition), 'BsmtExposure'] = all_features['BsmtExposure'].mode()[0]

condition1 = (all_features['BsmtCond'].isnull()) & (all_features['BsmtExposure'].notnull())  # 3个
all_features.ix[(condition1), 'BsmtCond'] = all_features.ix[(condition1), 'BsmtQual']

condition2 = (all_features['BsmtQual'].isnull()) & (all_features['BsmtExposure'].notnull())  # 2个
all_features.ix[(condition2), 'BsmtQual'] = all_features.ix[(condition2), 'BsmtCond']

# 对于BsmtFinType1和BsmtFinType2
condition3 = (all_features['BsmtFinType1'].notnull()) & (all_features['BsmtFinType2'].isnull())
all_features.ix[condition3, 'BsmtFinType2'] = 'Unf'

allBsmtNa = all_features_na.ix[BsmtList, :]
allBsmtNa_obj = allBsmtNa[allBsmtNa['dtype'] == 'object'].index
allBsmtNa_flo = allBsmtNa[allBsmtNa['dtype'] != 'object'].index
all_features[allBsmtNa_obj] = all_features[allBsmtNa_obj].fillna('None')
all_features[allBsmtNa_flo] = all_features[allBsmtNa_flo].fillna(0)

MasVnrM = all_features.groupby('MasVnrType')['MasVnrArea'].median()
mtypena = all_features[(all_features['MasVnrType'].isnull()) & (all_features['MasVnrArea'].notnull())][['MasVnrType', 'MasVnrArea']]
# for i in mtypena.index:
#     v = all_features.loc[i, ['MasVnrArea']].values
#     all_features.loc[i, ['MasVnrType']] = np.abs(v - MasVnrM).astype('float64').argmin()

all_features['MasVnrType'] = all_features["MasVnrType"].fillna("None")
all_features['MasVnrArea'] = all_features["MasVnrArea"].fillna(0)


all_features["MSZoning"] = all_features.groupby("MSSubClass")["MSZoning"].transform(lambda x: x.fillna(x.mode()[0]))

#使用多项式拟合填充

x = all_features.loc[all_features["LotFrontage"].notnull(), "LotArea"]
y = all_features.loc[all_features["LotFrontage"].notnull(), "LotFrontage"]
t = (x <= 25000) & (y <= 150)
p = np.polyfit(x[t], y[t], 1)
all_features.loc[all_features['LotFrontage'].isnull(), 'LotFrontage'] = \
    np.polyval(p, all_features.loc[all_features['LotFrontage'].isnull(), 'LotArea'])

all_features['KitchenQual'] = all_features['KitchenQual'].fillna(all_features['KitchenQual'].mode()[0])  # 用众数填充
all_features['Exterior1st'] = all_features['Exterior1st'].fillna(all_features['Exterior1st'].mode()[0])
all_features['Exterior2nd'] = all_features['Exterior2nd'].fillna(all_features['Exterior2nd'].mode()[0])
all_features["Functional"] = all_features["Functional"].fillna(all_features['Functional'].mode()[0])
all_features["SaleType"] = all_features["SaleType"].fillna(all_features['SaleType'].mode()[0])
all_features["Utilities"] = all_features["Utilities"].fillna(all_features['Utilities'].mode()[0])

all_features[["Fence", "MiscFeature"]] = all_features[["Fence", "MiscFeature"]].fillna('None')
all_features['FireplaceQu'] = all_features['FireplaceQu'].fillna('None')
all_features['Alley'] = all_features['Alley'].fillna('None')


# all_features.isnull().sum()[all_features.isnull().sum()>0]
#至此， 还有一个属性未填充 # GarageYrBlt    159

year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))
# 将年份对应映射
all_features.GarageYrBlt = all_features.GarageYrBlt.map(year_map)
all_features['GarageYrBlt']= all_features['GarageYrBlt'].fillna('None')# 必须 离散化之后再对应映射

# 处理序列型标称数据
ordinalList = ['ExterQual', 'ExterCond', 'GarageQual', 'GarageCond','PoolQC',\
              'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtQual','BsmtCond']
ordinalmap = {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'None': 0}
for c in ordinalList:
    all_features[c] = all_features[c].map(ordinalmap)
#其他类别标签化
all_features['BsmtExposure'] = all_features['BsmtExposure'].map({'None':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4})
all_features['BsmtFinType1'] = all_features['BsmtFinType1'].map({'None':0, 'Unf':1, 'LwQ':2,'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
all_features['BsmtFinType2'] = all_features['BsmtFinType2'].map({'None':0, 'Unf':1, 'LwQ':2,'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
all_features['Functional'] = all_features['Functional'].map({'Maj2':1, 'Sev':2, 'Min2':3, 'Min1':4, 'Maj1':5, 'Mod':6, 'Typ':7})
all_features['GarageFinish'] = all_features['GarageFinish'].map({'None':0, 'Unf':1, 'RFn':2, 'Fin':3})
all_features['Fence'] = all_features['Fence'].map({'MnWw':0, 'GdWo':1, 'MnPrv':2, 'GdPrv':3, 'None':4})
#部分属性创建二值新属性
MasVnrType_Any = all_features.MasVnrType.replace({'BrkCmn': 1,'BrkFace': 1,'CBlock': 1,'Stone': 1,'None': 0})
MasVnrType_Any.name = 'MasVnrType_Any' #修改该series的列名
SaleCondition_PriceDown = all_features.SaleCondition.replace({'Abnorml': 1,'Alloca': 1,'AdjLand': 1,'Family': 1,'Normal': 0,'Partial': 0})
SaleCondition_PriceDown.name = 'SaleCondition_PriceDown' #修改该series的列名

all_features = all_features.replace({'CentralAir': {'Y': 1,'N': 0}})
all_features = all_features.replace({'PavedDrive': {'Y': 1,'P': 0,'N': 0}})
newer_dwelling = all_features['MSSubClass'].map({20: 1,30: 0,40: 0,45: 0,50: 0,60: 1,70: 0,75: 0,80: 0,85: 0,90: 0,120: 1,150: 0,160: 0,180: 0,190: 0})
newer_dwelling.name= 'newer_dwelling' #修改该series的列名
all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)

# Neighborhood_Good = pd.DataFrame(np.zeros((all_features.shape[0],1)), columns=['Neighborhood_Good'])
# Neighborhood_Good[all_features.Neighborhood=='NridgHt'] = 1
# Neighborhood_Good[all_features.Neighborhood=='Crawfor'] = 1
# Neighborhood_Good[all_features.Neighborhood=='StoneBr'] = 1
# Neighborhood_Good[all_features.Neighborhood=='Somerst'] = 1
# Neighborhood_Good[all_features.Neighborhood=='NoRidge'] = 1
# # Neighborhood_Good = (alldata['Neighborhood'].isin(['StoneBr','NoRidge','NridgHt','Timber','Somerst']))*1 #(效果没有上面好)
# Neighborhood_Good.name='Neighborhood_Good'# 将该变量加入

season = (all_features['MoSold'].isin([5,6,7]))*1 #(@@@@@)
season.name='season'
all_features['MoSold'] = all_features['MoSold'].apply(str)

# 处理OverallQual：将该属性分成两个子属性，以5为分界线，大于5及小于5的再分别以序列
overall_poor_qu = all_features.OverallQual.copy()  # Series类型
overall_poor_qu = 5 - overall_poor_qu
overall_poor_qu[overall_poor_qu < 0] = 0
overall_poor_qu.name = 'overall_poor_qu'
overall_good_qu = all_features.OverallQual.copy()
overall_good_qu = overall_good_qu - 5
overall_good_qu[overall_good_qu < 0] = 0
overall_good_qu.name = 'overall_good_qu'

# 处理OverallCond ：将该属性分成两个子属性，以5为分界线，大于5及小于5的再分别以序列
overall_poor_cond = all_features.OverallCond.copy()  # Series类型
overall_poor_cond = 5 - overall_poor_cond
overall_poor_cond[overall_poor_cond < 0] = 0
overall_poor_cond.name = 'overall_poor_cond'
overall_good_cond = all_features.OverallCond.copy()
overall_good_cond = overall_good_cond - 5
overall_good_cond[overall_good_cond < 0] = 0
overall_good_cond.name = 'overall_good_cond'

# 处理ExterQual：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别以序列
exter_poor_qu = all_features.ExterQual.copy()
exter_poor_qu[exter_poor_qu < 3] = 1
exter_poor_qu[exter_poor_qu >= 3] = 0
exter_poor_qu.name = 'exter_poor_qu'
exter_good_qu = all_features.ExterQual.copy()
exter_good_qu[exter_good_qu <= 3] = 0
exter_good_qu[exter_good_qu > 3] = 1
exter_good_qu.name = 'exter_good_qu'

# 处理ExterCond：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别以序列
exter_poor_cond = all_features.ExterCond.copy()
exter_poor_cond[exter_poor_cond < 3] = 1
exter_poor_cond[exter_poor_cond >= 3] = 0
exter_poor_cond.name = 'exter_poor_cond'
exter_good_cond = all_features.ExterCond.copy()
exter_good_cond[exter_good_cond <= 3] = 0
exter_good_cond[exter_good_cond > 3] = 1
exter_good_cond.name = 'exter_good_cond'

# 处理BsmtCond：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别以序列
bsmt_poor_cond = all_features.BsmtCond.copy()
bsmt_poor_cond[bsmt_poor_cond < 3] = 1
bsmt_poor_cond[bsmt_poor_cond >= 3] = 0
bsmt_poor_cond.name = 'bsmt_poor_cond'
bsmt_good_cond = all_features.BsmtCond.copy()
bsmt_good_cond[bsmt_good_cond <= 3] = 0
bsmt_good_cond[bsmt_good_cond > 3] = 1
bsmt_good_cond.name = 'bsmt_good_cond'

# 处理GarageQual：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别以序列
garage_poor_qu = all_features.GarageQual.copy()
garage_poor_qu[garage_poor_qu < 3] = 1
garage_poor_qu[garage_poor_qu >= 3] = 0
garage_poor_qu.name = 'garage_poor_qu'
garage_good_qu = all_features.GarageQual.copy()
garage_good_qu[garage_good_qu <= 3] = 0
garage_good_qu[garage_good_qu > 3] = 1
garage_good_qu.name = 'garage_good_qu'

# 处理GarageCond：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别以序列
garage_poor_cond = all_features.GarageCond.copy()
garage_poor_cond[garage_poor_cond < 3] = 1
garage_poor_cond[garage_poor_cond >= 3] = 0
garage_poor_cond.name = 'garage_poor_cond'
garage_good_cond = all_features.GarageCond.copy()
garage_good_cond[garage_good_cond <= 3] = 0
garage_good_cond[garage_good_cond > 3] = 1
garage_good_cond.name = 'garage_good_cond'

# 处理KitchenQual：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别以序列
kitchen_poor_qu = all_features.KitchenQual.copy()
kitchen_poor_qu[kitchen_poor_qu < 3] = 1
kitchen_poor_qu[kitchen_poor_qu >= 3] = 0
kitchen_poor_qu.name = 'kitchen_poor_qu'
kitchen_good_qu = all_features.KitchenQual.copy()
kitchen_good_qu[kitchen_good_qu <= 3] = 0
kitchen_good_qu[kitchen_good_qu > 3] = 1
kitchen_good_qu.name = 'kitchen_good_qu'
#将构造的属性合并
qu_list = pd.concat((overall_poor_qu, overall_good_qu, overall_poor_cond, overall_good_cond, exter_poor_qu,
                     exter_good_qu, exter_poor_cond, exter_good_cond, bsmt_poor_cond, bsmt_good_cond, garage_poor_qu,
                     garage_good_qu, garage_poor_cond, garage_good_cond, kitchen_poor_qu, kitchen_good_qu), axis=1)
#对与时间相关属性处理
Xremoded = (all_features['YearBuilt'] != all_features['YearRemodAdd']) * 1  # (@@@@@)
Xrecentremoded = (all_features['YearRemodAdd'] >= all_features['YrSold']) * 1  # (@@@@@)
XnewHouse = (all_features['YearBuilt'] >= all_features['YrSold']) * 1  # (@@@@@)
XHouseAge = 2010 - all_features['YearBuilt']
XTimeSinceSold = 2010 - all_features['YrSold']
XYearSinceRemodel = all_features['YrSold'] - all_features['YearRemodAdd']

Xremoded.name = 'Xremoded'
Xrecentremoded.name = 'Xrecentremoded'
XnewHouse.name = 'XnewHouse'
XTimeSinceSold.name = 'XTimeSinceSold'
XYearSinceRemodel.name = 'XYearSinceRemodel'
XHouseAge.name = 'XHouseAge'

year_list = pd.concat((Xremoded, Xrecentremoded, XnewHouse, XHouseAge, XTimeSinceSold, XYearSinceRemodel), axis=1)
#构造新属性
from sklearn.svm import SVC

svm = SVC(C=100, gamma=0.0001, kernel='rbf')

pc = pd.Series(np.zeros(train_data.shape[0]))
pc[:] = 'pc1'
pc[train_data.SalePrice >= 150000] = 'pc2'
pc[train_data.SalePrice >= 220000] = 'pc3'
columns_for_pc = ['Exterior1st', 'Exterior2nd', 'RoofMatl', 'Condition1', 'Condition2', 'BldgType']
X_t = pd.get_dummies(train_data.loc[:, columns_for_pc], sparse=True)
svm.fit(X_t, pc)  # 训练
p = train_data.SalePrice / 100000

price_category = pd.DataFrame(np.zeros((all_features.shape[0], 1)), columns=['pc'])
X_t = pd.get_dummies(all_features.loc[:, columns_for_pc], sparse=True)
pc_pred = svm.predict(X_t)  # 预测

price_category[pc_pred == 'pc2'] = 1
price_category[pc_pred == 'pc3'] = 2
price_category.name = 'price_category'

#连续数据离散化
year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))
# 将年份对应映射
# all_features.GarageYrBlt = all_features.GarageYrBlt.map(year_map) # 在数据填充时已经完成该转换了（因为必须先转化后再填充，否则会出错（可以想想到底为什么呢？））
all_features.YearBuilt = all_features.YearBuilt.map(year_map)
all_features.YearRemodAdd = all_features.YearRemodAdd.map(year_map)
#简单函数 规范化 按照比例缩放
numeric_feats = all_features.dtypes[all_features.dtypes != "object"].index
t = all_features[numeric_feats].quantile(.75) # 取四分之三分位
use_75_scater = t[t != 0].index
all_features[use_75_scater] = all_features[use_75_scater]/all_features[use_75_scater].quantile(.75)

# 标准化数据使符合正态分布
from scipy.special import boxcox1p

t = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
     '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
# all_features.loc[:, t] = np.log1p(all_features.loc[:, t])
train_data["SalePrice"] = np.log1p(train_data["SalePrice"])  # 对于SalePrice 采用log1p较好---np.expm1(clf1.predict(X_test))

lam = 0.15  # 100 * (1-lam)% confidence
for feat in t:
    all_features[feat] = boxcox1p(all_features[feat], lam)  # 对于其他属性，采用boxcox1p较好

# 将标称型变量二值化

X = pd.get_dummies(all_features)
X = X.fillna(X.mean())

X = X.drop('Condition2_PosN', axis=1)
X = X.drop('MSZoning_C (all)', axis=1)
X = X.drop('MSSubClass_160', axis=1)
X = pd.concat((X, newer_dwelling, season, year_list, qu_list, MasVnrType_Any,SaleCondition_PriceDown), axis=1)  # SaleCondition_PriceDow

# 创建新属性

from itertools import product, chain
# chain(iter1, iter2, ..., iterN):
# 给出一组迭代器(iter1, iter2, ..., iterN)，此函数创建一个新迭代器来将所有的迭代器链接起来，
# 返回的迭代器从iter1开始生成项，知道iter1被用完，然后从iter2生成项，这一过程会持续到iterN中所有的项都被用完。
def poly(X):
    areas = ['LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'BsmtUnfSF']  # 5个
    t = chain(qu_list.axes[1].get_values(),year_list.axes[1].get_values(),ordinalList,
              ['MasVnrType_Any'])  #,'Neighborhood_Good','SaleCondition_PriceDown'
    for a, t in product(areas, t):
        x = X.loc[:, [a, t]].prod(1) # 返回各维数组的乘积
        x.name = a + '_' + t
        yield x

XP = pd.concat(poly(X), axis=1) # (2917, 155)
X = pd.concat((X, XP), axis=1) # (2917, 466)
X_train_data = X[:train_data.shape[0]]
X_test = X[train_data.shape[0]:]
print(X_train_data.shape)#(1458, 460)
print(X_test.shape)#(1458, 460)
Y= train_data.SalePrice
train_data_now = pd.concat([X_train_data,Y], axis=1)
test_now = X_test
#将处理后的结果进行保存，用于接下来构建模型

train_data_now.to_csv('house_prices/train_afterchange.csv')
test_now.to_csv('house_prices/test_afterchange.csv')



























































# #连续数值的特征做标准化,缺省补0，数乘运算使用apply
# all_features[numeric]=all_features[numeric].apply(lambda x:(x-x.mean())/x.std())
# all_features[numeric]=all_features[numeric].fillna(0)
#
# #get_dummies 是利用pandas实现one hot encode(特征数增加，即列数增加)
# all_features=pd.get_dummies(all_features,dummy_na=True)
# #将训练集和测试集分开，并且向量化
# n_train=train_data.shape[0]
# train_features=np.array(all_features[:n_train].values)
# test_features=np.array(all_features[n_train:].values)
# train_labels=np.array(train_data.SalePrice.values).reshape((-1,1))
# print("处理后数据:",train_features.shape)
# exam=train_features[0:4]
# print(exam)
