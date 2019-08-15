#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019-08-15 15:58
# @Author  : 冯佳欣
# @File    : tree.py
# @Desc    :
# 建立cart回归决策树
class Node:
    def __init__(self, isLeaf=True, feature_val=None, feature_name=None, feature_axis=None, y=None):
        self.isLeaf = isLeaf
        self.feature_val = feature_val
        self.y = y
        self.feature_name = feature_name
        self.feature_axis = feature_axis
        self.tree = {}
        self.result = {
                'y': self.y,
                'feature_val': self.feature_val,
                'feature_name': self.feature_name,
                'tree': self.tree
            }

    def __repr__(self):
        return '{}'.format(self.result)

    def add_child(self, feature_val, left_tree, right_tree):
        self.tree['left'] = left_tree
        self.tree['right'] = right_tree

    def predict(self, features):
        '''
        features是一个特征的列表
        '''
        if self.isLeaf:
            return self.y
        vec_val = features[self.feature_axis]
        if vec_val <= self.feature_val:
            return self.tree['left'].predict(features)
        else:
            return self.tree['right'].predict(features)


class DTree:
    def __init__(self, tolS, tolN):
        # 容许的误差下降值，如果切分后误差下降值< tolS，停止划分，作为叶子节点
        self.tolS = tolS
        # 容许的最少样本数，如果剩下的样本数<tolN,停止划分，作为叶子节点
        self.tolN = tolN
        self._tree = {}

    # 计算数据集的平均值
    def regleaf(self,dataset, label):
        '''
        dataset:df格式
        '''
        # 最后一列
        return dataset[label].mean()

    # 计算数据集的方差和
    def regErr(self,dataset,label):
        return dataset[label].var()

    # 切割数据
    def bin_split_data(self ,dataset, feature, value):
        mat0 = dataset.loc[dataset[feature] <= value]
        mat1 = dataset.loc[dataset[feature] > value]
        return mat0, mat1

    # 从数据集中选择最好的特征进行划分，返回feature_axis,feature_val,如果不能划分，返回None ,叶子节点值
    def choose_best_split(self,dataset, tolS, tolN):
        # 1.如果数据集的所有值都相等
        feature_list = dataset.columns[:-1]
        label = dataset.columns[-1]

        if len(dataset[label].value_counts()) == 1:
            return None, self.regleaf(dataset, label)
        # 不划分的误差和
        S = self.regErr(dataset, label)
        # 记录最小的误差和，对应的特征index,以及划分值
        best_s = float('inf')
        best_index = None
        best_value = None

        for feat_index, feat_name in enumerate(feature_list):
            for splitVal in dataset[feat_name].unique():
                mat0, mat1 = self.bin_split_data(dataset, feat_name, splitVal)
                # 切割后的数据过少
                if mat0.shape[0] < tolN or mat1.shape[0] < tolN:
                    continue
                new_s = self.regErr(mat0, label) + self.regErr(mat1, label)
                if new_s < best_s:
                    best_s = new_s
                    best_index = feat_index
                    best_value = splitVal
        # 如果误差减少不大，则不进行划分
        if S - best_s < tolS:
            return None, self.regleaf(dataset,label)
        return best_index, best_value

    def train(self, dataset):
        feat_index, feat_val = self.choose_best_split(dataset, self.tolS, self.tolN)
        features_list = dataset.columns[:-1]
        label = dataset.columns[-1]

        if feat_index is None:
            # 建立叶子节点
            return Node(isLeaf=True, y=self.regleaf(dataset, label))
        else:
            node_tree = Node(isLeaf=False, feature_val=feat_val, feature_name=features_list[feat_index],
                             feature_axis=feat_index)
            mat0, mat1 = self.bin_split_data(dataset, features_list[feat_index], feat_val)
            left_tree = self.train(mat0)
            right_tree = self.train(mat1)
            node_tree.add_child(feat_val,left_tree,right_tree)
            return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)

import pandas as pd
def load_df(file_name):
    temp_list = []
    with open(file_name,'r') as f:
        for line in f:
            vec = line.strip().split('\t')
            x = float(vec[0])
            y = float(vec[1])
            temp_list.append([x,y])
    return pd.DataFrame(temp_list,columns=['x','y'])

read_file = './ex_0.txt'
data_df = load_df(read_file)
tol_s = 0.1
tol_n = 2
dt = DTree(tol_s,tol_n)
tree = dt.fit(data_df)
print(tree)
