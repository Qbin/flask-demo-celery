#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding:utf-8 -*-

"""
Create on 2020/9/28 4:13 下午
@Author: dfsj
@Description:  Cmeans 文本聚类
"""
import logging

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import DBSCAN
import numpy as np
import skfuzzy as fuzz

from algorithms.config import *
from common.time_cost import calculate_runtime


class CMeans:
    """ CMeans 文本聚类算法 """

    vectorizer = None
    X = None
    dbscan = None
    svd = None
    is_draw = False

    def __init__(self, texts, eps=0.5, min_samples=2, n_components=100,
                 n_features=250000, use_hashing=False, use_idf=True, is_draw=False):
        """
        :param texts:           聚类文本
        :param num_clusters:    聚类数
        :param minibatch:       是否是否 MiniBatchKMeans
        :param n_components:    使用潜在语义分析处理文档，可以设置为 None 不进行压缩
        :param n_features:      特征（维度）的最大数量，特征压缩，只用于 hash 特征表示
        :param use_hashing:     hash 特征向量
        :param use_idf:         是否使用逆文档频率特征
        """
        self.texts = texts
        self.eps = eps
        self.min_samples = min_samples
        self.n_components = n_components
        self.n_features = n_features
        self.use_hashing = use_hashing
        self.use_idf = use_idf
        self.is_draw = is_draw
        self.text2vec()

    @calculate_runtime
    def text2vec(self):
        """ 文本向量化表示 """
        self.vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, use_idf=self.use_idf)
        self.X = self.vectorizer.fit_transform(self.texts)
        logger.info("n_samples: %d, n_features: %d" % self.X.shape)

        self.svd = TruncatedSVD(2)
        normalizer = Normalizer(copy=False)
        # lsa = make_pipeline(self.svd, normalizer)
        lsa = make_pipeline(self.svd)
        self.X = lsa.fit_transform(self.X)
        explained_variance = self.svd.explained_variance_ratio_.sum()
        logger.info("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

    @calculate_runtime
    def train(self):
        n_clusters = 3
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(self.X.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)
        logging.info("聚类中心 {}".format(cntr))
        labels = np.argmax(u, axis=0)
        logging.info("数据类别 {}".format(labels))
        return

    def print_top_terms(self, top_n=10):
        return

    def draw(self):
        pass
