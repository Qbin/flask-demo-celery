#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/19 15:48
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : dbscan.py


# -*- coding:utf-8 -*-

"""
Create on 2020/9/28 4:13 下午
@Author: dfsj
@Description:  Kmeans 文本聚类
"""
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np

from algorithms.config import *
from common.time_cost import calculate_runtime


class Dbscan:
    """ DBSCAN 文本聚类算法 """

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
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.dbscan.fit(self.X)
        return self.dbscan

    def print_top_terms(self, top_n=10):
        return
        if not self.use_hashing:
            if not self.km:
                _ = self.train()
            logger.info("Top terms per cluster:")
            if self.n_components:
                original_space_centroids = self.svd.inverse_transform(self.km.cluster_centers_)
                order_centroids = original_space_centroids.argsort()[:, ::-1]
            else:
                order_centroids = self.km.cluster_centers_.argsort()[:, ::-1]

            terms = self.vectorizer.get_feature_names()
            cluster_top_n = []
            for i in range(self.num_clusters):
                res = []
                for ind in order_centroids[i, :top_n]:
                    res.append(terms[ind])
                logger.info("Cluster {}: {}".format(i, " ".join(res)))
                cluster_top_n.append(res)
            return {"cluster_top_n": cluster_top_n}
        else:
            logger.warning("hash 编码方式不支持该方法")

    def draw(self):
        # Get cluster labels
        labels = self.dbscan.labels_

        # Get core samples mask
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[self.dbscan.core_sample_indices_] = True

        # Calculate density estimates for each cluster
        kde = KernelDensity(bandwidth=0.5)
        density_estimates = []
        for cluster_label in np.unique(labels):
            cluster_points = self.X[labels == cluster_label]
            kde.fit(cluster_points)
            density_estimates.append(kde.score_samples(cluster_points))

        # Plot cluster density
        plt.figure(figsize=(8, 6))
        for cluster_label, density_estimate in zip(np.unique(labels), density_estimates):
            cluster_points = self.X[labels == cluster_label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.5, label=f'Cluster {cluster_label}')
            x, y = np.meshgrid(np.linspace(cluster_points[:, 0].min(), cluster_points[:, 0].max(), 100),
                               np.linspace(cluster_points[:, 1].min(), cluster_points[:, 1].max(), 100))
            xy = np.vstack([x.ravel(), y.ravel()]).T
            z = np.exp(kde.score_samples(xy))
            z = z.reshape(x.shape)
            plt.contour(x, y, z, cmap='hot')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('DBSCAN Clustering - Cluster Density')
        plt.legend()
        plt.show()

        return self.X, "xxx", labels

    def find_n(self):
        # todo 待完善
        pass
