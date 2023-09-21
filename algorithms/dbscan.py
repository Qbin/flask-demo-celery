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
import logging

# import umap
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
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
        self.vec_X = None
        self.text2vec()

    @calculate_runtime
    def text2vec(self):
        """ 文本向量化表示 """
        self.vectorizer = TfidfVectorizer(max_features=self.n_features, max_df=0.5, min_df=2, use_idf=self.use_idf)
        self.X = self.vectorizer.fit_transform(self.texts)
        self.vec_X = self.X
        logger.info("n_samples: %d, n_features: %d" % self.X.shape)

        self.svd = TruncatedSVD(self.n_components, algorithm='arpack')
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(self.svd, normalizer)
        # lsa = make_pipeline(self.svd)
        self.X = lsa.fit_transform(self.X)
        explained_variance = self.svd.explained_variance_ratio_.sum()
        logger.info("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

    @calculate_runtime
    def train(self):
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.dbscan.fit(self.X)
        return self.dbscan

    def print_top_terms(self, top_n=10):
        # 找出每个簇的关键词
        keywords = []
        labels = self.dbscan.labels_
        for label in set(labels):
            if label == -1:
                continue

            # 获取属于当前簇的样本点的索引
            indices = [i for i, l in enumerate(labels) if l == label]

            # 计算当前簇的TF-IDF特征向量的均值
            cluster_mean = self.vec_X[indices].mean(axis=0)

            # 找到均值向量中最重要的特征的索引
            top_feature_indices = cluster_mean.A.ravel().argsort()[::-1][:top_n]

            # 获取对应的关键词
            keywords.append([self.vectorizer.get_feature_names()[i] for i in top_feature_indices])

        # 打印每个簇的关键词
        for i, k in enumerate(keywords):
            print(f"Cluster {i + 1} keywords: {', '.join(k)}")
        return keywords

    def draw(self):
        # Get cluster labels
        labels = self.dbscan.labels_
        # return self.dbscan.components_, "xxx", labels

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

        if self.X.shape[1] >= 2:
            pca = PCA(2)
            # normalizer = Normalizer(copy=False)
            # lsa = make_pipeline(svd, normalizer)
            self.X = pca.fit_transform(self.X)
            explained_variance = pca.explained_variance_ratio_.sum()
            logger.info("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

        # Plot cluster density
        # plt.figure(figsize=(8, 6))
        # for cluster_label, density_estimate in zip(np.unique(labels), density_estimates):
        #     cluster_points = self.X[labels == cluster_label]
        #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], alpha=0.5, label=f'Cluster {cluster_label}')
        #     x, y = np.meshgrid(np.linspace(cluster_points[:, 0].min(), cluster_points[:, 0].max(), 100),
        #                        np.linspace(cluster_points[:, 1].min(), cluster_points[:, 1].max(), 100))
        #     xy = np.vstack([x.ravel(), y.ravel()]).T
        #     z = np.exp(kde.score_samples(xy))
        #     z = z.reshape(x.shape)
        #     plt.contour(x, y, z, cmap='hot')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('DBSCAN Clustering - Cluster Density')
        # plt.legend()
        # plt.show()

        return self.X, "", self.dbscan.labels_
        # return self.dbscan.components_, "xxx", labels

    def draw_new2(self, decomposition_metch="pca"):
        if decomposition_metch == "umap":
            reducer = umap.UMAP(random_state=42)
            X = reducer.fit_transform(self.X)
        elif decomposition_metch == "tsne":
            tsne = TSNE(n_components=2)
            X = tsne.fit_transform(self.X)
        elif decomposition_metch == "pca":
            pca = PCA(n_components=2)
            X = pca.fit_transform(self.X)
        elif decomposition_metch == "svd":
            svd = TruncatedSVD(2)
            # # normalizer = Normalizer(copy=False)
            # # lsa = make_pipeline(svd, normalizer)
            X = svd.fit_transform(self.X)
            # explained_variance = svd.explained_variance_ratio_.sum()
            # logger.info("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

            # if self.X.shape[1] >= 2:
            #     pca = PCA(2)
            #     # normalizer = Normalizer(copy=False)
            #     # lsa = make_pipeline(svd, normalizer)
            #     self.X = pca.fit_transform(self.X)
            #     explained_variance = pca.explained_variance_ratio_.sum()
            #     logger.info("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

            # 获取聚类中心和预测的标签
        # centroids = self.km.cluster_centers_
        labels = self.dbscan.labels_
        # # 绘制数据点和聚类中心
        # plt.title(decomposition_metch)
        # plt.scatter(X[:, 0], X[:, 1], c=labels)
        # # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='r')
        # plt.show()
        return X, "centroids", labels

    def draw_new(self):
        # 获取核心点的索引
        core_samples_mask = np.zeros_like(self.dbscan.labels_, dtype=bool)
        core_samples_mask[self.dbscan.core_sample_indices_] = True

        # 获取每个点的标签
        labels = self.dbscan.labels_

        # 获取聚类数量
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # 画出密度图
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # 为噪点设置黑色
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = self.X[class_member_mask & core_samples_mask]
            plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=u'o', alpha=0.5)

            xy = self.X[class_member_mask & ~core_samples_mask]
            plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker=u'x', alpha=0.5)

        plt.title('DBSCAN Clustering')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    def find_density_max_point_indices(self):
        # 获取核心点的索引
        core_samples_indices = self.dbscan.core_sample_indices_

        # 计算每个簇的密度最大点的索引
        density_max_point_indices = []
        for cluster_label in set(self.dbscan.labels_):
            if cluster_label != -1:  # 排除噪声点
                cluster_core_samples = self.X[self.dbscan.labels_ == cluster_label]
                nbrs = NearestNeighbors(n_neighbors=len(cluster_core_samples)).fit(cluster_core_samples)
                distances, indices = nbrs.kneighbors(cluster_core_samples)
                density = np.sum(distances[:, 1:], axis=1)  # 计算密度
                max_density_index = np.argmax(density)  # 获取密度最大点的索引
                density_max_point_index = np.where(self.dbscan.labels_ == cluster_label)[0][max_density_index]
                density_max_point_indices.append(int(density_max_point_index))

        # 打印每个簇的密度最大点的索引
        for i, index in enumerate(density_max_point_indices):
            logging.info("簇{}的密度最大点的索引：{}".format(i, index))
            logging.info(self.X[index])

        return density_max_point_indices
