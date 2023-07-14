# -*- coding:utf-8 -*-

"""
Create on 2020/9/28 4:13 下午
@Author: dfsj
@Description:  Kmeans 文本聚类
"""
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

from algorithms.config import *
from common.time_cost import calculate_runtime


class KMEANS:
    """ KMeans 文本聚类算法 """

    vectorizer = None
    X = None
    km = None
    svd = None

    def __init__(self, texts, num_clusters=10, minibatch=True, n_components=2,
                 n_features=250000, use_hashing=False, use_idf=True):
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
        self.num_clusters = num_clusters
        self.minibatch = minibatch
        self.n_components = n_components
        self.n_features = n_features
        self.use_hashing = use_hashing
        self.use_idf = use_idf
        self.text2vec()

    @calculate_runtime
    def text2vec(self):
        """ 文本向量化表示 """
        if self.use_hashing:
            if self.use_idf:
                # Perform an IDF normalization on the output of HashingVectorizer
                hasher = HashingVectorizer(n_features=self.n_features, alternate_sign=False, norm=None)
                self.vectorizer = make_pipeline(hasher, TfidfTransformer())
            else:
                self.vectorizer = HashingVectorizer(n_features=self.n_features, alternate_sign=False, norm='l2')
        else:
            self.vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, use_idf=self.use_idf)
        self.X = self.vectorizer.fit_transform(self.texts)
        logger.info("n_samples: %d, n_features: %d" % self.X.shape)

        if self.n_components:
            logger.info("Performing dimensionality reduction using LSA")
            # Vectorizer results are normalized, which makes KMeans behave as
            # spherical k-means for better results. Since LSA/SVD results are
            # not normalized, we have to redo the normalization.
            self.svd = TruncatedSVD(self.n_components)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(self.svd, normalizer)
            self.X = lsa.fit_transform(self.X)
            explained_variance = self.svd.explained_variance_ratio_.sum()
            logger.info("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

    @calculate_runtime
    def train(self):
        if self.minibatch:
            self.km = MiniBatchKMeans(n_clusters=self.num_clusters, init='k-means++', n_init=1,
                                      init_size=1000, batch_size=1000, verbose=False)
        else:
            self.km = KMeans(n_clusters=self.num_clusters, init='k-means++', max_iter=100, n_init=1, verbose=False)

        self.km.fit(self.X)
        return self.km

    def print_top_terms(self, top_n=10):
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

    def print_summary(self, labels=None):
        """ labels 为该数据集的真实类别标签，真实数据可能不存在该标签，因此部分指标可能不可用 """
        if not self.km:
            _ = self.train()
        if labels is not None:
            logger.info("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, self.km.labels_))
            logger.info("Completeness: %0.3f" % metrics.completeness_score(labels, self.km.labels_))
            logger.info("V-measure: %0.3f" % metrics.v_measure_score(labels, self.km.labels_))
            logger.info("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, self.km.labels_))
        logger.info("Silhouette Coefficient: %0.3f" %
                    metrics.silhouette_score(self.X, self.km.labels_, metric='euclidean'))

        result = list(self.km.predict(self.X))
        logger.info('Cluster distribution:')
        logger.info(dict([(i, result.count(i)) for i in result]))
        logger.info(-self.km.score(self.X))

    def draw(self, km_model=None):
        if km_model:
            model = km_model
        else:
            model = self.km
        if self.X.shape[1] > 2:
            pca = PCA(2)
            # normalizer = Normalizer(copy=False)
            # lsa = make_pipeline(svd, normalizer)
            X = pca.fit_transform(self.X)
            explained_variance = pca.explained_variance_ratio_.sum()
            logger.info("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
        else:
            X = self.X

        # 获取聚类中心和预测的标签
        centroids = model.cluster_centers_
        labels = model.labels_
        # # 绘制数据点和聚类中心
        # # plt.scatter(self.X[:, 0], self.X[:, 1], c=labels)
        # plt.scatter(X[:, 0], X[:, 1], c=labels)
        # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='r')
        # plt.show()
        return X, centroids, labels

    def find_nearest_point(self):
        # 计算每个样本点到每个簇质心的距离
        distances = self.km.transform(self.X)
        labels = self.km.labels_
        nearest_points = []
        for i in range(self.km.n_clusters):
            cluster_points = self.X[labels == i]
            nearest_point_index = np.argmin(distances[labels == i, i])
            nearest_point = cluster_points[nearest_point_index]
            nearest_points.append(nearest_point.tolist())

        print(nearest_points)
        return nearest_points

    @calculate_runtime
    def find_optimal_clusters(self, max_k):
        iters = range(2, max_k + 1, 2)
        # svd = TruncatedSVD(2)
        # X = svd.fit_transform(self.X)
        sse = []
        for k in iters:
            sse.append(
                MiniBatchKMeans(n_clusters=k, init="k-means++", init_size=1024, batch_size=2048, random_state=20).fit(
                    self.X).inertia_)
            logger.info('Fit {} clusters'.format(k))

        f, ax = plt.subplots(1, 1)
        ax.plot(iters, sse, marker='o')
        ax.set_xlabel('Cluster Centers')
        ax.set_xticks(iters)
        ax.set_xticklabels(iters)
        ax.set_ylabel('SSE')
        ax.set_title('SSE by Cluster Center Plot')
        plt.show()
        plt.savefig(PIC_KMEANS)
