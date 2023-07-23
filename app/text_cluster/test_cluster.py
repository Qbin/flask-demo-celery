#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 11:40
# @File    : test_cluster.py
from algorithms.dbscan import Dbscan
from algorithms.kmeans import KMEANS
from app.text_cluster.analyzed_data_model import AnalyzedData

from mongoengine import connect

from app.text_cluster.utils import md5_encrypt
from common.time_cost import calculate_runtime

connect(db='prophet_k8s_test', host='mongodb://mongo:73c74b9461f00453@10.220.139.135:7830',
        authentication_source='admin')

from concurrent.futures import ThreadPoolExecutor


def join_analyzed_data(i):
    return ' '.join(i.analyzed_data)


def test_kmeans(corpus):
    kmeans = KMEANS(corpus, num_clusters=3, n_components=2, is_draw=True)
    model = kmeans.train()
    kmeans.draw()
    print(kmeans.find_closest_samples())


def test_dbscan(corpus):
    dbscan = Dbscan(corpus, eps=0.06, min_samples=2)
    model = dbscan.train()
    dbscan.draw()
    # model_id = self.save_model(model, cluster_params, "dbscan")
    # kmeans.print_top_terms()
    nearest_points = None
    # nearest_points = kmeans.find_nearest_point()
    # todo 待完善
    print(dbscan.find_density_max_point_indices())


@calculate_runtime
def main():
    data_indexes = list(range(1000))
    data_indexes = [str(i + 10000000) for i in range(0, 1000)]
    texts = AnalyzedData.batch_find_by_ids(data_indexes, filed_name=md5_encrypt("正文"))
    corpus = [' '.join(i.analyzed_data) for i in texts]
    # corpus = list(map(lambda i: ' '.join(i.analyzed_data), texts))

    # 创建线程池
    # with ThreadPoolExecutor() as executor:
    #     # 使用线程池并行处理任务
    #     corpus = list(executor.map(join_analyzed_data, texts))

    print(len(corpus))
    # test_kmeans(corpus)
    test_dbscan(corpus)
    # model_id = self.save_model(model, cluster_params)


if __name__ == '__main__':
    main()
