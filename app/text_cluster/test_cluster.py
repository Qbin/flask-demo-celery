#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 11:40
# @File    : test_cluster.py
from algorithms.dbscan import Dbscan
from algorithms.kmeans import KMEANS
from app.text_cluster.analyzed_data_model import AnalyzedData

from mongoengine import connect

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
    # kmeans.find_n()


@calculate_runtime
def main():
    data_indexes = list(range(1000))
    # data_indexes = ["10000002", "10000003", "10000004", "10000005", "10000006", "10000007", "10000008", "10000009",
    #                 "10000010", "10000011", "10000012", "10000013", "10000014", "10000015", "10000016", "10000017",
    #                 "10000018", "10000019", "10000020", "10000021", "10000022", "10000023", "10000024", "10000025",
    #                 "10000026", "10000027", "10000028", "10000029", "10000030", "10000031", "10000032", "10000033",
    #                 "10000034", "10000035"]
    texts = AnalyzedData.batch_find_by_ids(data_indexes)
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
