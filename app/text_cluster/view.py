#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 7:38 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : view.py
import logging
import os.path

from flask import request

from algorithms.kmeans import KMEANS
from algorithms.load_datas import Data
from algorithms.util import load_texts, save_texts
from app.text_cluster import text_cluster_bp

TMP_DATA_PATH = "/Users/qinbinbin/Documents/project/flask-demo-celery/tmp_data"


@text_cluster_bp.route('/', methods=['GET'])
def index():
    return {"test_msg": "hello text cluster"}


@text_cluster_bp.route('/analyze_data', methods=['POST'])
def analyze_data():
    params = request.form
    a_id = params.get("a_id")
    field_name = params.get("field_name")
    data_file = request.files.get("file")

    # todo 异步分词、保存数据

    data = Data("logs/1k.xlsx", field_name=field_name)
    if os.path.exists('{}_list.pkl'.format(a_id)):
        base_texts = load_texts(a_id)
        base_texts.extend(data.get_seg_corpus())
        texts = base_texts
    else:
        texts = data.get_seg_corpus()
    save_texts(texts, a_id)
    return {"a_id": a_id, "file": data_file.filename}


def use_kmeans(corpus, cluster_params):
    kmeans = KMEANS(corpus, num_clusters=cluster_params.get("num_clusters"))
    model = kmeans.train()
    # joblib.dump(model, MODEL_KMEANS)
    # todo 将数据向量化后聚类

    # return kmeans.print_top_terms()
    X, centroids, labels = kmeans.draw()
    kmeans.find_nearest_point()
    return {"X": X.tolist(), "centroids": centroids.tolist(), "labels": labels.tolist()}


@text_cluster_bp.route('/gen_cluster', methods=['POST'])
def gen_cluster():
    params = request.json
    a_id = params.get("a_id")
    data_indexes = params.get("data_indexes", None)
    cluster_type = params.get("cluster_type", "kmeans")
    cluster_params = params.get("cluster_params", None)

    # todo 根据a_id和data_indexes获取数据
    base_texts = load_texts(a_id)
    if data_indexes:
        corpus = [' '.join(base_texts[i]) for i in data_indexes]
    else:
        corpus = [' '.join(i) for i in base_texts]
    if cluster_type == "kmeans":
        return use_kmeans(corpus, cluster_params)


@text_cluster_bp.route('/draw_cluster', methods=['POST'])
def draw_cluster():
    params = request.json
    a_id = params.get("a_id")
    data_indexes = params.get("data_indexes", None)

    # todo 根据a_id获取聚类模型和数据

    # todo 返回画图数据

    return {"a_id": a_id, "data_indexes": data_indexes}
