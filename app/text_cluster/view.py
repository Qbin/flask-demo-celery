#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 7:38 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : view.py
import logging
import os.path

from flask import request

from algorithms.util import load_texts, save_texts
from app.text_cluster import text_cluster_bp
from app.text_cluster.text_cluster_controller import TextClusterController

TMP_DATA_PATH = "/Users/qinbinbin/Documents/project/flask-demo-celery/tmp_data"


@text_cluster_bp.route('/', methods=['GET'])
def index():
    return {"test_msg": "hello text cluster"}


@text_cluster_bp.route('/analyze_data', methods=['POST'])
def analyze_data():
    # todo 接口加锁
    # todo 支持数据批量覆盖
    params = request.form
    # a_id = params.get("a_id")
    field_name = params.get("field_name")
    data_file = request.files.get("file")

    # todo 异步分词、保存数据
    tcc = TextClusterController()
    tcc.analyze_data(data_file, field_name)

    return {"field_name": field_name, "file": data_file.filename}


@text_cluster_bp.route('/gen_cluster', methods=['POST'])
def gen_cluster():
    # todo 响应时间过长
    params = request.json
    a_id = params.get("a_id")
    field_name = params.get("field_name", "analyzed_data")
    data_indexes = params.get("data_indexes", None)
    cluster_type = params.get("cluster_type", "kmeans")
    cluster_params = params.get("cluster_params", None)

    # todo 根据a_id和data_indexes获取数据
    tcc = TextClusterController()
    result, _ = tcc.gen_cluster(data_indexes, cluster_type, cluster_params, field_name)
    return result


@text_cluster_bp.route('/draw_cluster', methods=['POST'])
def draw_cluster():
    # todo 增加距离质点最近的数据id
    # todo 校验model和data_indexes的匹配度
    params = request.json
    model_id = params.get("model_id")

    tcc = TextClusterController()
    return tcc.draw_cluster(model_id)
