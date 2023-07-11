#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 7:38 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : view.py
import logging

from flask import request

from app.text_cluster import text_cluster_bp


@text_cluster_bp.route('/', methods=['GET'])
def index():
    return {"test_msg": "hello text cluster"}


@text_cluster_bp.route('/analyze_data', methods=['POST'])
def analyze_data():
    params = request.form
    a_id = params.get("a_id")
    data_file = request.files.get("file")

    # todo 异步分词、保存数据

    return {"a_id": a_id, "file": data_file.filename}


@text_cluster_bp.route('/gen_cluster', methods=['POST'])
def gen_cluster():
    params = request.json
    a_id = params.get("a_id")
    data_indexes = params.get("data_indexes", None)

    # todo 根据a_id和data_indexes获取数据

    # todo 将数据向量化后聚类

    return {"a_id": a_id, "data_indexes": data_indexes}


@text_cluster_bp.route('/draw_cluster', methods=['POST'])
def draw_cluster():
    params = request.json
    a_id = params.get("a_id")
    data_indexes = params.get("data_indexes", None)

    # todo 根据a_id获取聚类模型和数据

    # todo 返回画图数据

    return {"a_id": a_id, "data_indexes": data_indexes}
