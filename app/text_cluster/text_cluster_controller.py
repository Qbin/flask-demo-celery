#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/13 20:59
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : text_cluster_controller.py
import io
import os
import uuid

import joblib
import pandas as pd

from flask import current_app

from algorithms.kmeans import KMEANS
from algorithms.load_datas import Data
from app.text_cluster.analyzed_data_model import AnalyzedData
from app.text_cluster.cluster_models_model import ClusterModels
from common.time_cost import calculate_runtime


class TextClusterController:
    # df: pd.DataFrame = None
    # field_name: str = None
    indexes: list = []
    is_draw: bool = False

    def __init__(self):
        pass

    def load_data(self, file: io.BytesIO, field_name: str):
        # 创建BytesIO对象，用于读取文件内容
        file_stream = io.BytesIO(file.read())
        # 使用read_excel函数创建DataFrame
        src_df = pd.read_excel(file_stream, engine='openpyxl', )
        # df = src_df[[field_name]].rename(columns={field_name: "data"})
        self.indexes = src_df["id"]
        return src_df[field_name]

    def insert_texts_2_db(self, texts):
        # todo 如果data_id已存在，则覆盖
        documents = [AnalyzedData(data_id=data_id, analyzed_data=text) for data_id, text in zip(self.indexes, texts)]

        # 批量插入文档
        AnalyzedData.batch_insert(documents)

    def insert_model_2_db(self, model_type, model_params):
        model = ClusterModels.add_model(data_indexes=self.indexes, model_type=model_type, model_params=model_params)
        return model.str_id

    def save_model(self, model, cluster_params):
        model_id = self.insert_model_2_db("kmeans", cluster_params)
        model_name = "{}.model".format(model_id)
        model_file_name = os.path.join(current_app.root_path, "model", model_name)
        joblib.dump(model, model_file_name)
        return model_id

    @calculate_runtime
    def get_analyzed_data(self, data_indexes):
        self.indexes = data_indexes
        query_set = AnalyzedData.batch_find_by_ids(data_indexes)
        return query_set

    def analyze_data(self, file: io.BytesIO, field_name: str):
        df = self.load_data(file, field_name)
        data = Data()
        texts = data.get_seg_corpus(df)
        self.insert_texts_2_db(texts)

    def use_kmeans(self, corpus, cluster_params):
        if self.is_draw:
            kmeans = KMEANS(corpus, num_clusters=cluster_params.get("num_clusters"), n_components=2,
                            is_draw=self.is_draw)
        else:
            kmeans = KMEANS(corpus, num_clusters=cluster_params.get("num_clusters"), is_draw=self.is_draw)
        model = kmeans.train()
        model_id = self.save_model(model, cluster_params)
        # kmeans.print_top_terms()
        nearest_points = None
        # nearest_points = kmeans.find_nearest_point()
        # todo 待完善
        # kmeans.find_n()

        return {"model_id": model_id, "nearest_points": nearest_points}, kmeans

        # return kmeans.print_top_terms()
        # X, centroids, labels = kmeans.draw()
        # return {"X": X.tolist(), "centroids": centroids.tolist(), "labels": labels.tolist()}

    def gen_cluster(self, data_indexes, cluster_type, cluster_params):
        texts = self.get_analyzed_data(data_indexes)
        corpus = [' '.join(i.analyzed_data) for i in texts]
        if cluster_type == "kmeans":
            return self.use_kmeans(corpus, cluster_params)

    def draw_cluster(self, model_id):
        self.is_draw = True
        model_obj = ClusterModels.get_by_id(model_id)
        data_indexes = model_obj.data_indexes
        model_params = model_obj.model_params
        model_type = model_obj.model_type
        _, kmeans = self.gen_cluster(data_indexes, model_type, model_params)
        X, centroids, labels = kmeans.draw()
        return {"X": X.tolist(), "centroids": centroids.tolist(), "labels": labels.tolist()}
