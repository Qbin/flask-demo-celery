#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/13 20:59
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : text_cluster_controller.py
import io
import logging
import os
import time

import joblib
import pandas as pd

from flask import current_app

from algorithms.dbscan import Dbscan
from algorithms.kmeans import KMEANS
from algorithms.load_datas import Data
from app.text_cluster.analyzed_data_model import AnalyzedData
from app.text_cluster.cluster_models_model import ClusterModels
from app.text_cluster.cluster_error import ClusterError
from app.text_cluster.utils import md5_encrypt
from common.time_cost import calculate_runtime


class TextClusterController:
    # df: pd.DataFrame = None
    # field_name: str = None
    indexes: list = []
    is_draw: bool = False
    field_name: str = None
    dim: int = 100

    def __init__(self):
        pass

    def load_data(self, file: io.BytesIO, field_name: str):
        # 创建BytesIO对象，用于读取文件内容
        file_stream = io.BytesIO(file.read())
        # 使用read_excel函数创建DataFrame
        src_df = pd.read_excel(file_stream, engine='openpyxl', )
        # src_df = pd.read_csv(file_stream)
        # df = src_df[[field_name]].rename(columns={field_name: "data"})
        self.indexes = src_df["id"]
        self.field_name = field_name
        return src_df[field_name]

    def insert_texts_2_db(self, texts):
        # todo 如果data_id已存在，则覆盖
        # 为确保id的类型一致，这里强制转成string
        # documents = [AnalyzedData(data_id=str(data_id), analyzed_data=text) for data_id, text in
        #              zip(self.indexes, texts)]
        # # 批量插入文档
        # AnalyzedData.batch_insert(documents, filed_name=md5_encrypt(self.field_name))

        AnalyzedData.batch_upsert(self.indexes, texts, filed_name=md5_encrypt(self.field_name))

    def insert_model_2_db(self, model_type, model_params):
        model = ClusterModels.add_model(data_indexes=self.indexes, model_type=model_type, model_params=model_params,
                                        field_name=self.field_name)
        return model.str_id

    def save_model(self, model, cluster_params, cluster_type="kmeans"):
        model_id = self.insert_model_2_db(cluster_type, cluster_params)
        model_name = "{}.model".format(model_id)
        model_file_name = os.path.join(current_app.root_path, "model", model_name)
        joblib.dump(model, model_file_name)
        return model_id

    @calculate_runtime
    def get_analyzed_data(self, data_indexes):
        self.indexes = data_indexes
        query_set = AnalyzedData.batch_find_by_ids(data_indexes, md5_encrypt(self.field_name))
        if not query_set:
            raise ClusterError(ClusterError.NO_DATA, "没有找到数据")
        return query_set

    def analyze_data(self, file: io.BytesIO, field_name: str):
        self.field_name = field_name
        df = self.load_data(file, field_name)
        data = Data()
        texts = data.get_seg_corpus(df)
        self.insert_texts_2_db(texts)

    def use_kmeans(self, corpus, cluster_params):
        try:
            if self.is_draw:
                kmeans = KMEANS(corpus, num_clusters=cluster_params.get("num_clusters"), n_components=2,
                                is_draw=self.is_draw)
            else:
                kmeans = KMEANS(corpus, num_clusters=cluster_params.get("num_clusters"), n_components=self.dim,
                                is_draw=self.is_draw)
        except ValueError as e:
            logging.exception(e)
            logging.error("向量化异常")
            raise ClusterError(ClusterError.TEXT2VEC, "向量化异常")
        model = kmeans.train()
        model_id = self.save_model(model, cluster_params)
        cluster_keywords = kmeans.print_top_terms()
        nearest_points = kmeans.find_closest_samples()
        kmeans.draw()

        return {"model_id": model_id, "nearest_points": nearest_points, "cluster_keywords": cluster_keywords,
                "X": kmeans.X.tolist(), "labels": kmeans.km.labels_.tolist()}, kmeans

        # return kmeans.print_top_terms()
        # X, centroids, labels = kmeans.draw()
        # return {"X": X.tolist(), "centroids": centroids.tolist(), "labels": labels.tolist()}

    def use_dbscan(self, corpus, cluster_params):
        try:
            dbscan = Dbscan(corpus, eps=cluster_params.get("eps", 0.5), n_components=self.dim,
                            min_samples=cluster_params.get("min_samples", 2))
        except Exception as e:
            logging.error("向量化异常")
            raise ClusterError(ClusterError.TEXT2VEC, "向量化异常")
        model = dbscan.train()
        model_id = self.save_model(model, cluster_params, "dbscan")
        # kmeans.print_top_terms()
        nearest_points = None
        # nearest_points = kmeans.find_nearest_point()
        dbscan.draw()
        # todo 待完善
        nearest_points = dbscan.find_density_max_point_indices()
        logging.info("nearest_points: {}, labels: {} ".format(nearest_points, model.labels_))

        return {"model_id": model_id, "nearest_points": nearest_points,
                "X": dbscan.X.tolist(), "labels": model.labels_.tolist()}, dbscan

        # return kmeans.print_top_terms()
        # X, centroids, labels = kmeans.draw()
        # return {"X": X.tolist(), "centroids": centroids.tolist(), "labels": labels.tolist()}

    def gen_cluster(self, data_indexes, cluster_type, cluster_params, field_name, dim=100):
        self.field_name = field_name
        self.dim = dim
        texts = self.get_analyzed_data(data_indexes)
        st = time.time()
        if os.getenv("DB_MODE") == "mongo":
            logging.info("use DB MODE")
            corpus = [' '.join(i["analyzed_data"]) for i in texts]
        else:
            corpus = [' '.join(i.analyzed_data) for i in texts]
        logging.info("get analyzed_data cost time {}".format(time.time() - st))
        if cluster_type == "kmeans":
            return self.use_kmeans(corpus, cluster_params)
        elif cluster_type == "dbscan":
            return self.use_dbscan(corpus, cluster_params)

    def draw_cluster(self, model_id):
        self.is_draw = True
        model_obj = ClusterModels.get_by_id(model_id)
        data_indexes = model_obj.data_indexes
        model_params = model_obj.model_params
        model_type = model_obj.model_type
        self.field_name = model_obj.field_name
        if model_type == "kmeans":
            _, kmeans = self.gen_cluster(data_indexes, model_type, model_params, self.field_name)
            X, centroids, labels = kmeans.draw()
            cluster_keywords = kmeans.print_top_terms()
            nearest_points = kmeans.find_closest_samples()
            return {"X": X.tolist(), "centroids": centroids.tolist(), "labels": labels.tolist(),
                    "nearest_points": nearest_points, "cluster_keywords": cluster_keywords}
        else:
            _, dbscan = self.gen_cluster(data_indexes, model_type, model_params, self.field_name)
            X, centroids, labels = dbscan.draw()
            nearest_points = dbscan.find_density_max_point_indices()
            # logging.info("DBSCAN nearest_points: {}, labels: {} ".format(nearest_points, labels))
            # model_name = "{}.model".format(model_id)
            # model_file_name = os.path.join(current_app.root_path, "model", model_name)
            # model = joblib.load(model_file_name)
            # core_labels = model.labels_[model.core_sample_indices_]
            return {"X": X.tolist(), "model_params": model_params, "labels": labels.tolist(),
                    "nearest_points": nearest_points}

    def is_seg_data(self, data_indexes, field_name):
        query_set = AnalyzedData.get_exist_data_id(data_indexes, md5_encrypt(field_name))
        return [i["data_id"] for i in query_set]
