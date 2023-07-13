#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/13 20:59
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : text_cluster_controller.py
import io

import pandas as pd

from algorithms.kmeans import KMEANS
from algorithms.load_datas import Data
from app.text_cluster.analyzed_data_model import AnalyzedData


class TextClusterController:
    # df: pd.DataFrame = None
    # field_name: str = None
    indexes: list = []

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

    def insert_2_db(self, texts):
        documents = [AnalyzedData(data_id=data_id, analyzed_data=[text]) for data_id, text in zip(self.indexes, texts)]

        # 批量插入文档
        AnalyzedData.batch_insert(documents)

    def analyze_data(self, file: io.BytesIO, field_name: str):
        df = self.load_data(file, field_name)
        data = Data()
        texts = data.get_seg_corpus(df)
        self.insert_2_db(texts)
