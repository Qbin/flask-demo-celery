#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/8 8:43 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : test_model.py

from mongoengine import Document, StringField, ListField

from common.time_cost import calculate_runtime


class AnalyzedData(Document):
    data_id = StringField(required=True)
    analyzed_data = ListField(required=True)

    @classmethod
    def batch_insert(cls, documents):
        cls.objects.insert(documents)

    # @classmethod
    # def batch_insert(cls, documents):
    #     bulk_operations = []
    #     for document in documents:
    #         data_id = document['data_id']
    #         operation = UpdateOne({'data_id': data_id}, {'$set': document}, upsert=True)
    #         bulk_operations.append(operation)
    #
    #     cls.objects.insert(bulk_operations)

    @classmethod
    @calculate_runtime
    def batch_find_by_ids(cls, id_list):
        if id_list:
            return cls.objects(data_id__in=id_list)
            # return cls.objects(data_id__in=id_list).only("analyzed_data")
        else:
            return cls.objects()


# 示例用法
if __name__ == "__main__":
    # 假设有一批要插入的文档
    documents = [
        AnalyzedData(data_id='1', analyzed_data=['分词结果1']),
        AnalyzedData(data_id='2', analyzed_data=['分词结果2']),
        AnalyzedData(data_id='3', analyzed_data=['分词结果3'])
    ]

    # 批量插入文档
    AnalyzedData.batch_insert(documents)

    # 假设有一批要查询的id列表
    id_list = ['1', '3']

    # 根据id列表批量查找文档
    results = TextCluster.batch_find_by_ids(id_list)
    for result in results:
        print(result.data_id, result.analyzed_data)
