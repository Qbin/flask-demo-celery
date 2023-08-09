# 下载代码
wget https://codeload.github.com/Qbin/flask-demo-celery/zip/refs/heads/text_cluster_mongo
# 解压
unzip text_cluster_mongo
# 删除注释
cd flask-demo-celery-text_cluster_mongo
find . -type f -name "*.py" -exec sed -i '' -e '/^[[:blank:]]*#/d' -e 's/#.*//' {} \;
# 构建镜像
docker build -t text_cluster:s_v5 .

