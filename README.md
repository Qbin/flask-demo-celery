## flask demo with celery

### 环境
1. python 3.7.6
2. mongo 4.0

### 目录结构
    ```
    .
    ├── Dockerfile  # docker文件
    ├── README
    ├── app # 应用目录
    │   ├── __init__.py # 注册蓝图
    │   ├── test    # 测试代码
    ├── application.py  # 配置Flask的app
    ├── common  # 基础公用代码
    │   ├── base_error.py  # 自定义异常
    │   ├── custom_response.py  # 自定义response
    │   ├── decorators.py  # 自定义装饰器
    │   └── service_decorator.py   # 服务装饰器
    ├── config  # 配置文件
    │   ├── config.py   # app配置
    │   └── logger.py   # 日志配置
    ├── db.py   # mysql实例
    ├── application.py   # flask app创建
    ├── celery_extensions.py   # celery实例
    ├── instance
    ├── requirements.txt    # python依赖
    ├── run.py  # 单机运行入口
    └── server.py   # gunicorn运行入口
    ```

### 启动
1. 单机启动
    > python run.py
2. gunicorn启动
    > gunicorn -b 0.0.0.0:80 --timeout 600 server:app
    > celery -A server.my_celery worker --loglevel=info

### 测试
1. 访问 http://localhost:8000/test/
