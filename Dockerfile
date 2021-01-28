#FROM python:3.7.6
FROM gaia:v0.1.1

COPY . /opt
WORKDIR /opt
#RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.douban.com/simple
CMD ["gunicorn", "-b", "0.0.0.0:80", "--timeout", "600", "server:app"]

EXPOSE 80