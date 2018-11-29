FROM python:3.6

RUN pip3 install --upgrade pip

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

RUN mkdir -p /data
WORKDIR /app
COPY resnet-model.py .

CMD ["python", "resnet-model.py"]