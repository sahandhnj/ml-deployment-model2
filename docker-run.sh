#!/bin/bash

DATA_DIR=$(pwd)/data
IMAGES_DIR=$(pwd)/images

docker build -t ml-resnet:latest .
docker run -v $DATA_DIR:/app/data -v $IMAGES_DIR:/app/images ml-resnet:latest