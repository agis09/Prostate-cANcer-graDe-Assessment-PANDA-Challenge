#!/bin/bash
docker run \
--runtime=nvidia \
--name kaggle \
-p 8891:8888 \
-it \
--shm-size=32gb \
-v $(pwd):/workdir \
-w /workdir \
kaggle \
"$@"
