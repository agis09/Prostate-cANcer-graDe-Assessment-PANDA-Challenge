FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
RUN    apt update  -qy
RUN    apt upgrade -qy
RUN    apt install wget -qy
RUN    apt install tar -qy
RUN    apt install curl -qy
RUN    apt install python3-pip -qy
RUN    apt install -qy vim 
RUN    apt install -qy git
RUN    pip3 install -U pip
RUN    pip install -q pandas
RUN    pip install -q tensorflow-gpu
RUN    pip install -q keras
RUN    pip install -q scikit-learn
RUN    pip install -q jupyterlab
RUN    pip install -q matplotlib
RUN    pip install -q seaborn
RUN    pip install -q tqdm
RUN    pip install -q umap-learn
RUN    pip install -q cupy-cuda100
RUN    pip install -q scikit-image
RUN    pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# RUN    pip install pytorch-warmup
RUN    pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
RUN    pip install efficientnet_pytorch
RUN    pip install albumentations
RUN    pip install imagecodecs