FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel
RUN apt-get update
RUN apt-get install vim tmux libsm6 libxext6 libxrender-dev git -y
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install  opencv-python thop   
RUN pip install --upgrade git+https://github.com/mit-han-lab/torchprofile.git
WORKDIR /tmp/unique_for_apex
#RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /root
RUN apt-get install libglib2.0-dev -y
RUN pip install Cython scipy
