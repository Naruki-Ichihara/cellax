FROM nvcr.io/nvidia/jax:25.04-py3

RUN apt update
RUN apt upgrade -y
RUN pip install --upgrade pip
RUN pip install numpy==1.26.0
RUN apt install software-properties-common -y
RUN add-apt-repository -y ppa:fenics-packages/fenics
RUN apt update
RUN apt install -y fenicsx

WORKDIR /home/
CMD ["/bin/bash"]
