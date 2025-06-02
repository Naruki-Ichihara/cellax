FROM nvcr.io/nvidia/jax:25.04-py3

RUN apt update
RUN apt upgrade -y
RUN pip install --upgrade pip
RUN pip install numpy==1.26.0
#RUN apt install libxinerama1 -y
#RUN apt install libxcursor1 -y
#RUN apt install libglu1-mesa -y
RUN apt install software-properties-common -y
RUN add-apt-repository -y ppa:fenics-packages/fenics
RUN apt-add-repository -y universe
RUN add-apt-repository -y ppa:ngsolve/ngsolve
RUN apt update


# TODO: Installing fenicsx in dockerfile breaks MPI setup. We need to find a way to install fenicsx without breaking MPI. 
# Please install fenicsx manually in the container.

#RUN pip install fenics-basix
#RUN apt install -y fenicsx
RUN apt install -y ngsolve

WORKDIR /home/
CMD ["/bin/bash"]
