FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update

RUN python -m pip install --upgrade pip

RUN pip install scikit-learn

RUN pip install tabulate

RUN pip install seaborn

RUN pip install tensorflow_addons

#VOLUME [ "./:/tmp" ]

#WORKDIR /tmp

#CMD ["bash"]

#podman-compose --podman-run-args="--security-opt=label=disable --gpus all --rm" run bg-track