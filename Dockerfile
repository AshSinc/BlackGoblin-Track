FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update

RUN python -m pip install --upgrade pip

RUN pip install scikit-learn

RUN pip install tabulate

RUN pip install seaborn

RUN pip install tensorflow_addons