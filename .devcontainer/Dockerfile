# Pytorch
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime


# If the steps of a `Dockerfile` use files that are different from the `context` file, COPY the
# file of each step separately; and RUN the file immediately after COPY
WORKDIR /app
COPY /.devcontainer/requirements.txt /app


# NLTK
# installation: https://www.nltk.org/install.html
# data: https://www.nltk.org/data.html
RUN apt update && apt -q -y upgrade && apt -y install sudo && sudo apt -y install graphviz && \
    pip install --upgrade pip && \
	pip install --requirement /app/requirements.txt --no-cache-dir && \
    python -m nltk.downloader -d /opt/conda/nltk_data all


# Port
# TensorBoard: 6006, 6007
# Ray: 8265, 6379
EXPOSE 6006 6007 8265 6379 5000


# ENTRYPOINT
ENTRYPOINT ["bash"]
