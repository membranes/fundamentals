# TensorFlow
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime


# Set application directory <app>
WORKDIR /app


# This command copies requirements.txt into /app
COPY /.devcontainer/requirements.txt /app

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt --no-cache


# Port
EXPOSE 8050


# ENTRYPOINT
ENTRYPOINT ["bash"]