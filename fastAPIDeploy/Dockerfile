FROM python:3.9-slim-buster AS build

# FROM public.ecr.aws/docker/library/python:3.12.0-slim-bullseye
# COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter
ENV PORT=8080

WORKDIR /var/task

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
COPY . .
CMD python main.py
EXPOSE 8080