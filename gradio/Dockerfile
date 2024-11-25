FROM public.ecr.aws/docker/library/python:3.11.10-slim

# Install AWS Lambda Web Adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /var/task

# Copy and install requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy application code and model
COPY app.py ./
COPY best_model.pt ./
COPY examples/ ./examples/

# Set command
CMD ["python3", "app.py"] 