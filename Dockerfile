# torch==2.5.0+cu121
FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime 
WORKDIR /code
COPY requirements.txt .
RUN pip install -r requirements.txt 
COPY . .

# tensorboard port
EXPOSE 6006 
