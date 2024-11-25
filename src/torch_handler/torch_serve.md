
```bash
torch-model-archiver --model-name mamba_out --version 1.0 \
    --serialized-file "./samples/checkpoints/mambaout.pt" \
    --handler "./src/torch_handler/mambaout_handler.py" \
    --extra-files "./src/torch_handler/mamba_classes/index_to_name.json"
```


```bash
# .mar file
|_ MAR-INF/MANIFEST.json
|_ index_to_name.json
|_ mambaout.pt
|_ mamabaout_handler.py
```


#### run on local

```bash
mkdir model_store/
mv .mar model_store/
#disable token
torchserve  --start --ncs  --model-store model_store  --disable-token-auth  --models all--ts-config /home/muthu/GitHub/Spaces/DogBreedsClassifier/src/torch_handler/config.properties --models mamba_out=./model_store/mamba_out.mar
torchserve --stop
```
<!-- #### create docker image in torchsever
and keep --net=host
```bash
docker run -it --rm --net=host -v `pwd`:/opt/src pytorch/torchserve:latest bash

cd /opt/src/
cd /model-store/".mar-file"
``` -->



```bash 
curl -X OPTIONS http://localhost:8080/ > swagger.json

curl "http://localhost:8081/models"
curl "http://localhost:8081/models/mamba_out"

curl -v -X PUT http://localhost:8081/models/mamba_out/1.0/set-default


curl http://127.0.0.1:8080/predictions/mamba_out -T samples/inputs/guess1.jpg 
```


## settin up gRPC Protocol
```bash
# Clone the following repo and install the dependencies
git clone https://github.com/pytorch/serve
cd serve
pip install -U grpcio protobuf grpcio-tools

# Create gRPC stubs for python using
python -m grpc_tools.protoc --proto_path=frontend/server/src/main/resources/proto/ --python_out=ts_scripts --grpc_python_out=ts_scripts frontend/server/src/main/resources/proto/inference.proto frontend/server/src/main/resources/proto/management.proto

# Test the gRPC inference using
python ts_scripts/torchserve_grpc_client.py infer mamba_out .samples/inputs/guess1.jpg 
```

## TorchServe Deployment
```bash
# Run the pytest test cases using
pytest --public_ip 3.108.221.28 --model cifar10_model --grpc_client /home/ubuntu/serve/ts_scripts/torchserve_grpc_client.py tests/test_serve/ -s
```