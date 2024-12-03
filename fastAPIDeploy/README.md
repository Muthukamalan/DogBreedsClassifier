```bash
source .venv/bin/activate
python -m pip install -r aws-reqs.txt
cdk bootstrap
cdk deploy  --verbose --progress bar
cdk destroy
```