
help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


pull: ## pulling data from s3
	dvc pull 

fastrun: ## train simple mode to check
	HYDRA_FULL_ERROR=1 python src/train.py trainer.max_epochs=3 trainer.precision=16


test:  ## pytest and code-cov
	pytest --cov --cov-report=xml
	# coverage run -m pytest
	# pytest --cov=tests/
	# setup code-coverage
	coverage xml -o coverage.xml


train: ## hparams search and push optimal-result to s3
	HYDRA_FULL_ERROR=1 python src/train.py -m
	echo "Best Hparams"
	cat multirun/*/*/optimization_results.yaml
	echo "pushing to S3"
	aws s3 cp multirun/ s3://mhema-dog-breeds-bucket/training-$$(date +"%m-%d-%H%M%S")/ --recursive



eval: ## do test and save confusion-matrix
	HYDRA_FULL_ERROR=1 python src/eval.py 


inference:
	rm -rf samples/outputs/*
	HYDRA_FULL_ERROR=1 python src/inference.py --input_folder samples/inputs/ --output_folder samples/outputs/ --ckpt_path samples/checkpoints/epoch_019.ckpt 


trash: ## clean data/dataset and logs
	rm -rf data/dogs_dataset
	find . -type d \( -name '__pycache__' -o -name 'logs' -o -name 'outputs' -o -name 'multirun' \) -exec rm -rf {} +
	rm -rf samples/outputs/*
	rm assets/test_confusion_matrix.png assets/train_confusion_matrix.png assets/val_confusion_matrix.png
	
	

mshow:  ## tensorboard logs on hparams
	tensorboard --logdir multirun/ --load_fast=false --bind_all  &

sshow: ## tensorboard logs on fastrun
	tensorboard --logdir outputs/ --load_fast=false --bind_all &


showoff: ## turnoff tensorboard
	# kill -9 $(lsof -ti :6006)
	@PID=$$(lsof -ti :6006); \
	if [ -n "$$PID" ]; then \
		echo "Killing process $$PID"; \
		/usr/bin/kill -9 $$PID; \
	else \
		echo "No process found on port 6006"; \
	fi

clean: ## Clean autogenerated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

clean-logs: ## Clean logs
	rm -rf logs/**

format: ## Run pre-commit hooks
	pre-commit run -a

sync: ## Merge changes from main branch to your current branch
	git pull
	git pull origin main
