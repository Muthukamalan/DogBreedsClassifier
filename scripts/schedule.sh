#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
dvc pull
python src/train.py experiment=hparams logger=csv
python src/eval.py
python src/inference.py --input_folder samples/inputs/ --output_folder samples/outputs/ --ckpt_path samples/checkpoints/epoch_019.ckpt 
gradio deploy