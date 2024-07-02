#!/bin/bash
for i in 1 2 3; do
  python run_experiment.py --learning_rate 0.001 --training_epochs 10 --weight_decay 0.0001
done