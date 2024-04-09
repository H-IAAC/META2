#!/bin/bash
for i in 1 2 3; do
  python3 run_experiment.py --learning_rate 0.001 --training_epochs 10 --weight_decay 0.0001
  python3 run_experiment.py --learning_rate 0.0005 --training_epochs 10 --weight_decay 0.0001
  python3 run_experiment.py --learning_rate 0.0001 --training_epochs 10 --weight_decay 0.0001
  python3 run_experiment.py --learning_rate 0.00005 --training_epochs 10 --weight_decay 0.0001
  python3 run_experiment.py --learning_rate 0.001 --training_epochs 10 --weight_decay 0.0005
  python3 run_experiment.py --learning_rate 0.0005 --training_epochs 10 --weight_decay 0.0005
  python3 run_experiment.py --learning_rate 0.0001 --training_epochs 10 --weight_decay 0.0005
  python3 run_experiment.py --learning_rate 0.00005 --training_epochs 10 --weight_decay 0.0005
  python3 run_experiment.py --learning_rate 0.001 --training_epochs 10 --weight_decay 0.001
  python3 run_experiment.py --learning_rate 0.0005 --training_epochs 10 --weight_decay 0.001
  python3 run_experiment.py --learning_rate 0.0001 --training_epochs 10 --weight_decay 0.001
  python3 run_experiment.py --learning_rate 0.00005 --training_epochs 10 --weight_decay 0.001
done