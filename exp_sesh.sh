#!/bin/bash
for i in 1 2 3 4 5; do
  
  python run_experiment.py --learning_rate 0.005 --training_epochs 20 --weight_decay 0.0001 --meta_plasticity 0.7 --experiment_file ucihar_wamdf_metaplas_lora.cfg
  
done