#!/bin/bash
for i in 1 2 3 4 5; do
  python run_experiment.py --learning_rate 0.001 --training_epochs 20 --weight_decay 0.0001 --experiment_file pamap_wamdf.cfg
  python run_experiment.py --learning_rate 0.001 --training_epochs 20 --weight_decay 0.0001 --plasticity_factor 0.8 --experiment_file pamap_wamdf_plas.cfg
  python run_experiment.py --learning_rate 0.0005 --training_epochs 20 --weight_decay 0.0005 --experiment_file pamap_waadb.cfg
  python run_experiment.py --learning_rate 0.0005 --training_epochs 20 --weight_decay 0.0005 --plasticity_factor 0.8 --experiment_file pamap_waadb_plas.cfg

  python run_experiment.py --learning_rate 0.0005 --training_epochs 20 --weight_decay 0.0001 --experiment_file ucihar_wamdf.cfg
  python run_experiment.py --learning_rate 0.0005 --training_epochs 20 --weight_decay 0.0001 --plasticity_factor 0.8 --experiment_file ucihar_wamdf_plas.cfg
  python run_experiment.py --learning_rate 0.001 --training_epochs 20 --weight_decay 0.0001 --experiment_file ucihar_waadb.cfg
  python run_experiment.py --learning_rate 0.001 --training_epochs 20 --weight_decay 0.0001 --plasticity_factor 0.8 --experiment_file ucihar_waadb_plas.cfg

  python run_experiment.py --learning_rate 0.00005 --training_epochs 20 --weight_decay 0.0001 --experiment_file dsads_wamdf.cfg
  python run_experiment.py --learning_rate 0.00005 --training_epochs 20 --weight_decay 0.0001 --plasticity_factor 0.8 --experiment_file dsads_wamdf_plas.cfg
  python run_experiment.py --learning_rate 0.0001 --training_epochs 20 --weight_decay 0.0001 --experiment_file dsads_waadb.cfg
  python run_experiment.py --learning_rate 0.0001 --training_epochs 20 --weight_decay 0.0001 --plasticity_factor 0.8 --experiment_file dsads_waadb_plas.cfg

  python run_experiment.py --learning_rate 0.001 --training_epochs 20 --weight_decay 0.0001 --experiment_file hapt_wamdf.cfg
  python run_experiment.py --learning_rate 0.001 --training_epochs 20 --weight_decay 0.0001 --plasticity_factor 0.8 --experiment_file hapt_wamdf_plas.cfg
  python run_experiment.py --learning_rate 0.00005 --training_epochs 20 --weight_decay 0.001 --experiment_file hapt_waadb.cfg
  python run_experiment.py --learning_rate 0.00005 --training_epochs 20 --weight_decay 0.001 --plasticity_factor 0.8 --experiment_file hapt_waadb_plas.cfg
done