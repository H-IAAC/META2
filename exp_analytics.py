import json 
import os 
import numpy as np
from scratch.strategic import CEKDLossPlugin, FullyConnectedNetwork
import matplotlib.pyplot as plt

def get_best(scenario, strat, criteria, *params_to_compare):
    folder = "../Resultados/" + scenario + '/' + strat
    exp_dict = dict()
    exp_grouped_dict = dict()
    for i in os.listdir(folder):
        
        with open(os.path.join(folder, i, 'metadata.json')) as new_file:
            file_contents = new_file.read()
            exp_dict[i] = json.loads(file_contents)

    for i in exp_dict.values():
        #dict lr, decay
        key = tuple([i['training_parameters'][j] for j in params_to_compare])
        
        if key not in exp_grouped_dict.keys():
            exp_grouped_dict[key] = [i['results'][criteria][-1]]
        
        else:
            exp_grouped_dict[key] += [i['results'][criteria][-1]]
            
    
    for j in exp_grouped_dict:
        
        exp_grouped_dict[j] = np.mean(exp_grouped_dict[j])
    print("Sequência de parâmetros: ", [j for j in params_to_compare])
    print("Melhor grupo de parâmetros é:",sorted(exp_grouped_dict.items(), key=lambda x: x[1])[-1])
    return {(scenario, strat): sorted(exp_grouped_dict.items(), key=lambda x: x[1])[-1]}

def retrieve_values(scenario, strat, criteria, param_values):
    folder = "../Resultados/" + scenario + '/' + strat
    exp_dict = dict()
    exp_grouped_dict = dict()

    if len(param_values) == 2:
        params_to_compare = ['learning_rate', 'weight_decay']
    else:
        params_to_compare = ['learning_rate', 'weight_decay', 'plasticity_factor']

    for i in os.listdir(folder):
        
        with open(os.path.join(folder, i, 'metadata.json')) as new_file:
            file_contents = new_file.read()
            exp_dict[i] = json.loads(file_contents)

    for i in exp_dict.values():
        #dict lr, decay
        key = tuple([i['training_parameters'][j] for j in params_to_compare])
        
        if key not in exp_grouped_dict.keys():
            exp_grouped_dict[key] = [i['results'][criteria]]
        
        else:
            exp_grouped_dict[key] += [i['results'][criteria]]

    return exp_grouped_dict[param_values]

if __name__ == '__main__':
    best_param_dict = dict()
    for scenario in ["DSADS_TI", "HAPT_TI", "UCIHAR_TI", "PAMAP_TI"]:
        for strat in ["naive", 'waadb', 'wamdf']:
            print(f"Results below for: {scenario, strat}")
            best_param_dict.update(get_best(scenario, strat, "F1-macro de teste por tarefa", 'learning_rate', 'weight_decay'))

    for scenario in ["DSADS_TI", "HAPT_TI", "UCIHAR_TI", "PAMAP_TI"]:
        for strat in ['waadb_plasticity', 'wamdf_plasticity']:
            print(f"Results below for: {scenario, strat}")
            best_param_dict.update(get_best(scenario, strat, "F1-macro de teste por tarefa", 'learning_rate', 'weight_decay', 'plasticity_factor'))
    
    bottom_limits = {"DSADS_TI": 0.02, "HAPT_TI": 0.023, "UCIHAR_TI": 0.225, "PAMAP_TI": 0.0449}
    upper_limits = {"DSADS_TI": 0.7787, "HAPT_TI": 0.7084, "UCIHAR_TI": 0.8894, "PAMAP_TI": 0.8210}
    
    for scenario in ["DSADS_TI", "HAPT_TI", "UCIHAR_TI", "PAMAP_TI"]:
        
        best_params = [(i[1], best_param_dict[i]) for i in best_param_dict if scenario in i]
        plt.figure(figsize = (12, 12))
        xsize = 10
        for j in best_params:
            if "naive" not in j[0]: 
                all_matches = np.array(retrieve_values(scenario, j[0], "F1-macro de teste por tarefa", j[1][0]))
                mean = np.mean(all_matches, axis = 0)
                std = np.std(all_matches, axis = 0)
                xsize = mean.shape[0]
                plt.errorbar([i for i in range(mean.shape[0])], list(mean), list(std), label = j[0], marker = '*')
        plt.plot([i for i in range(xsize)], [bottom_limits[scenario] for i in range(xsize)], label = "Bottom limit")
        plt.plot([i for i in range(xsize)], [upper_limits[scenario] for i in range(xsize)], label = "Upper limit")
        plt.ylim(0, 1)
        plt.xlim(0 - 0.3, xsize - 0.7)

        plt.xlabel("Experiência", fontsize = 20)
        plt.ylabel("F1-Macro de teste", fontsize = 20)

        plt.xticks([i for i in range(xsize)], fontsize = '15')
        plt.yticks(np.arange(0, 1, step = 0.1), fontsize = '15')