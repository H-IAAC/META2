import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import  json

def line_plots(exp_dict, title, path, y_label, x_label = 'Tarefa'):
    plt.plot([i for i in list(exp_dict.keys())], [i for i in list(exp_dict.values())])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(path)
    with open(os.path.join(os.path.dirname(os.path.dirname(path)), 'logs', os.path.split(path)[-1][:-4] + '.json'), 'w') as json_file:
        json.dump(exp_dict, json_file)

def bar_plots(exp_dict, title, path, filename, y_label, x_label = 'Classe'):
    num_exp = len(exp_dict)

    for i in range(num_exp):    
        plt.figure(figsize=(10,8))
        plt.bar(exp_dict[i].keys(), exp_dict[i].values())
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title + "na experiência " + str(i))
        plt.savefig(os.path.join(path, 'plots', filename + 'exp'+ str(i) + '.png'))
        with open(os.path.join(path, 'logs', filename + 'exp'+ str(i) + '.json'), 'w') as json_file:
            json.dump(exp_dict, json_file)

def save_test_stream_metrics(avalanche_metrics, sklearn_metrics, exp_dir):

    print("Saving test stream metrics...")
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok = True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok = True)
    #line plots
    acc_per_exp = {i: avalanche_metrics['Top1_Acc_Stream/eval_phase/test_stream/Task000'][1][i] for i in range(len(sklearn_metrics.classifications))}
    line_plots(acc_per_exp, "Acurácia de teste por tarefa", 
               os.path.join(exp_dir, 'plots', 'acc_per_exp.png'), "Acurácia")
    
    loss_per_exp = {i: avalanche_metrics['Loss_Stream/eval_phase/test_stream/Task000'][1][i] for i in range(len(sklearn_metrics.classifications))}
    line_plots(loss_per_exp, "Loss de teste por tarefa", 
               os.path.join(exp_dir, 'plots', 'loss_per_exp.png'), "Loss")
    
    forgetting_per_exp = {i: avalanche_metrics['StreamForgetting/eval_phase/test_stream'][1][i] for i in range(len(sklearn_metrics.classifications))}
    line_plots(forgetting_per_exp, "Forgetting de teste por tarefa", 
               os.path.join(exp_dir, 'plots', 'forgetting_per_exp.png'), "Forgetting")
    
    #f1 score macro and weighted exp: metric
    f1_macro_per_exp = {i: sklearn_metrics.classifications[i]['macro avg']['f1-score'] for i in range(len(sklearn_metrics.classifications))}
    line_plots(f1_macro_per_exp, "F1-macro de teste por tarefa", 
               os.path.join(exp_dir, 'plots', 'f1_macro.png'), "f1-macro")

    f1_weighted_per_exp = {i: sklearn_metrics.classifications[i]['weighted avg']['f1-score'] for i in range(len(sklearn_metrics.classifications))}
    line_plots(f1_weighted_per_exp, "F1-weighted de teste por tarefa", 
               os.path.join(exp_dir, 'plots', 'f1_weighted.png'), "f1-weighted")

    #heatmap plot per exp
    confusion_mtx_per_exp = {i: sklearn_metrics.confusion_matrices[i] for i in range(len(sklearn_metrics.classifications))}
    for i in confusion_mtx_per_exp:
        df_cm = pd.DataFrame(confusion_mtx_per_exp[i], index = [i for i in len(confusion_mtx_per_exp[i])],
                  columns = [i for i in len(confusion_mtx_per_exp[i])])
        plt.figure(figsize = (16,16))
        sns.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join(exp_dir, 'plots', + "confusion_mtx_task" + str(i) + ".png"))

    #save json per exp
    sklearn_metrics_per_exp = {i: sklearn_metrics.classifications[i] for i in range(len(sklearn_metrics.classifications))}
    for i in range(len(sklearn_metrics)):
        with open((os.path.join(exp_dir), 'logs', 'sklearn_metrics_per_exp' + str(i) + '.json'), 'w') as json_file:
            json.dump(sklearn_metrics_per_exp[i], json_file)

    #barplot per exp
    precision_per_classexp = {j: {int(i): sklearn_metrics.classifications[j][i]['precision'] for i in sklearn_metrics.classifications[j] if i.isnumeric()} for j in range(len(sklearn_metrics.classifications))}
    bar_plots(precision_per_classexp, "Precisão por classe", exp_dir, 'precision', 'Precisão')
    recall_per_classexp = {j: {int(i): sklearn_metrics.classifications[j][i]['recall'] for i in sklearn_metrics.classifications[j] if i.isnumeric()} for j in range(len(sklearn_metrics.classifications))}
    bar_plots(recall_per_classexp, "Recall por classe", exp_dir, 'recall', 'Recall')
    print("Done!")
    