import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def line_plots(exp_dict, title, path, y_label, x_label='Tarefa', y_range=None):
    x = [i for i in list(exp_dict.keys())]
    y = [i for i in list(exp_dict.values())]
    plt.plot(x, y)
    plt.xticks(x)
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(path)
    
    plt.clf()


def bar_plots(exp_dict, title, path, filename, y_label, x_label='Classe', y_range=None):
    num_exp = len(exp_dict)

    for i in range(num_exp):
        plt.figure(figsize=(10, 8))
        plt.bar(exp_dict[i].keys(), exp_dict[i].values())
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title + " na experiência " + str(i))
        if y_range is not None:
            plt.ylim(y_range[0], y_range[1])
        plt.savefig(os.path.join(path, 'plots',
                    filename + 'exp' + str(i) + '.png'))
        
        plt.clf()


def save_test_stream_metrics(avalanche_metrics, sklearn_metrics, exp_dir):
    result_dict = dict()
    print("Saving test stream metrics...")
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    # line plots
    acc_per_exp = {i: sklearn_metrics.classifications[i]['accuracy']
                        for i in range(len(sklearn_metrics.classifications))}
    result_dict['Accuracy de teste por tarefa'] = list(acc_per_exp.values())
    line_plots(acc_per_exp, "Acurácia de teste por tarefa",
               os.path.join(exp_dir, 'plots', 'acc_per_exp.png'), "Acurácia", y_range=(0, 1.05))

    loss_per_exp = {i: avalanche_metrics['Loss_Stream/eval_phase/test_stream'][1][i]
                    for i in range(len(sklearn_metrics.classifications))}
    result_dict['Loss de teste por tarefa'] = list(loss_per_exp.values())
    line_plots(loss_per_exp, "Loss de teste por tarefa",
               os.path.join(exp_dir, 'plots', 'loss_per_exp.png'), "Loss")

    forgetting_per_exp = {i: avalanche_metrics['StreamForgetting/eval_phase/test_stream'][1][i]
                          for i in range(len(sklearn_metrics.classifications))}
    result_dict['Forgetting de teste por tarefa'] = list(forgetting_per_exp.values())
    line_plots(forgetting_per_exp, "Forgetting de teste por tarefa",
               os.path.join(exp_dir, 'plots', 'forgetting_per_exp.png'), "Forgetting")

    # f1 score macro, weighted and micro per exp: metric
    f1_macro_per_exp = {i: sklearn_metrics.classifications[i]['macro avg']
                        ['f1-score'] for i in range(len(sklearn_metrics.classifications))}
    result_dict['F1-macro de teste por tarefa'] = list(f1_macro_per_exp.values())
    line_plots(f1_macro_per_exp, "F1-macro de teste por tarefa",
               os.path.join(exp_dir, 'plots', 'f1_macro.png'), "f1-macro", y_range=(0, 1.05))
    
    f1_weighted_per_exp = {
        i: sklearn_metrics.classifications[i]['weighted avg']['f1-score'] for i in range(len(sklearn_metrics.classifications))}
    
    result_dict['F1-weighted de teste por tarefa'] = list(f1_weighted_per_exp.values())
    line_plots(f1_weighted_per_exp, "F1-weighted de teste por tarefa",
               os.path.join(exp_dir, 'plots', 'f1_weighted.png'), "f1-weighted", y_range=(0, 1.05))

    f1_micro_per_exp = {i: sklearn_metrics.classifications[i]['F1-micro']
                         for i in range(len(sklearn_metrics.classifications))}
    result_dict['F1-micro de teste por tarefa'] = list(f1_micro_per_exp.values())
    line_plots(f1_micro_per_exp, "F1-micro de teste por tarefa",
               os.path.join(exp_dir, 'plots', 'f1_micro.png'), "f1-micro", y_range=(0, 1.05))

    # heatmap plot per exp
    
    confusion_mtx_per_exp = {i: sklearn_metrics.confusion_matrices[i].tolist() for i in range(
        len(sklearn_metrics.classifications))}
    result_dict['Matriz de confusão por experiência'] = confusion_mtx_per_exp
    for i in confusion_mtx_per_exp:
        df_cm = pd.DataFrame(confusion_mtx_per_exp[i], index=[i for i in range(len(confusion_mtx_per_exp[i]))],
                             columns=[i for i in range(len(confusion_mtx_per_exp[i]))])
        plt.figure(figsize=(16, 16))
        plt.title(f"Matriz de confusão na experiência {i}")
        sns.heatmap(df_cm, annot=True)
        plt.savefig(os.path.join(exp_dir, 'plots',
                    "confusion_mtx_task" + str(i) + ".png"))
        plt.clf()
    
    # save json per exp
    sklearn_metrics_per_exp = {i: sklearn_metrics.classifications[i] for i in range(
        len(sklearn_metrics.classifications))}
    
    result_dict['SKlearn metrics'] = sklearn_metrics_per_exp

    # barplot per exp
    precision_per_classexp = {j: {int(i): sklearn_metrics.classifications[j][i]['precision'] for i in sklearn_metrics.classifications[j] if i.isnumeric(
    )} for j in range(len(sklearn_metrics.classifications))}
    result_dict['Precisão por classe por experiência'] = precision_per_classexp
    bar_plots(precision_per_classexp, "Precisão por classe",
              exp_dir, 'precision', 'Precisão', y_range=(0, 1.05))
    recall_per_classexp = {j: {int(i): sklearn_metrics.classifications[j][i]['recall'] for i in sklearn_metrics.classifications[j] if i.isnumeric(
    )} for j in range(len(sklearn_metrics.classifications))}
    result_dict['Recall por classe por experiência'] = recall_per_classexp
    bar_plots(recall_per_classexp, "Recall por classe",
              exp_dir, 'recall', 'Recall', y_range=(0, 1.05))
    print("Done!")
    return result_dict


