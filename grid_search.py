import subprocess
from tqdm import tqdm

def run_experiment(command):
    subprocess.run(command, capture_output=True)


'''best_hp = [{'lr': 0.0005, 'decay': 0.0001, 'file': 'pamap_waadb.cfg'},
           {'lr': 0.0005, 'decay': 0.00001, 'file': 'pamap_wamdf.cfg'}]

commands = []
for i in range(7):
    for best in best_hp:
        commands.append(['python', '.\\Codigos\\run_experiment.py', '--learning_rate', str(best['lr']), '--training_epochs', '25',
                    '--weight_decay', str(best['decay']), '--experiment_file', str(best['file'])])'''


commands = []
datasets = ['hapt']
strategies = ['wamdf']
lrs = [5e-5]
decays = [0.0001]

for _ in range(17):
    for dataset in datasets:
        for strategy in strategies:
            for lr in lrs:
                for decay in decays:
                        commands.append(['python', '.\\Codigos\\run_experiment.py', '--learning_rate', str(lr), '--training_epochs', '20',
                            '--weight_decay', str(decay), '--experiment_file', f'{dataset}_{strategy}.cfg'])

for command in tqdm(commands):
    
    run_experiment(command)
