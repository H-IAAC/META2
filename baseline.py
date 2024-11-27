import torch
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset
from scratch.datasets import DatasetFactory
from scratch.datasets.pamap import PAMAP2DataProcessor
from scratch.datasets.ucihar import UCIHARDataProcessor
from scratch.datasets.dsads import DSADSDataProcessor
from scratch.datasets.hapt import HAPTDataProcessor
from scratch.benchmarks.split import ClassSplit, SubjectSplit, SamplerInterface
from scratch.strategic.models import FullyConnectedNetwork, Microtransformer, HARTransformer, CrossAttnHARTransformer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report, confusion_matrix

class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr, decay_rate, decay_start_step, last_epoch=-1):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_start_step = decay_start_step
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.decay_start_step:
            return [self.initial_lr for _ in self.base_lrs]
        else:
            return [
                base_lr * (self.decay_rate ** (self.last_epoch - self.decay_start_step))
                for base_lr in self.base_lrs
            ]

if __name__ == '__main__':

    dataset = DatasetFactory.get_dataset('HAPT', 'HAPT_2.56_50.cfg', sampler = SubjectSplit([[1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30], [2, 4, 9, 10, 12, 13, 18, 20, 24]]), activities_to_use = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CrossAttnHARTransformer((1, 128, 6), 4, 12, dropout=0.15, sensor_group=3, wordsize=128)

    '''model = FullyConnectedNetwork(input_shape=(1, 104, 27),
                                      hidden_layer_dimensions=[486, 243, 121],
                                      num_classes=12)'''
    model.cuda()
    
    train_loader = torch.utils.data.DataLoader(dataset[0], batch_size = 32, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset[1], batch_size = 32)
    #training_loop

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-4)
    epochs = 100

    initial_lr = 1e-3
    decay_rate = 0.93
    decay_start_step = 15
    scheduler = CustomLRScheduler(optimizer, initial_lr, decay_rate, decay_start_step)
    vals = []
    trains = []


    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device).to(torch.long)
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            predictions = torch.empty((0), dtype=torch.int64).to(device)
            true = torch.empty((0), dtype=torch.int64).to(device)
            for inp, lab in test_loader:
                predictions = torch.cat((
                    predictions, torch.argmax(model(inp.to(device)), dim=1)))
                true = torch.cat((true, lab.to(device).to(torch.long)))

            val_f1 =  classification_report(
                true.to('cpu'), predictions.to('cpu'), output_dict=True, zero_division = 0)['macro avg']['f1-score']  
            
            for inp, lab in train_loader:
                predictions = torch.cat((
                    predictions, torch.argmax(model(inp.to(device)), dim=1)))
                true = torch.cat((true, lab.to(device).to(torch.long)))

            train_f1 =  classification_report(
                true.to('cpu'), predictions.to('cpu'), output_dict=True, zero_division = 0)['macro avg']['f1-score']  
            
            trains.append(train_f1)
            vals.append(val_f1)
            print(f"Epoch {epoch}, train: {train_f1}, test: {val_f1}")
    print(f"FINALLY Train {np.mean(trains[-10:])}, val {np.mean(vals[-10:])}")