import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scratch.datasets import DatasetFactory
from scratch.datasets.pamap import PAMAP2DataProcessor
from scratch.datasets.ucihar import UCIHARDataProcessor
from scratch.datasets.dsads import DSADSDataProcessor
from scratch.datasets.hapt import HAPTDataProcessor
from scratch.benchmarks.split import ClassSplit, SubjectSplit, SamplerInterface
from scratch.strategic.models import FullyConnectedNetwork, Microtransformer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':

    dataset = DatasetFactory.get_dataset('PAMAP2', 'PAMAP_5.2_20.cfg', sampler = SubjectSplit([[1, 2, 3, 4, 7, 8, 9], [5, 6]]), activities_to_use = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Microtransformer(31, 104, 12, 5, 0.2, device, FullyConnectedNetwork(input_shape=(1, 104, 31),
                                      hidden_layer_dimensions=[486, 243, 121],
                                      num_classes=12)).to(device)
    
    
    train_loader = torch.utils.data.DataLoader(dataset[0], batch_size = 32, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset[1], batch_size = 32)
    #training_loop

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = 5e-4, weight_decay = 1e-4)
    epochs = 100
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
        model.eval()
        with torch.no_grad():
            predictions = torch.empty((0), dtype=torch.int64).to(device)
            true = torch.empty((0), dtype=torch.int64).to(device)
            for inp, lab in test_loader:
                predictions = torch.cat((
                    predictions, torch.argmax(model(inp.to(device)), dim=1)))
                true = torch.cat((true, lab.to(device).to(torch.long)))
                
            print(classification_report(
                true.to('cpu'), predictions.to('cpu'), output_dict=True, zero_division = 0)['macro avg']['f1-score'])