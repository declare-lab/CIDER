import pandas as pd
from torch.utils.data import Dataset, DataLoader

class NLIDataset(Dataset):
    def __init__(self, filename, mode):
        pairs, labels = [], []
        if mode  == '0':
            ## 0: roberta-mnli ##
            mapping = {"contradiction": 0, "neutral": 1, "entailment": 2}
        elif mode == '1':
            ## 1: roberta-mnli-anli-fever ##
            mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
        elif mode  == '2':
            ## 2: roberta ##
            mapping = {"entailment": 0, "contradiction": 1}
        
        with open(filename) as f:
            for line in f:
                content = line.strip().split('\t')
                pairs.append([content[0], " ".join([content[1], content[2], content[3]])]) 
                labels.append(mapping[content[4]])
                
        self.pairs = pairs
        self.labels = labels
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index): 
        p = self.pairs[index]
        l = self.labels[index]
        return p, l
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
def NLILoader(filename, mode, batch_size, shuffle):
    dataset = NLIDataset(filename, mode)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return loader
