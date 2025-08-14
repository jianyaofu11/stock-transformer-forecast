import torch
from torch.utils.data import Dataset

class MultiStockDataset(Dataset):
    """PyTorch Dataset for multi-stock time series data"""
    
    def __init__(self, processed_data):
        self.sequences = []
        self.targets = []
        self.symbols = []
        
        for data in processed_data:
            for i, (seq, target) in enumerate(zip(data['sequences'], data['targets'])):
                self.sequences.append(seq)
                self.targets.append(target)
                self.symbols.append(data['symbol'])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx]),
            self.symbols[idx]
        )
