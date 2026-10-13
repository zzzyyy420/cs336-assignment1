import random
import numpy as np
from typing import List
import torch

class DataLoader:
    def __init__(self, data:List[int] | np.ndarray, batch_size:int, context_length:int, shuffle=True, device="cpu"):
        if batch_size <= 0 or context_length <= 0:
            raise ValueError("batch_size and context_length must be positive")
        if len(data) <= context_length:
            raise ValueError("data must contain at least context_length + 1 tokens")
        self.data = data
        self.data_len = len(data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.context_length = context_length
        self.device = device
    # def __iter__(self):
    #     if self.shuffle:
    #         self.data = random.shuffle(self.data)
    #     for i in range(0, len(self.data), self.batch_size):
    #         yield self.data[i:i+self.batch_size]

    def get_train_batch_data(self):
        
        idxs = np.random.randint(0, self.data_len - self.context_length, size=self.batch_size)
        x = np.stack([self.data[i:i+self.context_length] for i in idxs])
        y = np.stack([self.data[i+1:i+self.context_length+1] for i in idxs])
        return (
            torch.tensor(x, dtype=torch.long, device=self.device),
            torch.tensor(y, dtype=torch.long, device=self.device),
        )
    
    def get_valid_batch_data_iter(self):
        
        for batch_idx in range(len(self)):
            bias = batch_idx * self.batch_size
            idxs = range(bias, bias + self.batch_size)
            x = np.stack([self.data[i:i+self.context_length] for i in idxs])
            y = np.stack([self.data[i+1:i+self.context_length+1] for i in idxs])
            yield (
                torch.tensor(x, dtype=torch.long, device=self.device),
                torch.tensor(y, dtype=torch.long, device=self.device),
            )

    def __len__(self):

        return (self.data_len - self.context_length) // self.batch_size
