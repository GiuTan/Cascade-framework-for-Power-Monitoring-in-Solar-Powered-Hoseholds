import torch
import numpy as np


class NILMDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, seq_len, pred_len, stride):
        self.x = x
        self.y = y
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride

    def __len__(self):
        return int(np.ceil((len(self.x) - self.seq_len) / self.stride) + 1)

    def __getitem__(self, index):
        start_index = index * self.stride
        end_index = np.min((len(self.x), index * self.stride + self.seq_len))        
        x = self.padding_seqs(self.x[start_index:end_index])
        y = self.padding_seqs(self.y[start_index:end_index])
        y = y[-self.pred_len:]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    def padding_seqs(self, in_array):
        if len(in_array) == self.seq_len:
            return in_array
        try:
            out_array = np.zeros((self.seq_len, in_array.shape[1]))
        except:
            out_array = np.zeros(self.seq_len)

        out_array[:len(in_array)] = in_array
        return out_array
