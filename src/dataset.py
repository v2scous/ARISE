# src/dataset.py

import torch
from torch.utils.data import Dataset

class EmbeddingSequenceDataset(Dataset):
    """
    dat: pandas.DataFrame
         (component column + temperature + target)
    embedding_table: initial embedding table of each component
                     (DataFrame: columns = components, rows = initial embedding dimension)
    max_len: maximum length of sequence (component upper boundary)
    """
    def __init__(self, dat, embedding_table, max_len=15):
        # dat: pandas.DataFrame
        self.dat = dat.reset_index(drop=True)
        self.component = list(embedding_table.columns)
        self.comp_size = len(self.component)
        self.embedding_dim = len(embedding_table.iloc[:, 0])
        self.max_len = max_len

        # embedding table for each component
        self.embedding_dict = {
            ch: torch.tensor(embedding_table[ch].values, dtype=torch.float32)
            for ch in self.component
        }

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        sample = self.dat.iloc[idx].values
        # mole fraction of components
        comp_fraction = sample[:self.comp_size]
        # temperature
        temperature = sample[self.comp_size]
        # target property
        target = sample[self.comp_size + 1]

        tokens = []
        for i, fraction in enumerate(comp_fraction):
            if fraction > 0:
                comp = self.component[i]
                comp_emb = self.embedding_dict[comp]
                token = torch.cat(
                    [comp_emb, torch.tensor([fraction, temperature], dtype=comp_emb.dtype)]
                )
                tokens.append(token)

        actual_len = len(tokens)

        # padding
        if actual_len < self.max_len:
            pad_token = torch.zeros(self.embedding_dim + 2, dtype=torch.float32)
            tokens.extend([pad_token] * (self.max_len - actual_len))

        tokens = torch.stack(tokens[: self.max_len])  # (max_len, embedding_dim + 2)
        mask = torch.tensor(
            [1] * min(actual_len, self.max_len)
            + [0] * max(0, self.max_len - actual_len),
            dtype=torch.long,
        )

        return {
            "input": tokens,           # (max_len, embedding_dim + 2)
            "attention_mask": mask,    # (max_len,)
            "target": torch.tensor(target, dtype=torch.float32),
        }

