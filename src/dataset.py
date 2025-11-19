# src/dataset.py

import torch
from torch.utils.data import Dataset

class EmbeddingSequenceDataset(Dataset):
    """
    dat: pandas.DataFrame
         (26개 component column + sentence_feature(예: 온도 등) + target)
    embedding_table: 각 component에 대한 임베딩 테이블
                     (DataFrame: columns = component 이름, rows = 임베딩 차원)
    max_len: 시퀀스 최대 길이 (component 개수 상한)
    """
    def __init__(self, dat, embedding_table, max_len=15):
        # dat: pandas.DataFrame
        self.dat = dat.reset_index(drop=True)
        self.vocab = list(embedding_table.columns)
        self.vocab_size = len(self.vocab)
        self.embedding_dim = len(embedding_table.iloc[:, 0])
        self.max_len = max_len

        # component별 임베딩 딕셔너리 생성
        self.embedding_dict = {
            ch: torch.tensor(embedding_table[ch].values, dtype=torch.float32)
            for ch in self.vocab
        }

    def __len__(self):
        return len(self.dat)

    def __getitem__(self, idx):
        sample = self.dat.iloc[idx].values
        # 앞 vocab_size개의 값: 각 component의 비율(예: mole fraction)
        word_ratios = sample[:self.vocab_size]
        # 그 다음 1개: sentence_feature (예: 온도)
        sentence_feature = sample[self.vocab_size]
        # 마지막: target 값 (예: log(viscosity))
        target = sample[self.vocab_size + 1]

        tokens = []
        for i, ratio in enumerate(word_ratios):
            if ratio > 0:
                word = self.vocab[i]
                word_emb = self.embedding_dict[word]
                token = torch.cat(
                    [word_emb, torch.tensor([ratio, sentence_feature], dtype=word_emb.dtype)]
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
