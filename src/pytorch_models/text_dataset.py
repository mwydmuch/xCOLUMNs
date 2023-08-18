import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from random import randrange


class TextDataset(Dataset):
    """
    Dataset that wraps data in text format and use tokenizer.
    """
    def __init__(self, 
                input: list[str], 
                target: csr_matrix = None,
                tokenizer = None,
                max_seq_length: int = 512,
                lazy_encode=True, 
                target_dense_vec=True):
        super().__init__()
        self.input = input
        self.target = target
        self.lazy_encode = lazy_encode
        self.target_dense_vec = target_dense_vec

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if not self.lazy_encode:
            print("Tokenizing dataset ...")
            self.encodings = self.tokenize(self.input)

        self.encodings = None

    @staticmethod
    def prepare_encodings(encodings, idx):
        return {key: val[idx] for key, val in encodings.items()}
    
    @staticmethod
    def sparse_to_dense_tensor(csr_vec, shape=None, dtype=torch.float32):
        shape = csr_vec.shape if shape is None else shape
        tensor = torch.zeros(shape, dtype=dtype)
        tensor[csr_vec.indices] = torch.tensor(csr_vec.data, dtype=dtype)
        return tensor

    def tokenize(self, text):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer.batch_encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
    
    def setup(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if not self.lazy_encode:
            print("Tokenizing dataset ...")
            self.encodings = self.tokenize(self.input)

    def __len__(self):
        return len(self.input)

    def target_size(self):
        return self.target.shape[1]

    def __getitem__(self, idx):
        if self.encodings:
            item = TextDataset.prepare_encodings(self.encodings, idx)
        else:
            item = TextDataset.prepare_encodings(self.tokenize([self.input[idx]]), 0)

        if self.target is not None:
            target_idx = self.target[idx]
            item["target"] = TextDataset.sparse_to_dense_tensor(target_idx, self.target.shape[1])
        
        return item