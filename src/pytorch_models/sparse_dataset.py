import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from random import randrange


# TODO: Move some static functions to utils

class SparseDataset(Dataset):
    """
    Dataset that wraps data in sparse csr_matrix format.
    """
    def __init__(
        self,
        input: csr_matrix,
        target: csr_matrix = None,
        original_target: csr_matrix = None,
        norm_input: str = "l2",
        input_dense_vec: bool = True,  # Include also dense vector representation of input
        target_dense_vec: bool = True,  # Include also dense vector representation of target
        target_negative_samples: int = 0,  # Number of negative samples to add to target
    ):
        super().__init__()
        
        if not isinstance(input, csr_matrix):
            raise ValueError(f"Unsupported type {type(input)}, use scipy.sparse.csr_matrix")

        if target is not None and not isinstance(target, csr_matrix):
            raise ValueError(f"Unsupported type {type(target)}, use scipy.sparse.csr_matrix")

        if original_target is not None and not isinstance(original_target, csr_matrix):
            raise ValueError(f"Unsupported type {type(original_target)}, use scipy.sparse.csr_matrix")
        
        if target is not None:
            assert input.shape[0] == target.shape[0]

        if original_target is not None:
            assert input.shape[0] == original_target.shape[0]

        if norm_input is not None:
            input = normalize(input, norm=norm_input, axis=1)

        self.input = input
        self.input_dense_vec = input_dense_vec
        self.target = target
        self.target_dense_vec = target_dense_vec
        self.target_negative_samples = target_negative_samples
        self.original_target = original_target

        print(f"Initializing SparseDataset with input shape={self.input.shape} as {'dense' if input_dense_vec else 'sparse'}", end="")
        if self.target is not None:
            print(f", target shape={target.shape} as {'dense' if target_dense_vec else 'sparse'}")
        else:
            print("")

    # Collate related
    @staticmethod
    def collate_sequence(items, prefix, batch_first=True, mask=False, padding_value=0):
        ids_key = f"{prefix}_ids"
        values_key = f"{prefix}_values"
        # len_key = f"{prefix}_len"
        seq_batch = {
            ids_key: pad_sequence(
                [i[ids_key] for i in items],
                batch_first=batch_first,
                padding_value=padding_value,
            ),
            values_key: pad_sequence(
                [i[values_key] for i in items],
                batch_first=batch_first,
                padding_value=padding_value,
            ),
            # len_key: torch.LongTensor([i[len_key] for i in items])
        }

        if mask:
            mask_key = f"{prefix}_mask"
            seq_batch[mask_key] = (seq_batch[ids_key] > 0).type(torch.float32)

        return seq_batch

    @staticmethod
    def collate_dense_vec(items, key):
        return torch.vstack([i[key] for i in items])

    @staticmethod
    def collate_fn(items):
        batch = {}
        for key in ["input", "target", "original_target"]:
            if key + "_ids" in items[0]:
                batch.update(SparseDataset.collate_sequence(items, key, mask=True))
            if key in items[0]:
                batch[key] = SparseDataset.collate_dense_vec(items, key)
                
        return batch

    def collate_function(self):
        return SparseDataset.collate_fn

    def __len__(self):
        return self.input.shape[0]

    def input_size(self):
        return self.input.shape[1]

    def target_size(self):
        return self.target.shape[1]
    
    def original_target_size(self):
        return self.original_target.shape[1]

    @staticmethod
    def get_seq_data(data, prefix):
        return {
            f"{prefix}_ids": torch.LongTensor(data.indices),
            f"{prefix}_values": torch.FloatTensor(data.data),
            # f'{prefix}_len': data.indices.shape[0]
        }

    @staticmethod
    def sparse_to_dense_tensor(csr_vec, shape=None, dtype=torch.float32):
        shape = csr_vec.shape if shape is None else shape
        tensor = torch.zeros(shape, dtype=dtype)
        tensor[csr_vec.indices] = torch.tensor(csr_vec.data, dtype=dtype)
        return tensor
    
    def add_negative_samples(csr_vec, num_negative_samples, max_samples=None):
        num_positive_samples = csr_vec.indices.shape[0]
        all_samples = num_positive_samples + num_negative_samples
        if max_samples is not None:
            all_samples = min(all_samples, max_samples)
        num_labels = csr_vec.shape[1]
        
        indptr = csr_vec.indptr.copy()
        indptr[1] = all_samples
        indices = csr_vec.indices.copy()
        indices.resize(all_samples)
        data = csr_vec.data.copy()
        data.resize(all_samples)

        # TODO: This is very slow and naive. 1) Optimize it with Numba, 2) Use a better sampling method
        for i in range(num_positive_samples, all_samples):
            while True:
                l = randrange(0, num_labels)
                if l not in indices[:i]:
                    break
            indices[i] = l
            data[i] = 0
        
        return csr_matrix((data, indices, indptr), shape=csr_vec.shape)

    def __getitem__(self, idx):
        input_idx = self.input[idx]
        item = SparseDataset.get_seq_data(input_idx, "input")  # Return as padded sequence
        if self.input_dense_vec:  # Return as dense vec
            item["input"] = SparseDataset.sparse_to_dense_tensor(input_idx, self.input.shape[1])

        if self.target is not None:
            target_idx = self.target[idx]
            if self.target_negative_samples > 0:
                target_idx = SparseDataset.add_negative_samples(target_idx, self.target_negative_samples, self.target.shape[1])

            item.update(SparseDataset.get_seq_data(target_idx, "target"))  # Return as padded sequence with mask
            if self.target_dense_vec:  # Add Return as dense vec
                item["target"] = SparseDataset.sparse_to_dense_tensor(target_idx, self.target.shape[1])
        
        if self.original_target is not None:
            original_target_idx = self.original_target[idx]
            item.update(SparseDataset.get_seq_data(self.original_target[idx], "original_target"))
            if self.target_dense_vec:  # Add Return as dense vec
                item["original_target"] = SparseDataset.sparse_to_dense_tensor(original_target_idx, self.original_target.shape[1])  

        return item
