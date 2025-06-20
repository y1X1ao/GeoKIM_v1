import torch
from torch.utils.data import Dataset
import numpy as np

class GeochemMAEDataset(Dataset):
    def __init__(self, X, mask_ratio=0.4, mask_probs=None, mode='correlation',
                 return_side_values=False, side_indices=None, force_mask_side_features=True):

        self.X = X.astype(np.float32)
        self.mask_ratio = mask_ratio
        self.mask_probs = mask_probs if mode == 'correlation' else None
        self.mode = mode
        self.n_features = self.X.shape[1]

        self.return_side_values = return_side_values
        self.side_indices = side_indices if side_indices is not None else []
        self.force_mask_side_features = force_mask_side_features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        total_mask = int(self.n_features * self.mask_ratio)

        if self.force_mask_side_features and len(self.side_indices) > 0:
            side_set = set(self.side_indices)
            available_indices = [i for i in range(self.n_features) if i not in side_set]
            remaining_mask = max(0, total_mask - len(self.side_indices))

            if self.mode == 'correlation' and self.mask_probs is not None:
                probs = np.array([self.mask_probs[i] for i in available_indices])
                probs = probs / probs.sum()
                sampled = np.random.choice(available_indices, size=remaining_mask, replace=False, p=probs)
            else:
                sampled = np.random.choice(available_indices, size=remaining_mask, replace=False)

            mask_indices = np.concatenate([sampled, self.side_indices])
        else:
            if self.mode == 'correlation' and self.mask_probs is not None:
                mask_indices = np.random.choice(self.n_features, size=total_mask, replace=False, p=self.mask_probs)
            else:
                mask_indices = np.random.choice(self.n_features, size=total_mask, replace=False)

        mask = np.zeros(self.n_features, dtype=bool)
        mask[mask_indices] = True

        x_masked = x.copy()
        x_masked[mask] = 0.0  
        if self.return_side_values:
            side_vals = x[self.side_indices]
            return torch.tensor(x_masked), torch.tensor(x), torch.tensor(mask), torch.tensor(side_vals)
        else:
            return torch.tensor(x_masked), torch.tensor(x), torch.tensor(mask)

