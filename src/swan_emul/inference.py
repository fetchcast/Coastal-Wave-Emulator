import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, X, Y, L, idxs=None):
        self.X, self.Y, self.L = X, Y, L
        N = len(X) - L
        self.idxs = np.arange(N, dtype=int) if idxs is None else np.asarray(idxs, int)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        t = self.idxs[i]
        return (torch.from_numpy(self.X[t:t+self.L]).float(),
                torch.from_numpy(self.Y[t+self.L]).float())

def run_inference(model, X, L=12, batch=1, device="cpu"):
    ds = SeqDataset(X, np.zeros_like(X[:,:4]), L)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, drop_last=False)
    preds = []
    model.to(device)
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)  # (B,L,C,H,W)
            yb = model(xb)      # (B,4,H,W)
            preds.append(yb.cpu().numpy())
    return np.concatenate(preds, 0)  # (N,4,H,W)
