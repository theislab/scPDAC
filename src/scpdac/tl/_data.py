"""Memory-efficient torch helpers that densify expression rows on the fly.

Training/evaluating the MLPs on a full atlas would otherwise require a dense
``(n_cells, n_genes)`` float32 matrix in RAM. These utilities keep the matrix in
its native (typically sparse CSR) form and only densify the rows of the current
mini-batch, so peak memory scales with the batch size rather than the dataset.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import issparse
from torch.utils.data import Dataset


def densify_rows(x, rows: slice | np.ndarray | None = None) -> np.ndarray:
    """Return a dense ``float32`` array for ``x[rows]`` (or all of ``x``).

    Accepts dense arrays or scipy sparse matrices; ``rows`` may be a slice, an
    index array, or ``None`` for the whole matrix.
    """
    sub = x if rows is None else x[rows]
    if issparse(sub):
        sub = sub.toarray()
    return np.asarray(sub, dtype=np.float32)


class scDataset(Dataset):
    """A ``(features, label)`` dataset that densifies a whole mini-batch at once.

    ``__getitem__`` accepts either a single index or a sequence of indices (as
    produced by a :class:`~torch.utils.data.BatchSampler`). Passing a batch of
    indices lets the sparse matrix be sliced and densified in a single
    vectorised ``toarray`` call, which is dramatically faster than densifying one
    cell at a time — use :func:`make_loader` to wire this up.

    Parameters
    ----------
    x
        Feature matrix, dense ``ndarray`` or scipy sparse (CSR recommended for
        fast row access), shape ``(n_cells, n_genes)``.
    y
        Integer class labels, shape ``(n_cells,)``.
    """

    def __init__(self, x, y):
        self.x = x.tocsr() if issparse(x) else np.asarray(x, dtype=np.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self._sparse = issparse(self.x)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i):
        rows = densify_rows(self.x, i)
        return torch.from_numpy(np.atleast_2d(rows).astype(np.float32, copy=False)), self.y[i]


def make_loader(x, y, batch_size: int, *, shuffle: bool, drop_last: bool, num_workers: int = 0):
    """Build a :class:`~torch.utils.data.DataLoader` that yields densified batches.

    Uses a :class:`~torch.utils.data.BatchSampler` so each fetch hands a list of
    indices to :class:`scDataset`, which densifies the whole batch in one shot
    (``batch_size=None`` disables PyTorch's per-sample collation).
    """
    from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

    base = RandomSampler(range(x.shape[0])) if shuffle else SequentialSampler(range(x.shape[0]))
    sampler = BatchSampler(base, batch_size=batch_size, drop_last=drop_last)
    return DataLoader(scDataset(x, y), sampler=sampler, batch_size=None, num_workers=num_workers)


@torch.no_grad()
def predict_indices(model, x, device: str = "cpu", batch_size: int = 4096) -> np.ndarray:
    """Return ``argmax`` class indices for ``x``, densifying in batches.

    Parameters
    ----------
    model
        A trained classifier returning logits; run in ``eval`` mode.
    x
        Feature matrix (dense or sparse), shape ``(n_cells, n_genes)``.
    device
        Torch device to run inference on.
    batch_size
        Number of rows densified and scored at once.
    """
    n = x.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    model.eval()
    out = []
    for start in range(0, n, batch_size):
        chunk = densify_rows(x, slice(start, min(start + batch_size, n)))
        out.append(model(torch.as_tensor(chunk, device=device)).argmax(1).cpu().numpy())
    return np.concatenate(out)
