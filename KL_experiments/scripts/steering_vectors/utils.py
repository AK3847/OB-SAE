import os
import torch
import pandas as pd


def hs_of(out):
    """
    Extract hidden states from transformer block output.
    Handles:
    tensor
    tuple(tensor, ...)
    """
    return out[0] if isinstance(out, tuple) else out



def save_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)



def save_vectors(vectors, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(vectors, path)



def load_vectors(path):
    return torch.load(path, map_location="cpu")