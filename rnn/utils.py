import torch

def normal_init(size, mean=0, stdev=0.02):
    """Make a tenser of a given size by sampling a normal distribution.

    Args:
        size (torch.Size): Size of the Tensor to initialize.
        mean (float): Mean of the normal distribution.
        stdev (float): Standard deviation of the normal distribution.

    Returns:
        torch.FloatTensor: Initialized tensor.

    """
    return torch.Tensor(size).normal_(mean, stdev)

def pad_sequences(x, seq_element_shape, pad_index=0):
    """
    Pad x so they all have the same length.

    Source: https://github.com/btaille/tagger/blob/c378d526fa4678694d6480660e43db4051b183de/loader.py

    Args:
        x (:obj:`list` of :obj:`list`): sequences of varying-length.
        seq_element_shape (:obj:`list` of int): Shape of an element of a sequence.
        pad_index (float): Value to pad with.

    """
    lens = [len(s) for s in x]
    maxlen = max(lens)
    sorted_indices = sorted(range(len(lens)), key=lambda k: lens[k], reverse=True)

    sequences = pad_index * torch.ones(len(x), maxlen, *seq_element_shape).long()

    for i, idx in enumerate(sorted_indices):
        sequences[i, :lens[idx]] = torch.from_numpy(x[idx])

    return sequences, sorted(lens, reverse=True), sorted_indices
