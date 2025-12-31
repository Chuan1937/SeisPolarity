from .file import download_http, download_ftp, callback_if_uncached
from .trace_ops import trace_has_spikes, stream_to_array, rotate_stream_to_zne

def pad_packed_sequence(seq, axis=0):
    import numpy as np
    if not seq:
        return np.array([])
        
    max_size = np.array([max([x.shape[i] for x in seq]) for i in range(seq[0].ndim)])

    new_seq = []
    for i, elem in enumerate(seq):
        d = max_size - np.array(elem.shape)
        if (d != 0).any():
            pad = [(0, d_dim) for d_dim in d]
            new_seq.append(np.pad(elem, pad, "constant", constant_values=0))
        else:
            new_seq.append(elem)

    return np.stack(new_seq, axis=axis)
