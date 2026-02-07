from .file import callback_if_uncached, download_ftp, download_http
from .trace_ops import rotate_stream_to_zne, stream_to_array, trace_has_spikes

__all__ = [
    "callback_if_uncached",
    "download_ftp",
    "download_http",
    "rotate_stream_to_zne",
    "stream_to_array",
    "trace_has_spikes",
    "pad_packed_sequence",
]


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
