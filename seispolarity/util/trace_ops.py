import logging

import numpy as np

logger = logging.getLogger("seispolarity")

def trace_has_spikes(data, factor=25, quantile=0.975):
    """
    Checks for bit flip errors in the data using a simple quantile rule
    Check for bit-flip errors in data using simple quantile rule
    """
    q = np.quantile(np.abs(data), quantile, axis=1, keepdims=True)
    return np.any(data > q * factor)


def stream_to_array(stream, component_order):
    """
    Convert single-station waveform stream to numpy array according to
    given component order.
    """
    starttime = min(trace.stats.starttime for trace in stream)
    endtime = max(trace.stats.endtime for trace in stream)
    sampling_rate = stream[0].stats.sampling_rate

    samples = int((endtime - starttime) * sampling_rate) + 1

    completeness = 0.0
    data = np.zeros((len(component_order), samples), dtype="float64")
    for c_idx, c in enumerate(component_order):
        c_stream = stream.select(channel=f"??{c}")
        if len(c_stream) > 1:
            logger.warning(
                f"Found multiple traces for {c_stream[0].id} starting at "
                f"{stream[0].stats.starttime}. Completeness will be wrong in case "
                f"of overlapping traces."
            )
            c_stream = sorted(c_stream, key=lambda x: x.stats.npts)

        c_completeness = 0.0
        for trace in c_stream:
            start_sample = int((trace.stats.starttime - starttime) * sampling_rate)
            tr_length = min(len(trace.data), samples - start_sample)
            data[c_idx, start_sample : start_sample + tr_length] = trace.data[
                :tr_length
            ]
            c_completeness += tr_length

        completeness += min(1.0, c_completeness / samples)

    data -= np.mean(data, axis=1, keepdims=True)

    completeness /= len(component_order)
    return starttime, data, completeness


def rotate_stream_to_zne(stream, inventory):
    """
    Tries to rotate the stream to ZNE inplace.
    Try to rotate stream to ZNE in-place.
    """
    try:
        stream.rotate("->ZNE", inventory=inventory)
    except ValueError:
        pass
    except NotImplementedError:
        pass
    except AttributeError:
        pass
    except Exception:
        pass
