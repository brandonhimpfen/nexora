import numpy as np

float32 = np.dtype("float32")
float64 = np.dtype("float64")
int32 = np.dtype("int32")
int64 = np.dtype("int64")
bool_ = np.dtype("bool")

SUPPORTED_DTYPES = {float32, float64, int32, int64, bool_}


def normalize_dtype(dtype):
    if dtype is None:
        return None
    dt = np.dtype(dtype)
    if dt not in SUPPORTED_DTYPES:
        raise TypeError(f"Unsupported dtype: {dt}")
    return dt
