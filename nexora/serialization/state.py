import io
import json
import zipfile
import numpy as np
from pathlib import Path
from nexora.errors import SerializationError
from .format import FORMAT_NAME, FORMAT_VERSION


def save(state, path):
    path = Path(path)
    arrays = {str(k): np.asarray(v.data if hasattr(v, "data") else v) for k, v in state.items()}
    metadata = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "tensors": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
    }
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        zf.writestr("tensors.npz", buffer.getvalue())


def load(path):
    path = Path(path)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            metadata = json.loads(zf.read("metadata.json"))
            if metadata.get("format") != FORMAT_NAME or metadata.get("version") != FORMAT_VERSION:
                raise SerializationError("Unsupported Nexora state format or version.")
            with np.load(io.BytesIO(zf.read("tensors.npz")), allow_pickle=False) as archive:
                state = {key: archive[key].copy() for key in archive.files}
    except SerializationError:
        raise
    except Exception as exc:
        raise SerializationError(f"Failed to load Nexora state file: {exc}") from exc
    return state
