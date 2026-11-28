"""Pure-stdlib binary GGUF v2/v3 header reader using `struct` (zero dependencies).

Reads architecture, model name, native context length, exact quantization (from `general.file_type`), parameter count
(from `general.size_label` or tensor shapes), LoRA adapter detection (`general.type == 'adapter'`), and file size
directly from `.gguf` files without relying on filenames.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from . import config

# llama.cpp llama_ftype -> canonical quantization tag
_FTYPE_MAP: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0",
    8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
    13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4",
}

# GGUF metadata value type codes -> (struct fmt, byte size) for fixed-width scalars
_SCALAR_TYPES: dict[int, tuple[str, int]] = {
    0: ("<B", 1),   # UINT8
    1: ("<b", 1),   # INT8
    2: ("<H", 2),   # UINT16
    3: ("<h", 2),   # INT16
    4: ("<I", 4),   # UINT32
    5: ("<i", 4),   # INT32
    6: ("<f", 4),   # FLOAT32
    7: ("<?", 1),   # BOOL
    10: ("<Q", 8),  # UINT64
    11: ("<q", 8),  # INT64
    12: ("<d", 8),  # FLOAT64
}
_TYPE_STRING = 8
_TYPE_ARRAY = 9


@dataclass
class GGUFInfo:
    path: Path
    size_bytes: int
    version: int = 0
    tensor_count: int = 0
    architecture: str = ""
    name: str = ""
    quant: str = ""
    context_length: int = 0
    size_label: str = ""
    param_count: int = 0
    is_lora: bool = False
    general_type: str = ""

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def summary(self) -> str:
        bits = [x for x in (
            self.architecture,
            self.params_str,
            self.quant,
            f"ctx {self.context_length:,}" if self.context_length else "",
            f"{self.size_gb:.2f} GB" if self.size_gb >= 0.1 else f"{self.size_bytes / 1e6:.0f} MB",
        ) if x]
        return " · ".join(bits)

    @property
    def params_str(self) -> str:
        if self.size_label:
            return self.size_label
        if self.param_count >= 1_000_000_000:
            return f"{self.param_count / 1e9:.1f}B"
        if self.param_count >= 1_000_000:
            return f"{self.param_count / 1e6:.0f}M"
        return ""


def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError("truncated GGUF header")
    return data


def _read_str(f, max_len: int = 1024 * 1024) -> str:
    (n,) = struct.unpack("<Q", _read_exact(f, 8))
    if n > max_len:
        f.seek(n, 1)
        return ""
    return _read_exact(f, n).decode("utf-8", errors="replace")


def _skip_value(f, vtype: int) -> None:
    if vtype in _SCALAR_TYPES:
        f.seek(_SCALAR_TYPES[vtype][1], 1)
    elif vtype == _TYPE_STRING:
        (n,) = struct.unpack("<Q", _read_exact(f, 8))
        f.seek(n, 1)
    elif vtype == _TYPE_ARRAY:
        (elem_t, count) = struct.unpack("<IQ", _read_exact(f, 12))
        if elem_t in _SCALAR_TYPES:
            f.seek(_SCALAR_TYPES[elem_t][1] * count, 1)
        else:
            for _ in range(min(count, 2_000_000)):
                _skip_value(f, elem_t)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def _read_value(f, vtype: int):
    if vtype in _SCALAR_TYPES:
        fmt, sz = _SCALAR_TYPES[vtype]
        return struct.unpack(fmt, _read_exact(f, sz))[0]
    if vtype == _TYPE_STRING:
        return _read_str(f, max_len=4096)
    _skip_value(f, vtype)
    return None


def read_gguf(path: Path | str) -> GGUFInfo | None:
    """Reads binary GGUF header metadata. Returns None if file is not a valid GGUF v2/v3 file."""
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    size_bytes = p.stat().st_size
    info = GGUFInfo(path=p, size_bytes=size_bytes)
    want_exact = {
        "general.architecture", "general.name", "general.file_type",
        "general.size_label", "general.type", "adapter.type",
    }
    kv: dict[str, object] = {}
    try:
        with p.open("rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version, tensor_count, kv_count = struct.unpack("<IQQ", _read_exact(f, 20))
            if version not in (2, 3) or kv_count > 100_000:
                return None
            info.version = version
            info.tensor_count = tensor_count
            for _ in range(kv_count):
                key = _read_str(f, max_len=512)
                (vtype,) = struct.unpack("<I", _read_exact(f, 4))
                if key in want_exact or key.endswith((".context_length", ".block_count", ".embedding_length")):
                    kv[key] = _read_value(f, vtype)
                else:
                    _skip_value(f, vtype)
            # Sum exact parameter count from tensor info table (fast: ~200–600 headers, no weight data read)
            if tensor_count <= 20_000:
                total_p = 0
                for _ in range(tensor_count):
                    _skip_value(f, _TYPE_STRING)
                    (n_dims,) = struct.unpack("<I", _read_exact(f, 4))
                    if n_dims > 8:
                        break
                    dims = struct.unpack(f"<{n_dims}Q", _read_exact(f, 8 * n_dims))
                    f.seek(12, 1)   # ggml_type (4) + offset (8)
                    prod = 1
                    for d in dims:
                        prod *= d
                    total_p += prod
                info.param_count = total_p
    except (OSError, EOFError, struct.error, ValueError):
        if not kv:
            return None

    arch = str(kv.get("general.architecture") or "").strip()
    info.architecture = arch
    info.name = str(kv.get("general.name") or "").strip()
    info.size_label = str(kv.get("general.size_label") or "").strip()
    info.general_type = str(kv.get("general.type") or "").strip().lower()
    adapter_type = str(kv.get("adapter.type") or "").strip().lower()
    info.is_lora = (info.general_type == "adapter" or adapter_type == "lora" or "lora" in p.name.lower())
    ft = kv.get("general.file_type")
    if isinstance(ft, int) and ft in _FTYPE_MAP:
        info.quant = _FTYPE_MAP[ft]
    else:
        info.quant = config.detect_quant(p.name)
    ctx_v = kv.get(f"{arch}.context_length") if arch else None
    if ctx_v is None:
        ctx_v = next((v for k, v in kv.items() if k.endswith(".context_length") and isinstance(v, int)), 0)
    if isinstance(ctx_v, int) and ctx_v > 0:
        info.context_length = ctx_v
    return info


def scan_folder(folder: Path | str, recursive: bool = False) -> tuple[list[GGUFInfo], list[GGUFInfo]]:
    """Scans a directory for `.gguf` files using real GGUF headers. Returns `(base_models, lora_adapters)`."""
    root = Path(folder).expanduser()
    if not root.is_dir():
        return [], []
    it = root.rglob("*.gguf") if recursive else root.glob("*.gguf")
    bases: list[GGUFInfo] = []
    loras: list[GGUFInfo] = []
    for p in sorted(it):
        if "-0000" in p.name and not p.name.endswith("-00001-of-00001.gguf") and "-00001-of-" not in p.name:
            continue                                # skip secondary split shards (llama-server loads shard 00001)
        info = read_gguf(p)
        if not info:
            continue
        if info.is_lora:
            loras.append(info)
        else:
            bases.append(info)
    return bases, loras
