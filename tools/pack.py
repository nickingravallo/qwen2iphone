"""Pack the pinned Qwen3.5-0.8B text checkpoint for a freestanding decoder."""
import json
import struct
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

build = Path("build")
source = build / "source"
checkpoint = next(source.glob("*.safetensors"))
with checkpoint.open("rb") as f:
    header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    start = f.tell()
if checkpoint.stat().st_size != start + max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"):
    raise ValueError("Incomplete safetensors checkpoint")

data = np.memmap(checkpoint, mode="r", dtype="u1")
tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
prompt = "<|im_start|>user\nHi how are you<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tokenizer.encode(prompt, add_special_tokens=False).ids
tables = [{} for _ in range(24)]
globals_ = {}
segment_starts = {}
with (build / "weights.tmp").open("wb") as out:
    def put(array):
        out.write(bytes((-out.tell()) % 16))
        offset = out.tell()
        array.tofile(out)
        return offset

    for name, spec in sorted(header.items()):
        if not name.startswith("model.language_model."):
            continue
        key = name.removeprefix("model.language_model.").removesuffix(".weight")
        table = globals_
        if key.startswith("layers."):
            _, layer, key = key.split(".", 2)
            table = tables[int(layer)]
        group = f"layer-{int(layer):02d}" if table is not globals_ else key
        segment_starts.setdefault(group, out.tell())
        key = key.replace("linear_attn.", "").replace("self_attn.", "").replace("mlp.", "")
        dtype = {"BF16": "<u2", "F32": "<f4"}[spec["dtype"]]
        raw = np.ndarray(spec["shape"], dtype, data, start + spec["data_offsets"][0])
        def floats(block):
            if dtype == "<u2":
                return (block.astype("<u4") << 16).view("<f4")
            return block
        if raw.ndim == 2:
            rows, cols = raw.shape
            scales = np.empty(rows, dtype="<f4")
            out.write(bytes((-out.tell()) % 16))
            offset = out.tell()
            for i in range(0, rows, 256):
                if key == "embed_tokens" and i and i % 65536 == 0:
                    segment_starts[f"embedding-row-{i}"] = out.tell()
                block = floats(raw[i:i+256])
                scale = np.maximum(np.abs(block).max(axis=1) / 127, 1e-30)
                scales[i:i+len(block)] = scale
                np.rint(block / scale[:, None]).astype("i1").tofile(out)
            table[key] = f"{{{offset}, {put(scales)}, {rows}, {cols}}}"
        else:
            table[key] = str(put(floats(raw)))
    angles = np.arange(128)[:, None] * 1e7 ** (-np.arange(32) / 32)
    rope = put(np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype("<f4"))
    # Invert the byte-level BPE alphabet, preserving bytes across token boundaries.
    visible = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
    byte_map = {chr(i): i for i in visible}
    for n, b in enumerate(i for i in range(256) if i not in visible):
        byte_map[chr(256+n)] = b
    pieces = bytearray()
    offsets = []
    for i in range(248320):
        offsets.append(len(pieces))
        token = tokenizer.id_to_token(i) or ""
        pieces.extend(bytes(byte_map[c] for c in token) if all(c in byte_map for c in token) else token.encode())
    offsets.append(len(pieces))
    strings = put(np.frombuffer(pieces, dtype="u1"))
    index = put(np.asarray(offsets, dtype="<u4"))
    size = out.tell()

def fields(table):
    return ", ".join(f".{k} = {v}" for k,v in table.items())

with (build / "model.h").open("w") as h:
    print("static const Layer layers[] = {", file=h)
    for layer in tables:
        print("    {" + fields(layer) + "},", file=h)
    print("};", file=h)
    print("static const Matrix embedding =", globals_["embed_tokens"], ";", file=h)
    print("static const unsigned prompt_ids[] = {", *[f"{v}," for v in ids], "};", file=h)
    for k,v in dict(final_norm=globals_["norm"], rope_offset=rope, index_offset=index,
                    strings_offset=strings, packed_bytes=size).items():
        print(f"#define {k} {v}u", file=h)
(build / "layout.json").write_text(json.dumps(segment_starts, indent=2))
(build / "weights.tmp").replace(build / "weights.bin")
