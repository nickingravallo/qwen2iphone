"""Split packed weights only at tensor-group or embedding-row boundaries."""
import json
import zlib
from pathlib import Path

build = Path("build")
out = build / "segments"
out.mkdir(exist_ok=True)
starts = json.loads((build / "layout.json").read_text())
entries = []
with (build / "weights.bin").open("rb") as source:
    ends = [*list(starts.values())[1:], source.seek(0, 2)]
    source.seek(0)
    for (name, offset), end in zip(starts.items(), ends):
        length = end - offset
        if offset != source.tell() or not 0 < length <= 64 << 20:
            raise ValueError(f"Invalid block at {name}")
        data = source.read(length)
        if len(data) != length:
            raise ValueError(f"Truncated block at {name}")
        filename = name + ".bin"
        temporary = out / (filename + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(out / filename)
        entries.append(dict(file=filename, offset=offset, size=length, crc32=zlib.crc32(data)))

(out / "manifest.json").write_text(json.dumps(dict(version=1, blocks=entries), indent=2) + "\n")
with (build / "segments.h.tmp").open("w") as header:
    print("static const struct { unsigned start, size, crc32; } layout[] = {", file=header)
    for entry in entries:
        print("    {%(offset)du, %(size)du, %(crc32)du}," % entry, file=header)
    print("};", file=header)
(build / "segments.h.tmp").replace(build / "segments.h")
