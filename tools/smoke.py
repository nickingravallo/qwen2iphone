"""Exercise the flat binary at EL1 and capture both guest framebuffer states."""
import argparse
import json
import platform
import re
import struct
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
variants = parser.add_mutually_exclusive_group()
variants.add_argument("--display-only", action="store_true")
variants.add_argument("--console", action="store_true")
variants.add_argument("--segmented", action="store_true")
args = parser.parse_args()
display_only = args.display_only
image = "console" if args.console else "segmented" if args.segmented else "display-only" if display_only else "qwen"
build = Path("build")
width, height = 1120, 512
base = 0x41000000
fb = 0x78000000
symbols = subprocess.check_output(["aarch64-elf-nm", str(build / f"{image}.elf")], text=True)
idle = base + int(re.search(r"^(\w+) T payload_idle$", symbols, re.M)[1], 16)
hvf = platform.system() == "Darwin" and platform.machine() == "arm64"
command = ["qemu-system-aarch64", "-M", "virt", "-accel", "hvf" if hvf else "tcg",
           "-cpu", "host" if hvf else "max", "-m", "1G", "-nodefaults", "-display", "none",
           "-S", "-qmp", "stdio", "-device",
           "loader,file=build/shim.bin,addr=0x40200000,cpu-num=0,force-raw=on",
           "-device", f"loader,file=build/{image}.bin,addr=0x41000000,force-raw=on"]

if args.segmented:
    folder = build / "segments"
    manifest = json.loads((folder / "manifest.json").read_text())
    blocks = manifest["blocks"]
    # Reverse physical order and leave gaps, so accidental contiguous reads fail.
    addr = 0x43000000
    locations = {}
    for block in reversed(blocks):
        locations[block["file"]] = addr
        command += ["-device", f"loader,file={folder/block['file']},addr={addr:#x},force-raw=on"]
        addr = ((addr + block["size"] + 65535) & ~65535) + 65536
    if addr > fb:
        raise ValueError("Test layout overlaps framebuffer")
    descriptor = struct.pack("<II", manifest["version"], len(blocks))
    for block in blocks:
        descriptor += struct.pack("<QQ", locations[block["file"]], block["size"])
    (build / "qemu-weight-map.bin").write_bytes(descriptor)
    command += ["-device", "loader,file=build/qemu-weight-map.bin,addr=0x40400000,force-raw=on"]

def save_png(path, pixels):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data +
                struct.pack(">I", zlib.crc32(kind + data)))
    rgb = pixels[:, :, [2, 1, 0]]
    scan = b"".join(b"\0" + row.tobytes() for row in rgb)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width,height,8,2,0,0,0))
                     + chunk(b"IDAT",zlib.compress(scan)) + chunk(b"IEND",b""))

with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as vm:
    def execute(name, args=None):
        vm.stdin.write(json.dumps({"execute": name, "arguments": args or {}}) + "\n")
        vm.stdin.flush()
        while line := vm.stdout.readline():
            msg = json.loads(line)
            if "return" in msg:
                return msg["return"]
            if "error" in msg:
                raise RuntimeError(msg["error"])
        raise RuntimeError("QEMU stopped")
    def snapshot():
        path = build / f"{image}-display.raw"
        execute("pmemsave", {"val": fb, "size": width*height*4, "filename": str(path)})
        return np.fromfile(path,dtype="u1").reshape(height,width,4)
    try:
        execute("qmp_capabilities")
        execute("cont")
        deadline = time.monotonic()+20
        while time.monotonic() < deadline:
            pixels = snapshot()
            # Sample the center of each horizontal band after the flush.
            sample = pixels[(np.arange(7)*2+1)*height//14, width//2, :3]
            colors = [0xff0000,0xff8000,0xffff00,0x00c040,0x0080ff,0x4030b0,0xa020f0]
            if all(bytes(bgr) == c.to_bytes(3,"little") for bgr,c in zip(sample,colors)):
                save_png(build / ("display-rainbow.png" if display_only else "rainbow.png"), pixels)
                break
            time.sleep(.05)
        else:
            raise RuntimeError("Startup rainbow was not observed")
        # Read the two countdown cells directly from guest framebuffer pixels.
        font = [int(v,16) for v in re.findall(r"0x[0-9a-f]+",Path("src/font.h").read_text())]
        deadline = time.monotonic() + (60 if hvf else 600)
        for seconds in range(10,0,-1):
            digits = str(seconds).ljust(2)
            expected = np.zeros((7,12),dtype=bool)
            for cell, char in enumerate(digits):
                for row in range(7):
                    for col in range(5):
                        expected[row,cell*6+col] = (font[ord(char)] >> (5*row+col)) & 1
            while time.monotonic() < deadline:
                pixels = snapshot()
                if np.array_equal(pixels[60:88:4,24:72:4,0] == 0, expected):
                    if seconds == 10:
                        save_png(build / f"{image}-countdown.png",pixels)
                    break
                time.sleep(.1)
            else:
                raise RuntimeError(f"Missing countdown value {seconds}")
        deadline = time.monotonic() + (300 if hvf else 3600)
        while time.monotonic() < deadline:
            if args.console:
                # The virt machine has no A11 UART. Observe the display at UART handoff.
                pixels = snapshot()
                label = "Type ping or status"
                expected = np.array([[bool(font[ord(c)] & (1 << (row*5+col))) if col < 5 else False
                                      for c in label for col in range(6)] for row in range(7)])
                if np.array_equal(pixels[60:88:4,24:24+24*len(label):4,0] == 0, expected):
                    break
                time.sleep(.1)
                continue
            regs = execute("human-monitor-command", {"command-line":"info registers"})
            pc = int(re.search(r"PC=([0-9a-fA-F]+)",regs)[1],16)
            if pc in (idle,idle+4):
                break
            if 0x40201000 <= pc < 0x40201020:
                raise RuntimeError("Guest exception\n"+regs)
            time.sleep(.5)
        else:
            raise RuntimeError("Decoder timed out")
        pixels = snapshot()
        if not (pixels[24:52,:,0] == 0).any():
            raise RuntimeError("Missing prompt")
        if args.console:
            capture = build / "console-ready.png"
        elif display_only:
            if not np.all(pixels[52:, :, :3] == 255):
                raise RuntimeError("Expected a white screen below the prompt")
            capture = build / "display-prompt.png"
        else:
            if (pixels[96:,:,0] == 0).sum() < 3000:
                raise RuntimeError("Missing generated response")
            capture = build / "response.png"
        if not display_only and not args.console:
            def read_symbol(name, size):
                address = base + int(re.search(r"^(\w+) B " + name + "$",symbols,re.M)[1],16)
                path = build / f"{image}-{name}.raw"
                execute("pmemsave", {"val":address, "size":size, "filename":str(path)})
                return path.read_bytes()
            count, = struct.unpack("<I", read_symbol("generated_count",4))
            if not 1 <= count <= 128:
                raise RuntimeError(f"Invalid generated token count {count}")
            trace = read_symbol("generated_ids",4*count)
            if args.segmented:
                if trace != (build / "qwen-generated_ids.raw").read_bytes():
                    raise RuntimeError("Segmented generation diverged from reference token IDs")
                capture = build / "segmented-response.png"
                print(f"Matched all {count} token IDs against the contiguous image.")
        save_png(capture,pixels)
    finally:
        vm.terminate()
print(f"Verified startup rainbow and final framebuffer. See {capture}.")
