# Qwen on an A11 framebuffer

The flat ARM64 image runs one text-only Qwen3.5-0.8B chat turn for “Hi how are you”. Entry paints seven horizontal rainbow bands, cleans the framebuffer cache lines, and holds the image for one second using the ARM generic timer. After the display check, the screen reads “Connect serial cable” and counts down from 10 to 1 at one-second intervals. The prompt and generated reply follow on white. Input is fixed at build time. The separate console image provides an A11 UART diagnostic prompt.

## Build and run

The build uses Clang with an AArch64 bare-metal target, GNU aarch64-elf binutils, Make, and Python 3 with NumPy and tokenizers. The first build downloads the model and needs about 5 GiB of disk space. QEMU is needed for the framebuffer smoke test.

```sh
make
make host
make smoke
```

The outputs are `build/qwen.bin`, a raw image with the weights embedded, and `build/qwen.elf`, its symbolized counterpart. The host target runs the same C decoder and prints generated tokens. The smoke target runs the raw image in a QEMU ARM64 guest with a memory-backed framebuffer. It checks all seven startup colors, waits for generation to finish, and captures the guest pixels in `build/rainbow.png` and `build/response.png`.

## Model and memory

The packer uses the [official Qwen checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17). Each matrix row has signed 8-bit weights and one floating-point scale. The image includes the tied embedding/output matrix, all 24 text layers, the output vocabulary, and precomputed rotary positions. The decoder uses greedy sampling with a 128-token budget shared by the chat prompt and reply. The original model license downloads to `build/source/LICENSE`.

The image is about 724.2 MiB. Its contiguous runtime region, including zero-initialized state and stack, is about 746.6 MiB. Framebuffer storage is separate. The ELF symbol `stack_top` gives the exact reservation size for a given build.

## Cable swap and serial diagnostics

`make console` builds `build/console.bin` with no model dependency. The first visible action is the startup rainbow. When “Connect serial cable” appears, the display counts from 10 to 1. After the full ten seconds it displays “Serial console” and begins polling the inherited A11 debug UART. The console continues listening if the adapter connects late. Use the serial accessory's terminal port at 115200 baud, 8 data bits, no parity, one stop bit. Send `ping` followed by Enter to get `pong`. Send `status` for the implemented transport and model state.

USB weight transfer is still pending. Starting the serial console does not provide a USB endpoint. Concurrent debug UART and USB data availability depends on the accessory and has not been verified on hardware.

## Segmented model

Run `make segments` to build `build/segmented.bin` and the block files listed in `build/segments/manifest.json`. This variant splits the embedding at row boundaries, keeps each decoder layer together, and stores the final normalization and token tables in the last block. All 29 blocks fit within 64 MiB each. The packed data still totals 724.2 MiB. The small executable reserves approximately 22.5 MiB for its code, inference state, and stack, in addition to the blocks.

The consumer takes a version-1 `WeightMap` defined in `src/infer.h`. It expects readable, 16-byte-aligned blocks in manifest order, at independently chosen addresses. Each block stays resident for the generation. The decoder checks the count, sizes and CRC32 values before reading weights. The manifest offsets identify positions in the original model byte stream.

Run `make smoke-segments` to compare the contiguous image and the segmented image in QEMU. The test puts blocks in reverse physical order with gaps between them and compares generated token IDs through the stop token. The emulator supplies the blocks before entry. There is no on-device USB block receiver in this build.

## Framebuffer entry contract and device status

The assembly entry expects a page-aligned image, an available stack and state region at the offsets given by the linker script, and the 9phone boot-arguments layout in x0. The framebuffer must already be active, addressable, and use 32-bit pixels. The display code uses the supplied byte stride and dimensions. The tested execution environment is EL1 with access to the ARM physical counter and cache-maintenance instructions.

The local PongoOS checkout caps each USB upload at 128 MiB in `src/shell/usbloader.c`. The single packed image exceeds that limit. Physical iPhone X transport, contiguous placement, display color order, and execution remain unverified.
