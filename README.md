# Qwen2iPhone

Running Qwen3.5-0.8B on a bare metal iPhone X

## How?

Using the exploit known as checkm8, which is a BootROM exploit, and a small pre-boot environment known as PongoOS, custom data can be uploaded and executed on the iPhone.

PongoOS handles most of the setup. We load to an initial binary, which waits for a serial interface for communication. The model weights are provided over USB and communication is completed through serial. Basic debug information such as confirming initialization is communicated through the framebuffer.

## Instructions

Coming soon...
