# Qwen2iPhone

Running Qwen3.5-0.8B on a bare metal iPhone X

## How?

Using the exploit known as checkm8, which is a BootROM exploit, and a small pre-boot environment known as PongoOS, custom data can be uploaded and executed on the iPhone.

PongoOS handles most of the setup. We load to an initial binary, which waits for a serial interface for communication. The model weights are provided over USB and communication is completed through serial. Basic debug information such as confirming initialization is communicated through the framebuffer.

## Instructions

You'll need the following to run this project:

- an iPhone X
- A Mac 
- USB C to USB A Adapter 
- USB A to USB C Lightning Cable
- DSCD iPhone Lightning Cable (for serial communication: optional soon)
- Checkra1n / Palera1n (Exploit your iPhone and enter the pre-boot exec environment known as PongoOS)

- Prompt your local agent: "I have the requirements from the README. Please guide me through running this project and getting it setup. Once we've loaded PongoOS and uploaded the console to the device and the user has plugged in the DSCD cable take control and tell the user what to do next." 

## Why?

I thought it would be fun to run Qwen on an iPhone. I had some old framebuffer and serial code for debugging an OS initialization through Pongo that I had written a few months ago when working on a Plan9 (9front) port for the iPhone. An inference provider seemed like a logical next step.
