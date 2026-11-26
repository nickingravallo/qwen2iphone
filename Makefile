CLANG ?= clang
CROSS ?= aarch64-elf-
FLAGS = --target=aarch64-none-elf -mcpu=apple-a11 -mstrict-align -ffreestanding -fno-stack-protector -fno-builtin -O3 -ffast-math -Ibuild
REV = 2fc06364715b967f1860aea9cf38778875588b17
FILES = config.json tokenizer.json LICENSE model.safetensors-00001-of-00001.safetensors
SOURCE = $(addprefix build/source/,$(FILES))

.PHONY: all host
all: build/qwen.bin
host: build/host
	build/host

build/source/%:
	mkdir -p build/source
	curl --fail --location --retry 3 'https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/$(REV)/$*' -o $@.part
	mv $@.part $@

build/weights.bin: tools/pack.py $(SOURCE)
	python3 tools/pack.py
build/model.h: build/weights.bin
	@test -f $@ || python3 tools/pack.py

build/infer.o: src/infer.c src/infer.h build/model.h Makefile
	$(CLANG) $(FLAGS) -c $< -o $@
build/display.o: src/display.c src/infer.h src/font.h
	mkdir -p build
	$(CLANG) $(FLAGS) -c $< -o $@
build/start.o: src/start.S
	mkdir -p build
	$(CLANG) $(FLAGS) -c $< -o $@
build/weights.o: src/weights.S build/weights.bin
	$(CLANG) $(FLAGS) -c $< -o $@
build/memory.o: src/memory.c Makefile
	mkdir -p build
	$(CLANG) $(FLAGS) -c $< -o $@
build/qwen.elf: build/memory.o build/start.o build/display.o build/infer.o build/weights.o src/raw.ld
	$(CROSS)ld -T src/raw.ld $(filter %.o,$^) -o $@
build/qwen.bin: build/qwen.elf
	$(CROSS)objcopy -O binary $< $@
build/host: src/host.c src/infer.c src/infer.h build/model.h
	$(CLANG) -O3 -ffast-math -Ibuild src/host.c src/infer.c -o $@

build/shim.bin: tools/qemu-entry.S
	mkdir -p build
	$(CLANG) --target=aarch64-none-elf -c $< -o build/shim.o
	$(CROSS)objcopy -O binary build/shim.o $@
.PHONY: smoke
smoke: build/qwen.bin build/shim.bin
	python3 tools/smoke.py

.PHONY: display smoke-display
display: build/display-only.bin
build/display-only.o: src/display.c src/infer.h src/font.h Makefile
	mkdir -p build
	$(CLANG) $(FLAGS) -DDISPLAY_ONLY -c $< -o $@
build/display-only.elf: build/start.o build/display-only.o src/raw.ld
	$(CROSS)ld -T src/raw.ld $(filter %.o,$^) -o $@
build/display-only.bin: build/display-only.elf
	$(CROSS)objcopy -O binary $< $@
smoke-display: display build/shim.bin
	python3 tools/smoke.py --display-only

.PHONY: segments smoke-segments
segments: build/segmented.bin
build/layout.json: build/weights.bin
	@test -f $@ || python3 tools/pack.py
build/segments.h: build/layout.json build/weights.bin tools/segment.py
	python3 tools/segment.py
build/infer-segments.o: src/infer.c src/infer.h build/model.h build/segments.h Makefile
	$(CLANG) $(FLAGS) -DSEGMENTED -c $< -o $@
build/display-segments.o: src/display.c src/infer.h src/font.h Makefile
	mkdir -p build
	$(CLANG) $(FLAGS) -DSEGMENTED -c $< -o $@
build/segmented.elf: build/start.o build/memory.o build/infer-segments.o build/display-segments.o src/raw.ld
	$(CROSS)ld -T src/raw.ld $(filter %.o,$^) -o $@
build/segmented.bin: build/segmented.elf
	$(CROSS)objcopy -O binary $< $@
smoke-segments: build/qwen.bin build/segmented.bin build/shim.bin
	python3 tools/smoke.py
	python3 tools/smoke.py --segmented

.PHONY: console
console: build/console.bin
build/console-display.o: src/display.c src/serial.h src/infer.h src/font.h Makefile
	mkdir -p build
	$(CLANG) $(FLAGS) -DSERIAL_CONSOLE -c $< -o $@
build/serial.o: src/serial.c src/serial.h Makefile
	mkdir -p build
	$(CLANG) $(FLAGS) -c $< -o $@
build/console.elf: build/start.o build/console-display.o build/serial.o src/raw.ld
	$(CROSS)ld -T src/raw.ld $(filter %.o,$^) -o $@
build/console.bin: build/console.elf
	$(CROSS)objcopy -O binary $< $@
.PHONY: smoke-console
smoke-console: console build/shim.bin
	python3 tools/smoke.py --console
