ENTRY(_start)
SECTIONS {
    . = 0;
    .text : { KEEP(*(.text.start)) *(.text*) }
    .rodata : { *(.rodata*) }
    .data : { *(.data*) }
    .weights : ALIGN(16) { *(.weights) }
    .bss (NOLOAD) : ALIGN(16) {
        bss_start = .;
        *(.bss*) *(COMMON)
        . = ALIGN(16);
        bss_end = .;
        . += 64K;
        stack_top = .;
    }
    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) *(.llvm_addrsig) }
}
