.section .text.start,"ax"
.global _start
_start:
    // x0 contains the framebuffer boot arguments. Preserve it until payload().
    mov x20, x1
    mrs x1, CurrentEL
    cmp x1, #4
    b.ne 1f
    mrs x1, cpacr_el1
    orr x1, x1, #(3 << 20)
    msr cpacr_el1, x1
    isb
1:
    adrp x1, stack_top
    add x1, x1, :lo12:stack_top
    mov sp, x1
    mov x19, x0
    bl boot_rainbow
    cbz w0, 4f
    mov x0, x19
    adrp x1, bss_start
    add x1, x1, :lo12:bss_start
    adrp x2, bss_end
    add x2, x2, :lo12:bss_end
2:
    cmp x1, x2
    b.hs 3f
    stp xzr, xzr, [x1], #16
    b 2b
3:
    mov x1, x20
    bl payload
.global payload_idle
payload_idle:
4:
    wfe
    b 4b
