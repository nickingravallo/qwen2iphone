.text
adr x2, vectors
msr vbar_el1, x2
isb
adr x0, args
mov x1, #0x40400000
mov x3, #0x41000000
br x3
.balign 8
args:
.quad 0, 0, 0x40000000, 0x40000000, 0
.quad 0x78000000, 0, 4480, 1120, 512, 32
.balign 2048
vectors:
.rept 16
b fault
.balign 128
.endr
fault:
mrs x22, elr_el1
mrs x23, esr_el1
mrs x24, far_el1
b .
