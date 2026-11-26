.section .weights,"a"
.balign 16
.global packed_start, packed_end
packed_start:
.incbin "build/weights.bin"
packed_end:
