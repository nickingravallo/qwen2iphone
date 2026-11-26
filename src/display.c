#include "infer.h"
#ifdef SERIAL_CONSOLE
#include "serial.h"
#endif
#include "font.h"

typedef struct {
    uint64_t header, virt_base, phys_base, mem_size, kernel_top;
    uint64_t base, display, row_bytes, width, height, depth;
} BootArgs;
extern const uint8_t packed_start[], packed_end[];

static volatile uint32_t *fb;
static unsigned stride, width, height, x, y;
static unsigned unicode, remaining;
enum { scale = 4, margin = 24, cell = 6*scale, line = 9*scale };

static void flush_range(uintptr_t lo, uintptr_t hi) {
    lo &= ~(uintptr_t)63;
    for (; lo < hi; lo += 64) __asm__ volatile("dc cvac, %0" :: "r"(lo) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
}

static void flush(unsigned first, unsigned last) {
    flush_range((uintptr_t)(fb + first*stride), (uintptr_t)(fb + last*stride));
}

static void clear(void) {
    for (unsigned row = 0; row < height; row++)
        for (unsigned col = 0; col < width; col++) fb[row*stride+col] = 0xffffffff;
    flush(0,height);
    x = y = margin;
}

static void glyph(unsigned c) {
    if (x + cell > width-margin || c == '\n') {
        x = margin;
        y += line;
        if (y+line > height-margin) clear();
        if (c == '\n') return;
    }
    uint64_t bits = c < 128 ? font[c] : 0;
    if (c == 0x1f60a) bits = 0xe4540540eull;
    if (!bits && c != ' ') bits = font['?'];
    for (unsigned row = 0; row < 7*scale; row++)
        for (unsigned col = 0; col < 5*scale; col++)
            if ((bits >> (5*(row/scale)+col/scale)) & 1)
                fb[(y+row)*stride+x+col] = 0xff000000;
    x += cell;
    flush(y,y+7*scale);
}

static void emit(const uint8_t *bytes, unsigned count) {
    for (unsigned i = 0; i < count; i++) {
        unsigned b = bytes[i];
        if (remaining && (b & 0xc0) == 0x80) {
            unicode = (unicode << 6) | (b & 63);
            if (--remaining == 0) glyph(unicode);
        } else {
            remaining = 0;
            if (b < 128) glyph(b);
            else if (b >= 0xc2 && b <= 0xf4) {
                remaining = b < 0xe0 ? 1 : b < 0xf0 ? 2 : 3;
                unicode = b & (0x3f >> remaining);
            } else glyph('?');
        }
    }
}

static void wait_one_second(void) {
    uint64_t start, now, frequency;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(frequency));
    __asm__ volatile("mrs %0, cntpct_el0" : "=r"(start));
    do {
        __asm__ volatile("mrs %0, cntpct_el0" : "=r"(now));
    } while (now-start < frequency);
}

// Called from the assembly entry before BSS initialization. Only use local
// variables, the supplied boot arguments, and read-only image data here.
int boot_rainbow(const BootArgs *args) {
    if (!args || !args->base || args->width < 128 || args->width > 8192 ||
        args->height < 128 || args->height > 8192 || args->row_bytes % 4 ||
        args->row_bytes < 4*args->width || args->row_bytes > 65536) return 0;
    static const uint32_t colors[] = {
        0xffff0000, 0xffff8000, 0xffffff00, 0xff00c040,
        0xff0080ff, 0xff4030b0, 0xffa020f0
    };
    for (unsigned row = 0; row < args->height; row++) {
        volatile uint32_t *line_start = (volatile uint32_t *)(args->base + row*args->row_bytes);
        uint32_t color = colors[row*7/args->height];
        for (unsigned col = 0; col < args->width; col++) line_start[col] = color;
    }
    flush_range(args->base, args->base + args->height*args->row_bytes);
    wait_one_second();
    return 1;
}

static void cable_countdown(void) {
    static const uint8_t message[] = "Connect serial cable\n";
    clear();
    emit(message,sizeof(message)-1);
    unsigned digit_x = x, digit_y = y;
    for (unsigned seconds = 10; seconds > 0; seconds--) {
        for (unsigned row = digit_y; row < digit_y+7*scale; row++)
            for (unsigned col = digit_x; col < digit_x+2*cell; col++)
                fb[row*stride+col] = 0xffffffff;
        x = digit_x;
        y = digit_y;
        if (seconds == 10) glyph('1');
        glyph('0'+seconds%10);
        wait_one_second();
    }
    clear();
}

void payload(const BootArgs *args, const WeightMap *map) {
    fb = (volatile uint32_t *)(uintptr_t)args->base;
    stride = args->row_bytes/4;
    width = args->width;
    height = args->height;
    cable_countdown();
#ifdef SERIAL_CONSOLE
    static const uint8_t status[] = "Serial console\nType ping or status";
    emit(status,sizeof(status)-1);
    serial_console();
#else
    static const uint8_t prompt[] = "Hi how are you\n\n";
    emit(prompt,sizeof(prompt)-1);
#ifdef SEGMENTED
    if (infer_segments(map,emit)) {
        static const uint8_t error[] = "Weight block validation failed";
        emit(error,sizeof(error)-1);
    }
#elif !defined(DISPLAY_ONLY)
    infer(packed_start,(uintptr_t)packed_end-(uintptr_t)packed_start,emit);
#endif
#endif
}
