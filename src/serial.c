#include "serial.h"
#include <stdint.h>
#ifdef HOST_SERIAL_TEST
#include <stdio.h>
static int receive(void) { return getchar(); }
static void transmit(char c) { putchar(c); }
#else
// 9phone's A11 debug UART. The handoff must preserve the port configuration.
static volatile uint32_t *const uart = (volatile uint32_t *)(uintptr_t)0x22e600000ull;
static int receive(void) {
    while (!(uart[0x10/4] & 1u)) {}
    return uart[0x24/4] & 255;
}
static void transmit(char c) {
    while (!(uart[0x10/4] & 4u)) {}
    uart[0x20/4] = (uint8_t)c;
}
#endif

static void line(const char *s) {
    while (*s) transmit(*s++);
    transmit('\r');
    transmit('\n');
}

static int same(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return *a == *b;
}

void serial_console(void) {
    char command[24];
    unsigned length = 0;
    int overflow = 0;
    line("Ready. Commands: ping, status");
    for (;;) {
        int c = receive();
        if (c < 0) return;
        if (c == '\r' || c == '\n') {
            command[length] = 0;
            if (overflow) line("Command too long");
            else if (same(command,"ping")) line("pong");
            else if (same(command,"status")) line("Serial active. USB receiver unavailable. Weights unloaded.");
            else if (length) line("Commands: ping, status");
            length = 0;
            overflow = 0;
        } else if (c >= 32 && c < 127) {
            if (length + 1 < sizeof(command)) command[length++] = (char)c;
            else overflow = 1;
        }
    }
}
