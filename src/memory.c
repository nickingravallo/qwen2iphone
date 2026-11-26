#include <stddef.h>
void *memset(void *destination, int byte, size_t size) {
    unsigned char *p = destination;
    while (size--) *p++ = (unsigned char)byte;
    return destination;
}
