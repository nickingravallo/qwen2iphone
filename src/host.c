#include "infer.h"
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void emit(const uint8_t *bytes, unsigned count) {
    fwrite(bytes,1,count,stdout);
    fflush(stdout);
}
int main(void) {
    int fd = open("build/weights.bin",O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd,&st)) return 1;
    void *data = mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    close(fd);
    if (data == MAP_FAILED) return 1;
    puts("Hi how are you\n");
    int error = infer(data,st.st_size,emit);
    puts("");
    return error != 0;
}
