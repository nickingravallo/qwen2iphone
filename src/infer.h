#pragma once
#include <stdint.h>
// One generation per cold start. The caller supplies zero-initialized BSS.
int infer(const uint8_t *packed, unsigned long size, void (*emit)(const uint8_t *, unsigned));

typedef struct {
    const uint8_t *data;
    uint64_t size;
} WeightBlock;
typedef struct {
    uint32_t version, count;
    WeightBlock blocks[];
} WeightMap;
int infer_segments(const WeightMap *map, void (*emit)(const uint8_t *, unsigned));
