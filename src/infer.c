#include <stdint.h>
#include "infer.h"

typedef struct { unsigned offset, scale, rows, cols; } Matrix;
typedef struct {
    Matrix in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj;
    Matrix q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj;
    unsigned input_layernorm, post_attention_layernorm, norm, q_norm, k_norm;
    unsigned A_log, dt_bias, conv1d;
} Layer;
#include "model.h"

enum { hidden = 1024, context = 128, linear_width = 2048 };
#ifdef SEGMENTED
#include "segments.h"
static const WeightMap *model;
#define BLOCK_COUNT (sizeof(layout) / sizeof(layout[0]))

// The packer keeps every accessed row or array wholly inside one block.
static const uint8_t *weight_at(unsigned offset) {
    unsigned lo = 0, hi = BLOCK_COUNT;
    while (lo + 1 < hi) {
        unsigned mid = (lo + hi) / 2;
        if (layout[mid].start <= offset) lo = mid;
        else hi = mid;
    }
    return model->blocks[lo].data + (offset - layout[lo].start);
}

static int validate_blocks(const WeightMap *map) {
    if (!map || map->version != 1 || map->count != BLOCK_COUNT) return -1;
    uint32_t table[256];
    for (unsigned i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int bit = 0; bit < 8; bit++)
            crc = (crc >> 1) ^ ((0u - (crc & 1)) & 0xedb88320u);
        table[i] = crc;
    }
    for (unsigned i = 0; i < BLOCK_COUNT; i++) {
        const WeightBlock *block = &map->blocks[i];
        if (!block->data || ((uintptr_t)block->data & 15) ||
            block->size != layout[i].size) return -1;
        uint32_t crc = ~0u;
        for (unsigned j = 0; j < block->size; j++)
            crc = table[(crc ^ block->data[j]) & 255] ^ (crc >> 8);
        if (~crc != layout[i].crc32) return -1;
    }
    return 0;
}
#else
static const uint8_t *weights;
static const uint8_t *weight_at(unsigned offset) { return weights + offset; }
#endif
// Captured by the emulator equivalence test, including the stop token.
unsigned generated_count, generated_ids[context];
static float state[18][16][128][128], history[18][6144][3];
static float keys[6][context][512], values[6][context][512];
static float x[hidden], input[hidden], residual[hidden], mixed[6144], gate[2048];
static float attention[2048], up[3584], mlp[3584];

static const float *floats(unsigned offset) {
    return (const float *)weight_at(offset);
}

static float dot(const float *a, const float *b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

static float row_dot(Matrix m, unsigned row, const float *v) {
    const int8_t *w = (const int8_t *)weight_at(m.offset + row * m.cols);
    float sum = 0;
    for (unsigned j = 0; j < m.cols; j++) sum += w[j] * v[j];
    return sum * floats(m.scale)[row];
}

static void project(Matrix m, const float *v, float *out) {
    for (unsigned i = 0; i < m.rows; i++) out[i] = row_dot(m, i, v);
}

// Base-2 range reduction keeps the exponential polynomial close to zero.
static float exponential(float a) {
    if (a < -87) return 0;
    if (a > 87) a = 87;
    int n = (int)(a * 1.44269504f + (a >= 0 ? .5f : -.5f));
    float r = a - n * .69314718f;
    union { uint32_t bits; float value; } scale = { (uint32_t)(n + 127) << 23 };
    return scale.value * (1 + r*(1 + r*(.5f + r*(1.f/6 + r*(1.f/24 + r/120)))));
}

static float sigmoid(float a) { return 1 / (1 + exponential(-a)); }
static float silu(float a) { return a * sigmoid(a); }

static float softplus(float a) {
    float e = exponential(-__builtin_fabsf(a));
    float y = e/(2+e), y2 = y*y;
    return (a > 0 ? a : 0) + 2*y*(1+y2*(1.f/3+y2*(1.f/5+y2*(1.f/7+y2/9))));
}

static void norm(const float *v, float *out, int n, unsigned offset, float bias) {
    float factor = 1 / __builtin_sqrtf(dot(v,v,n)/n + 1e-6f);
    const float *w = floats(offset);
    for (int i = 0; i < n; i++) out[i] = v[i] * factor * (w[i] + bias);
}

static void delta_attention(const Layer *l, int slot) {
    float a[16], b[16];
    project(l->in_proj_a, input, a);
    project(l->in_proj_b, input, b);
    project(l->in_proj_qkv, input, mixed);
    project(l->in_proj_z, input, gate);
    const float *kernel = floats(l->conv1d);
    for (int i = 0; i < 6144; i++) {
        float *h = history[slot][i];
        float sample = mixed[i];
        mixed[i] = silu(dot(h,kernel+4*i,3) + sample * kernel[4*i+3]);
        h[0] = h[1]; h[1] = h[2]; h[2] = sample;
    }
    for (int head = 0; head < 16; head++) {
        float *q = mixed + head*128, *k = q+linear_width, *v = k+linear_width;
        float *out = attention + head*128;
        float qscale = .08838834765f / __builtin_sqrtf(dot(q,q,128)+1e-6f);
        float kscale = 1 / __builtin_sqrtf(dot(k,k,128)+1e-6f);
        float decay = exponential(-exponential(floats(l->A_log)[head]) *
                                  softplus(a[head]+floats(l->dt_bias)[head]));
        float beta = sigmoid(b[head]);
        float prediction[128] = {0};
        for (int i = 0; i < 128; i++) {
            k[i] *= kscale;
            q[i] *= qscale;
            float *row = state[slot][head][i];
            for (int j = 0; j < 128; j++) {
                row[j] *= decay;
                prediction[j] += k[i] * row[j];
            }
        }
        for (int j = 0; j < 128; j++) {
            v[j] = beta * (v[j] - prediction[j]);
            out[j] = 0;
        }
        for (int i = 0; i < 128; i++) {
            float *row = state[slot][head][i];
            for (int j = 0; j < 128; j++) {
                row[j] += k[i] * v[j];
                out[j] += q[i] * row[j];
            }
        }
        norm(out,out,128,l->norm,0);
        for (int j = 0; j < 128; j++) out[j] *= silu(gate[head*128+j]);
    }
    project(l->out_proj, attention, residual);
}

static void rotate(float *v, int pos) {
    const float *cs = floats(rope_offset) + pos*64;
    for (int i = 0; i < 32; i++) {
        float a = v[i], b = v[i+32];
        v[i] = a*cs[2*i] - b*cs[2*i+1];
        v[i+32] = b*cs[2*i] + a*cs[2*i+1];
    }
}

static void full_attention(const Layer *l, int slot, int pos) {
    float *k = keys[slot][pos];
    project(l->k_proj,input,k);
    project(l->v_proj,input,values[slot][pos]);
    project(l->q_proj,input,mixed);
    for (int h = 0; h < 2; h++) {
        norm(k+256*h,k+256*h,256,l->k_norm,1);
        rotate(k+256*h,pos);
    }
    for (int h = 0; h < 8; h++) {
        float *q = mixed + 512*h, *out = attention + 256*h;
        int group = 256*(h/4);
        norm(q,q,256,l->q_norm,1);
        rotate(q,pos);
        float scores[context], largest = -1e30f, total = 0;
        for (int t = 0; t <= pos; t++) {
            scores[t] = dot(q,keys[slot][t]+group,256)/16;
            if (scores[t] > largest) largest = scores[t];
        }
        for (int t = 0; t <= pos; t++) total += (scores[t] = exponential(scores[t]-largest));
        for (int j = 0; j < 256; j++) {
            float sum = 0;
            for (int t = 0; t <= pos; t++) sum += scores[t]*values[slot][t][group+j];
            out[j] = sum/total * sigmoid(q[256+j]);
        }
    }
    project(l->o_proj,attention,residual);
}

static void forward(unsigned token, int pos) {
    const int8_t *row = (const int8_t *)weight_at(embedding.offset + token*hidden);
    for (int i = 0; i < hidden; i++) x[i] = row[i] * floats(embedding.scale)[token];
    for (int i = 0; i < 24; i++) {
        const Layer *l = &layers[i];
        norm(x,input,hidden,l->input_layernorm,1);
        if (i%4 == 3) full_attention(l,i/4,pos);
        else delta_attention(l,i-i/4);
        for (int j = 0; j < hidden; j++) x[j] += residual[j];
        norm(x,input,hidden,l->post_attention_layernorm,1);
        project(l->gate_proj,input,mlp);
        project(l->up_proj,input,up);
        for (int j = 0; j < 3584; j++) mlp[j] = silu(mlp[j])*up[j];
        project(l->down_proj,mlp,residual);
        for (int j = 0; j < hidden; j++) x[j] += residual[j];
    }
}

#ifdef SEGMENTED
int infer_segments(const WeightMap *map, void (*emit)(const uint8_t *, unsigned)) {
    if (validate_blocks(map)) return -1;
    model = map;
#else
int infer(const uint8_t *packed, unsigned long size, void (*emit)(const uint8_t *, unsigned)) {
    if (size != packed_bytes) return -1;
    weights = packed;
#endif
    int pos = 0;
    for (unsigned i = 0; i < sizeof(prompt_ids)/sizeof(*prompt_ids); i++)
        forward(prompt_ids[i],pos++);
    const unsigned *offsets = (const unsigned *)weight_at(index_offset);
    while (pos < context) {
        norm(x,input,hidden,final_norm,1);
        unsigned best = 0;
        float largest = -1e30f;
        for (unsigned i = 0; i < embedding.rows; i++) {
            float score = row_dot(embedding,i,input);
            if (score > largest) { largest = score; best = i; }
        }
        generated_ids[generated_count++] = best;
        if (best == 248044 || best == 248046) break;
        emit(weight_at(strings_offset+offsets[best]), offsets[best+1]-offsets[best]);
        forward(best,pos++);
    }
    return 0;
}
