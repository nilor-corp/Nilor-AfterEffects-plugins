// Nilor Dither — ordered Bayer dithering (compute shader, AE DirectX 12 GPU path)
//
// ABI (matches DirectXUtils.cpp DXShaderExecution binding order):
//   root param 0 = cbuffer params
//   root param 1 = UAV  (dst)
//   root param 2 = SRV  (src)
//
// AE GPU worlds are row-padded float4 buffers; pitch is in float4 elements.

struct DitherParams
{
    int   srcPitch;     // float4 elements per row, source
    int   dstPitch;     // float4 elements per row, destination
    int   width;
    int   height;
    float levels;       // quantization steps per channel (e.g. 6)
    float strength;     // 0..1 blend toward the dithered result
    int   matrixSize;   // 4 or 8 (Bayer matrix order)
    float phase;        // 0..1 animation phase (shifts the matrix pattern)
};

cbuffer Params : register(b0) { DitherParams p; }
RWStructuredBuffer<float4> dst : register(u0);
StructuredBuffer<float4>   src : register(t0);

// explicit root signature: table order matches DXShaderExecution's CBV -> UAV -> SRV binding
#define NILOR_RS "DescriptorTable(CBV(b0)), DescriptorTable(UAV(u0)), DescriptorTable(SRV(t0))"

static const float bayer4[16] = {
     0,  8,  2, 10,
    12,  4, 14,  6,
     3, 11,  1,  9,
    15,  7, 13,  5
};

static const float bayer8[64] = {
     0, 32,  8, 40,  2, 34, 10, 42,
    48, 16, 56, 24, 50, 18, 58, 26,
    12, 44,  4, 36, 14, 46,  6, 38,
    60, 28, 52, 20, 62, 30, 54, 22,
     3, 35, 11, 43,  1, 33,  9, 41,
    51, 19, 59, 27, 49, 17, 57, 25,
    15, 47,  7, 39, 13, 45,  5, 37,
    63, 31, 55, 23, 61, 29, 53, 21
};

float bayerAt(int x, int y, int size, float phase)
{
    int n2 = size * size;
    // animation: rotate the pattern index by the phase (golden-ratio scramble per frame)
    int shift = (int)(phase * (float)n2 * 1.61803398875f);
    int idx = ((y % size) * size + (x % size) + shift) % n2;
    if (idx < 0) idx += n2;
    float v = (size == 4) ? bayer4[idx] : bayer8[idx];
    return v / (float)n2;   // 0..1
}

float ditherChannel(float c, float threshold, float levels)
{
    float q = c * levels;
    float f = frac(q);
    float base = floor(q);
    return (base + (threshold > f ? 0.0f : 1.0f)) / levels;
}

[RootSignature(NILOR_RS)]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= (uint)p.width || tid.y >= (uint)p.height) return;

    uint sIdx = tid.y * (uint)p.srcPitch + tid.x;
    float4 c = src[sIdx];

    float t = bayerAt((int)tid.x, (int)tid.y, p.matrixSize, p.phase);

    float4 d;
    d.r = ditherChannel(c.r, t, p.levels);
    d.g = ditherChannel(c.g, t, p.levels);
    d.b = ditherChannel(c.b, t, p.levels);
    d.a = c.a;

    uint dIdx = tid.y * (uint)p.dstPitch + tid.x;
    dst[dIdx] = lerp(c, d, saturate(p.strength));
}
