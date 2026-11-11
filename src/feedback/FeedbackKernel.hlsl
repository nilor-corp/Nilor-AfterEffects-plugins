// Nilor Feedback — recursive frame feedback (compute shader, AE DirectX 12 GPU path)
//
//   out(t) = src(t)  OVER  (decay * transform(prevOut(t-1)))
//
// Two UAVs: the visible output AND the next accumulator get the same result (ping-pong;
// the read and write accumulator are different buffers, swapped by the host each frame).
// Bilinear sampling is manual: AE GPU worlds are raw row-padded float4 buffers, not textures.
//
// ABI (matches DirectXUtils.cpp DXShaderExecution binding order):
//   root param 0 = cbuffer params
//   root param 1 = UAV dst
//   root param 2 = UAV prevWrite
//   root param 3 = SRV src
//   root param 4 = SRV prevRead

struct FeedbackParams
{
    int   srcPitch;
    int   dstPitch;
    int   prevReadPitch;
    int   prevWritePitch;
    int   width;
    int   height;
    float m00, m01, m02;    // inverse transform about centre, row 0 (pixel space)
    float m10, m11, m12;    // inverse transform about centre, row 1
    float decay;
    int   hasPrev;
};

cbuffer Params : register(b0) { FeedbackParams p; }
RWStructuredBuffer<float4> dst       : register(u0);
RWStructuredBuffer<float4> prevWrite : register(u1);
StructuredBuffer<float4>   src       : register(t0);
StructuredBuffer<float4>   prevRead  : register(t1);

// explicit root signature: table order matches DXShaderExecution's CBV -> UAVs -> SRVs binding
#define NILOR_RS "DescriptorTable(CBV(b0)), DescriptorTable(UAV(u0, numDescriptors=2)), DescriptorTable(SRV(t0, numDescriptors=2))"

float4 readPrev(int x, int y)
{
    x = clamp(x, 0, p.width - 1);
    y = clamp(y, 0, p.height - 1);
    return prevRead[(uint)y * (uint)p.prevReadPitch + (uint)x];
}

float4 samplePrevBilinear(float fx, float fy)
{
    float2 pt = float2(fx - 0.5f, fy - 0.5f);
    int x0 = (int)floor(pt.x), y0 = (int)floor(pt.y);
    float tx = pt.x - (float)x0, ty = pt.y - (float)y0;
    float4 c00 = readPrev(x0,     y0);
    float4 c10 = readPrev(x0 + 1, y0);
    float4 c01 = readPrev(x0,     y0 + 1);
    float4 c11 = readPrev(x0 + 1, y0 + 1);
    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

[RootSignature(NILOR_RS)]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= (uint)p.width || tid.y >= (uint)p.height) return;

    float4 cur = src[tid.y * (uint)p.srcPitch + tid.x];
    float4 result = cur;

    if (p.hasPrev != 0)
    {
        float fx = (float)tid.x + 0.5f;
        float fy = (float)tid.y + 0.5f;
        float px = p.m00 * fx + p.m01 * fy + p.m02;
        float py = p.m10 * fx + p.m11 * fy + p.m12;
        float4 prev = samplePrevBilinear(px, py) * p.decay;
        // premultiplied 'over': current over decayed previous
        result = cur + prev * (1.0f - cur.a);
    }

    uint dIdx = tid.y * (uint)p.dstPitch + tid.x;
    dst[dIdx] = result;
    prevWrite[tid.y * (uint)p.prevWritePitch + tid.x] = result;
}
