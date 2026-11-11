// Nilor Feedback — recursive frame feedback for After Effects.
// CPU path (8/16/32f) + DirectX 12 compute path (GPU, float32).
//
// The accumulator (previous output frame) lives in sequence data:
//   - CPU path: a PF_Handle of float RGBA (tight pitch, converted per frame)
//   - GPU path: two PF_EffectWorld ping-pong buffers created with CreateGPUWorld,
//               read one + write the other inside the compute kernel, swap after.
// Continuity: only sequential forward rendering accumulates. Time jumps, size or
// device changes, and the Reset button all restart the accumulator at the current frame.

#include "NilorFeedback.h"

#include "AEFX_SuiteHandlerTemplate.h"
#include "AE_EffectGPUSuites.h"
#include "PrSDKPixelFormat.h"
#include "String_Utils.h"
#include "Smart_Utils.h"

#ifdef AE_OS_WIN
    #include "DirectXUtils.h"
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- helpers

static inline float Clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

// build the inverse of the user transform (translate + rotate + scale about the centre),
// mapping output pixel -> source pixel in the accumulator
static void BuildInverseTransform(
    float cx, float cy,          // centre (pixels)
    float offX, float offY,      // user offset (pixels)
    float rotDeg,                // user rotation
    float zoomPct,               // user zoom (100 = identity)
    float* m00, float* m01, float* m02,
    float* m10, float* m11, float* m12)
{
    double r = rotDeg * 3.14159265358979323846 / 180.0;
    double s = (zoomPct != 0.0f) ? (100.0 / zoomPct) : 1.0;   // inverse scale for the lookup
    double cs = cos(r) * s, sn = sin(r) * s;

    // forward: p' = R * S * (p - c) + c + off  →  inverse lookup: p = invS * R^T * (p' - c - off) + c
    // write as p = M * p' + t
    *m00 = (float)cs;  *m01 = (float)sn;
    *m10 = (float)-sn; *m11 = (float)cs;
    *m02 = (float)(cx - cs * (cx + offX) - sn * (cy + offY));
    *m12 = (float)(cy + sn * (cx + offX) - cs * (cy + offY));
}

// ---------------------------------------------------------------- effect commands

static PF_Err About(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    PF_SPRINTF(out_data->return_msg, "%s, v%d.%d\r%s", NAME, MAJOR_VERSION, MINOR_VERSION, DESCRIPTION);
    return PF_Err_NONE;
}

static PF_Err GlobalSetup(PF_InData* in_dataP, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;

    out_data->my_version = PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION, STAGE_VERSION, BUILD_VERSION);

    // temporal => NOT pix-independent; output varies with time even at constant params
    out_data->out_flags = PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_DEEP_COLOR_AWARE;

    // no THREADED_RENDERING on purpose: the accumulator is serial state
    out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_SMART_RENDER |
                           PF_OutFlag2_MUTABLE_RENDER_SEQUENCE_DATA_SLOWER;

    if (in_dataP->appl_id != 'PrMr')
    {
        out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 | PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING;
    }
    return err;
}

static PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Decay", FB_DECAY_MIN, FB_DECAY_MAX, FB_DECAY_MIN, FB_DECAY_MAX, FB_DECAY_DFLT,
                         1, PF_ValueDisplayFlag_PERCENT, 0, FB_DECAY);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Zoom", FB_ZOOM_MIN, FB_ZOOM_MAX, FB_ZOOM_MIN, FB_ZOOM_MAX, FB_ZOOM_DFLT,
                         1, PF_ValueDisplayFlag_PERCENT, 0, FB_ZOOM);

    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Rotate", 0, FB_ROTATE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Offset X", FB_OFF_MIN, FB_OFF_MAX, -200, 200, FB_OFF_DFLT,
                         1, PF_ValueDisplayFlag_NONE, 0, FB_OFFSET_X);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Offset Y", FB_OFF_MIN, FB_OFF_MAX, -200, 200, FB_OFF_DFLT,
                         1, PF_ValueDisplayFlag_NONE, 0, FB_OFFSET_Y);

    AEFX_CLR_STRUCT(def);
    PF_ADD_BUTTON("Reset Feedback", "Reset", FALSE, PF_ParamFlag_SUPERVISE, FB_RESET);

    out_data->num_params = FB_NUM_PARAMS;
    return err;
}

// ---------------------------------------------------------------- sequence data

static PF_Err SequenceSetup(PF_InData* in_dataP, PF_OutData* out_dataP)
{
    PF_Err err = PF_Err_NONE;
    if (!in_dataP->sequence_data)
    {
        AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
            AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1, out_dataP);
        PF_Handle h = handle_suite->host_new_handle(sizeof(FeedbackSeqData));
        if (h)
        {
            FeedbackSeqData* sd = (FeedbackSeqData*)handle_suite->host_lock_handle(h);
            memset(sd, 0, sizeof(FeedbackSeqData));
            sd->deviceIndex = -1;
            handle_suite->host_unlock_handle(h);
            in_dataP->sequence_data = h;
            out_dataP->out_flags |= PF_OutFlag_SEND_DO_DIALOG;   // not needed; keep flags minimal
            out_dataP->out_flags &= ~PF_OutFlag_SEND_DO_DIALOG;
        }
        else
        {
            err = PF_Err_OUT_OF_MEMORY;
        }
    }
    return err;
}

static PF_Err SequenceSetdown(PF_InData* in_dataP, PF_OutData* out_dataP)
{
    PF_Err err = PF_Err_NONE;
    if (in_dataP->sequence_data)
    {
        AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
            AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1, out_dataP);
        FeedbackSeqData* sd = (FeedbackSeqData*)handle_suite->host_lock_handle(in_dataP->sequence_data);
        if (sd->cpuPrevH) { handle_suite->host_dispose_handle(sd->cpuPrevH); sd->cpuPrevH = NULL; }
        if (sd->gpuPrev[0] || sd->gpuPrev[1])
        {
            AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite =
                AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_dataP);
            for (int k = 0; k < 2; k++)
            {
                if (sd->gpuPrev[k]) { gpu_suite->DisposeGPUWorld(in_dataP->effect_ref, sd->gpuPrev[k]); sd->gpuPrev[k] = NULL; }
            }
        }
        handle_suite->host_unlock_handle(in_dataP->sequence_data);
        handle_suite->host_dispose_handle(in_dataP->sequence_data);
        in_dataP->sequence_data = NULL;
    }
    return err;
}

// continuity check: sequential forward render of identically-shaped frames?
static PF_Boolean AccumulatorValid(FeedbackSeqData* sd, PF_InData* in_dataP, A_long w, A_long h)
{
    if (!sd->valid) return FALSE;
    if (sd->width != w || sd->height != h) return FALSE;
    if (sd->lastTime + sd->timeStep != in_dataP->current_time) return FALSE;
    if (sd->timeStep != in_dataP->time_step) return FALSE;
    return TRUE;
}

static void MarkRendered(FeedbackSeqData* sd, PF_InData* in_dataP, A_long w, A_long h)
{
    sd->valid = TRUE;
    sd->lastTime = in_dataP->current_time;
    sd->timeStep = in_dataP->time_step;
    sd->width = w;
    sd->height = h;
}

// ---------------------------------------------------------------- GPU (DirectX 12)

#ifdef AE_OS_WIN

struct FeedbackGPUData
{
    DXContextPtr    mContext;
    ShaderObjectPtr mFeedbackShader;
};

static PF_Err GPUDeviceSetup(PF_InData* in_dataP, PF_OutData* out_dataP, PF_GPUDeviceSetupExtra* extraP)
{
    PF_Err err = PF_Err_NONE;
    if (extraP->input->what_gpu != PF_GPU_Framework_DIRECTX) return PF_Err_UNRECOGNIZED_PARAM_TYPE;

    AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
        AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1, out_dataP);
    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpuDeviceSuite =
        AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_dataP);

    PF_GPUDeviceInfo device_info;
    AEFX_CLR_STRUCT(device_info);
    gpuDeviceSuite->GetDeviceInfo(in_dataP->effect_ref, extraP->input->device_index, &device_info);

    PF_Handle gpu_dataH = handle_suite->host_new_handle(sizeof(FeedbackGPUData));
    FeedbackGPUData* dx = reinterpret_cast<FeedbackGPUData*>(*gpu_dataH);
    memset(dx, 0, sizeof(FeedbackGPUData));

    dx->mContext = std::make_shared<DXContext>();
    dx->mFeedbackShader = std::make_shared<ShaderObject>();

    DX_ERR(dx->mContext->Initialize((ID3D12Device*)device_info.devicePV, (ID3D12CommandQueue*)device_info.command_queuePV));

    std::wstring csoPath, sigPath;
    DX_ERR(GetShaderPath(L"FeedbackKernel", csoPath, sigPath));
    DX_ERR(dx->mContext->LoadShader(csoPath.c_str(), sigPath.c_str(), dx->mFeedbackShader));

    extraP->output->gpu_data = gpu_dataH;
    out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    return err;
}

static PF_Err GPUDeviceSetdown(PF_InData* in_dataP, PF_OutData* out_dataP, PF_GPUDeviceSetdownExtra* extraP)
{
    PF_Err err = PF_Err_NONE;
    if (extraP->input->what_gpu == PF_GPU_Framework_DIRECTX)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        FeedbackGPUData* dx = reinterpret_cast<FeedbackGPUData*>(*gpu_dataH);
        if (dx)
        {
            dx->mContext->CloseWaitAndReset();
            dx->mContext.reset();
            dx->mFeedbackShader.reset();
        }
        AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
            AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1, out_dataP);
        handle_suite->host_dispose_handle(gpu_dataH);
    }
    return err;
}

// ensure the two GPU accumulator worlds exist and match the current shape/device
static PF_Err EnsureGpuAccumulator(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_GPUDeviceSuite1* gpu_suite,
    FeedbackSeqData* sd,
    A_long device_index,
    A_long w, A_long h,
    PF_PixelFormat fmt,
    A_short pixAR)
{
    PF_Err err = PF_Err_NONE;

    if (sd->onGpu && sd->deviceIndex == device_index && sd->width == w && sd->height == h &&
        sd->gpuPrev[0] && sd->gpuPrev[1])
    {
        return err; // reuse
    }

    // free any stale accumulator
    for (int k = 0; k < 2; k++)
    {
        if (sd->gpuPrev[k]) { gpu_suite->DisposeGPUWorld(in_dataP->effect_ref, sd->gpuPrev[k]); sd->gpuPrev[k] = NULL; }
    }
    if (sd->cpuPrevH)
    {
        AEFX_SuiteScoper<PF_HandleSuite1> hs =
            AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1, out_dataP);
        hs->host_dispose_handle(sd->cpuPrevH);
        sd->cpuPrevH = NULL;
    }

    for (int k = 0; k < 2 && !err; k++)
    {
        ERR(gpu_suite->CreateGPUWorld(
            in_dataP->effect_ref, device_index, w, h, pixAR, in_dataP->field, fmt, false, &sd->gpuPrev[k]));
    }
    sd->gpuCur = 0;
    sd->onGpu = TRUE;
    sd->deviceIndex = device_index;
    sd->valid = FALSE;   // fresh accumulator has no history
    return err;
}

#endif // AE_OS_WIN

// ---------------------------------------------------------------- CPU path

// convert any depth pixel row to float RGBA
static inline void ToFloat8(const PF_Pixel8* p, float* r, float* g, float* b, float* a)
{ *r = p->red / 255.f; *g = p->green / 255.f; *b = p->blue / 255.f; *a = p->alpha / 255.f; }
static inline void ToFloat16(const PF_Pixel16* p, float* r, float* g, float* b, float* a)
{ *r = p->red / 32768.f; *g = p->green / 32768.f; *b = p->blue / 32768.f; *a = p->alpha / 32768.f; }
static inline void ToFloat32(const PF_PixelFloat* p, float* r, float* g, float* b, float* a)
{ *r = p->red; *g = p->green; *b = p->blue; *a = p->alpha; }

// CPU accumulator stores float RGBA tightly packed
static float* CpuPrevPixel(FeedbackSeqData* sd, A_long x, A_long y)
{
    float* base = (float*)(*(sd->cpuPrevH));
    return base + ((size_t)y * (size_t)sd->width + (size_t)x) * 4;
}

static PF_Err FeedbackCPU(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    FeedbackSeqData* sd,
    float decay, float zoomPct, float rotDeg, float offX, float offY)
{
    PF_Err err = PF_Err_NONE;
    A_long w = output_worldP->width, h = output_worldP->height;

    AEFX_SuiteScoper<PF_HandleSuite1> hs =
        AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);

    // (re)allocate the accumulator if needed
    size_t needBytes = (size_t)w * (size_t)h * 4 * sizeof(float);
    if (!sd->cpuPrevH || sd->width != w || sd->height != h || sd->onGpu)
    {
        if (sd->cpuPrevH) { hs->host_dispose_handle(sd->cpuPrevH); sd->cpuPrevH = NULL; }
        sd->cpuPrevH = hs->host_new_handle(needBytes);
        if (!sd->cpuPrevH) return PF_Err_OUT_OF_MEMORY;
        sd->valid = FALSE;
        sd->onGpu = FALSE;
        sd->deviceIndex = -1;
    }

    PF_Boolean hasPrev = AccumulatorValid(sd, in_data, w, h);

    float m00, m01, m02, m10, m11, m12;
    BuildInverseTransform(w * 0.5f, h * 0.5f, offX, offY, rotDeg, zoomPct, &m00, &m01, &m02, &m10, &m11, &m12);

    for (A_long y = 0; y < h && !err; y++)
    {
        for (A_long x = 0; x < w; x++)
        {
            float r, g, b, a;
            A_long srcOff = y * input_worldP->rowbytes;
            A_long dstOff = y * output_worldP->rowbytes;
            switch (pixel_format)
            {
            case PF_PixelFormat_ARGB32:
                ToFloat8((PF_Pixel8*)((char*)input_worldP->data + srcOff) + x, &r, &g, &b, &a);
                break;
            case PF_PixelFormat_ARGB64:
                ToFloat16((PF_Pixel16*)((char*)input_worldP->data + srcOff) + x, &r, &g, &b, &a);
                break;
            case PF_PixelFormat_ARGB128:
                ToFloat32((PF_PixelFloat*)((char*)input_worldP->data + srcOff) + x, &r, &g, &b, &a);
                break;
            default:
                return PF_Err_BAD_CALLBACK_PARAM;
            }

            float or_ = r, og = g, ob = b, oa = a;
            if (hasPrev)
            {
                float fx = (float)x + 0.5f, fy = (float)y + 0.5f;
                float px = m00 * fx + m01 * fy + m02 - 0.5f;
                float py = m10 * fx + m11 * fy + m12 - 0.5f;
                int x0 = (int)floorf(px), y0 = (int)floorf(py);
                float tx = px - (float)x0, ty = py - (float)y0;
                int x1 = x0 + 1, y1 = y0 + 1;
                if (x0 < 0) x0 = 0; if (x0 > w - 1) x0 = w - 1;
                if (x1 < 0) x1 = 0; if (x1 > w - 1) x1 = w - 1;
                if (y0 < 0) y0 = 0; if (y0 > h - 1) y0 = h - 1;
                if (y1 < 0) y1 = 0; if (y1 > h - 1) y1 = h - 1;

                float* c00 = CpuPrevPixel(sd, x0, y0);
                float* c10 = CpuPrevPixel(sd, x1, y0);
                float* c01 = CpuPrevPixel(sd, x0, y1);
                float* c11 = CpuPrevPixel(sd, x1, y1);
                float pr = (c00[0] + (c10[0] - c00[0]) * tx);
                float pg = (c00[1] + (c10[1] - c00[1]) * tx);
                float pb = (c00[2] + (c10[2] - c00[2]) * tx);
                float pa = (c00[3] + (c10[3] - c00[3]) * tx);
                float qr = (c01[0] + (c11[0] - c01[0]) * tx);
                float qg = (c01[1] + (c11[1] - c01[1]) * tx);
                float qb = (c01[2] + (c11[2] - c01[2]) * tx);
                float qa = (c01[3] + (c11[3] - c01[3]) * tx);
                float vr = (pr + (qr - pr) * ty) * decay;
                float vg = (pg + (qg - pg) * ty) * decay;
                float vb = (pb + (qb - pb) * ty) * decay;
                float va = (pa + (qa - pa) * ty) * decay;

                // premultiplied over: current over decayed previous
                or_ = r + vr * (1.f - a);
                og = g + vg * (1.f - a);
                ob = b + vb * (1.f - a);
                oa = a + va * (1.f - a);
            }

            // write visible output
            switch (pixel_format)
            {
            case PF_PixelFormat_ARGB32: {
                PF_Pixel8* d = (PF_Pixel8*)((char*)output_worldP->data + dstOff) + x;
                d->red = (A_u_char)(Clamp01(or_) * 255.f + 0.5f);
                d->green = (A_u_char)(Clamp01(og) * 255.f + 0.5f);
                d->blue = (A_u_char)(Clamp01(ob) * 255.f + 0.5f);
                d->alpha = (A_u_char)(Clamp01(oa) * 255.f + 0.5f);
                break;
            }
            case PF_PixelFormat_ARGB64: {
                PF_Pixel16* d = (PF_Pixel16*)((char*)output_worldP->data + dstOff) + x;
                d->red = (A_u_short)(Clamp01(or_) * 32768.f);
                d->green = (A_u_short)(Clamp01(og) * 32768.f);
                d->blue = (A_u_short)(Clamp01(ob) * 32768.f);
                d->alpha = (A_u_short)(Clamp01(oa) * 32768.f);
                break;
            }
            case PF_PixelFormat_ARGB128: {
                PF_PixelFloat* d = (PF_PixelFloat*)((char*)output_worldP->data + dstOff) + x;
                d->red = or_; d->green = og; d->blue = ob; d->alpha = oa;
                break;
            }
            }

            // write the new accumulator (float RGBA)
            float* np = CpuPrevPixel(sd, x, y);
            np[0] = or_; np[1] = og; np[2] = ob; np[3] = oa;
        }
    }

    MarkRendered(sd, in_data, w, h);
    return err;
}

// ---------------------------------------------------------------- render plumbing

typedef struct
{
    float mDecay;
    float mZoom;
    float mRotate;
    float mOffX;
    float mOffY;
} FeedbackUserParams;

static PF_Err Render(PF_InData* in_dataP, PF_OutData* out_dataP, PF_ParamDef* params[], PF_LayerDef* output)
{
    // legacy non-smart path: no feedback state, just pass the input through
    PF_Err err = PF_Err_NONE;
    AEFX_SuiteScoper<PF_WorldTransformSuite1> wtP = AEFX_SuiteScoper<PF_WorldTransformSuite1>(
        in_dataP, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_dataP);
    ERR(wtP->copy_hq(in_dataP->effect_ref, &params[FB_INPUT]->u.ld, output, NULL, NULL));
    return err;
}

static PF_Err PreRender(PF_InData* in_dataP, PF_OutData* out_dataP, PF_PreRenderExtra* extraP)
{
    PF_Err err = PF_Err_NONE;

    extraP->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;

    FeedbackUserParams* infoP = (FeedbackUserParams*)malloc(sizeof(FeedbackUserParams));
    if (!infoP) return PF_Err_OUT_OF_MEMORY;

    PF_ParamDef cur_param;

    ERR(PF_CHECKOUT_PARAM(in_dataP, FB_DECAY, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) infoP->mDecay = (float)(cur_param.u.fs_d.value / 100.0);

    ERR(PF_CHECKOUT_PARAM(in_dataP, FB_ZOOM, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) infoP->mZoom = (float)cur_param.u.fs_d.value;

    ERR(PF_CHECKOUT_PARAM(in_dataP, FB_ROTATE, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) infoP->mRotate = (float)(cur_param.u.ad.value / 65536.0);

    ERR(PF_CHECKOUT_PARAM(in_dataP, FB_OFFSET_X, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) infoP->mOffX = (float)cur_param.u.fs_d.value;

    ERR(PF_CHECKOUT_PARAM(in_dataP, FB_OFFSET_Y, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) infoP->mOffY = (float)cur_param.u.fs_d.value;

    if (err) { free(infoP); return err; }

    extraP->output->pre_render_data = infoP;
    extraP->output->delete_pre_render_data_func = [](void* p) { free(p); };
    return err;
}

static PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extraP, bool isGPU)
{
    PF_Err err = PF_Err_NONE;
    PF_Err err2 = PF_Err_NONE;

    FeedbackUserParams* up = reinterpret_cast<FeedbackUserParams*>(extraP->input->pre_render_data);
    if (!up) return PF_Err_INTERNAL_STRUCT_DAMAGED;

    AEFX_SuiteScoper<PF_HandleSuite1> hs =
        AEFX_SuiteScoper<PF_HandleSuite1>(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
        AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1, out_data);

    if (!in_data->sequence_data) { ERR(SequenceSetup(in_data, out_data)); }
    if (!in_data->sequence_data) return PF_Err_INTERNAL_STRUCT_DAMAGED;

    FeedbackSeqData* sd = (FeedbackSeqData*)hs->host_lock_handle(in_data->sequence_data);

    PF_EffectWorld* input_worldP = NULL;
    PF_EffectWorld* output_worldP = NULL;

    ERR(extraP->cb->checkout_layer_pixels(in_data->effect_ref, FB_INPUT, &input_worldP));
    ERR(extraP->cb->checkout_output_pixels(in_data->effect_ref, &output_worldP));

    if (!err && input_worldP && output_worldP)
    {
        PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
        ERR(pixelFormatSuite->GetPixelFormat(output_worldP, &pixel_format));

#ifdef AE_OS_WIN
        if (isGPU)
        {
            err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
            if (pixel_format == PF_PixelFormat_GPU_BGRA128)
            {
                AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite = AEFX_SuiteScoper<PF_GPUDeviceSuite1>(
                    in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);

                err = EnsureGpuAccumulator(
                    in_data, out_data, *gpu_suite, sd, extraP->input->device_index,
                    input_worldP->width, input_worldP->height, pixel_format, input_worldP->pix_aspect_ratio);

                PF_Boolean hasPrev = FALSE;
                if (!err) hasPrev = AccumulatorValid(sd, in_data, input_worldP->width, input_worldP->height);

                if (!err)
                {
                    PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
                    FeedbackGPUData* dx = reinterpret_cast<FeedbackGPUData*>(*gpu_dataH);

                    FeedbackParams kp;
                    A_long bpp = 16;
                    kp.mSrcPitch = input_worldP->rowbytes / bpp;
                    kp.mDstPitch = output_worldP->rowbytes / bpp;
                    kp.mPrevReadPitch = sd->gpuPrev[sd->gpuCur]->rowbytes / bpp;
                    kp.mPrevWritePitch = sd->gpuPrev[1 - sd->gpuCur]->rowbytes / bpp;
                    kp.mWidth = input_worldP->width;
                    kp.mHeight = input_worldP->height;
                    kp.mDecay = up->mDecay;
                    kp.mHasPrev = hasPrev ? 1 : 0;
                    BuildInverseTransform(
                        kp.mWidth * 0.5f, kp.mHeight * 0.5f, up->mOffX, up->mOffY, up->mRotate, up->mZoom,
                        &kp.mM00, &kp.mM01, &kp.mM02, &kp.mM10, &kp.mM11, &kp.mM12);

                    void* src_mem = 0; void* dst_mem = 0;
                    void* prev_r = 0; void* prev_w = 0;
                    ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, input_worldP, &src_mem));
                    ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, output_worldP, &dst_mem));
                    ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, sd->gpuPrev[sd->gpuCur], &prev_r));
                    ERR(gpu_suite->GetGPUWorldData(in_data->effect_ref, sd->gpuPrev[1 - sd->gpuCur], &prev_w));

                    if (!err)
                    {
                        DXShaderExecution shaderExecution(dx->mContext, dx->mFeedbackShader, 5);
                        DX_ERR(shaderExecution.SetParamBuffer(&kp, sizeof(FeedbackParams)));
                        DX_ERR(shaderExecution.SetUnorderedAccessView((ID3D12Resource*)dst_mem, kp.mHeight * output_worldP->rowbytes));
                        DX_ERR(shaderExecution.SetUnorderedAccessView((ID3D12Resource*)prev_w, kp.mHeight * sd->gpuPrev[1 - sd->gpuCur]->rowbytes));
                        DX_ERR(shaderExecution.SetShaderResourceView((ID3D12Resource*)src_mem, kp.mHeight * input_worldP->rowbytes));
                        DX_ERR(shaderExecution.SetShaderResourceView((ID3D12Resource*)prev_r, kp.mHeight * sd->gpuPrev[sd->gpuCur]->rowbytes));
                        DX_ERR(shaderExecution.Execute((UINT)((kp.mWidth + 15) / 16), (UINT)((kp.mHeight + 15) / 16)));
                    }

                    if (!err)
                    {
                        sd->gpuCur = 1 - sd->gpuCur;   // ping-pong
                        MarkRendered(sd, in_data, kp.mWidth, kp.mHeight);
                    }
                }
            }
        }
        else
#endif
        {
            ERR(FeedbackCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, sd,
                            up->mDecay, up->mZoom, up->mRotate, up->mOffX, up->mOffY));
        }
    }

    hs->host_unlock_handle(in_data->sequence_data);

    if (input_worldP)
    {
        err2 = extraP->cb->checkin_layer_pixels(in_data->effect_ref, FB_INPUT);
    }
    if (!err) err = err2;
    return err;
}

// ---------------------------------------------------------------- events (Reset button)

static PF_Err UserChangedParam(PF_InData* in_dataP, PF_OutData* out_dataP, const PF_UserChangedParamExtra* extraP)
{
    PF_Err err = PF_Err_NONE;
    if (extraP->param_index == FB_RESET && in_dataP->sequence_data)
    {
        AEFX_SuiteScoper<PF_HandleSuite1> hs =
            AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1, out_dataP);
        FeedbackSeqData* sd = (FeedbackSeqData*)hs->host_lock_handle(in_dataP->sequence_data);
        sd->valid = FALSE;
        sd->lastTime = 0;
        hs->host_unlock_handle(in_dataP->sequence_data);
        out_dataP->out_flags |= PF_OutFlag_FORCE_RERENDER;
    }
    return err;
}

// ---------------------------------------------------------------- entry

PF_Err EffectMain(
    PF_Cmd          cmd,
    PF_InData*      in_dataP,
    PF_OutData*     out_data,
    PF_ParamDef*    params[],
    PF_LayerDef*    output,
    void*           extra)
{
    PF_Err err = PF_Err_NONE;

    try
    {
        switch (cmd)
        {
        case PF_Cmd_ABOUT:
            err = About(in_dataP, out_data, params, output);
            break;
        case PF_Cmd_GLOBAL_SETUP:
            err = GlobalSetup(in_dataP, out_data, params, output);
            break;
        case PF_Cmd_PARAMS_SETUP:
            err = ParamsSetup(in_dataP, out_data, params, output);
            break;
        case PF_Cmd_RENDER:
            err = Render(in_dataP, out_data, params, output);
            break;
        case PF_Cmd_SEQUENCE_SETUP:
        case PF_Cmd_SEQUENCE_RESETUP:
            err = SequenceSetup(in_dataP, out_data);
            break;
        case PF_Cmd_SEQUENCE_SETDOWN:
            err = SequenceSetdown(in_dataP, out_data);
            break;
        case PF_Cmd_USER_CHANGED_PARAM:
            err = UserChangedParam(in_dataP, out_data, (const PF_UserChangedParamExtra*)extra);
            break;
        case PF_Cmd_SMART_PRE_RENDER:
            err = PreRender(in_dataP, out_data, (PF_PreRenderExtra*)extra);
            break;
        case PF_Cmd_SMART_RENDER:
            err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, false);
            break;
        case PF_Cmd_SMART_RENDER_GPU:
            err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, true);
            break;
#ifdef AE_OS_WIN
        case PF_Cmd_GPU_DEVICE_SETUP:
            err = GPUDeviceSetup(in_dataP, out_data, (PF_GPUDeviceSetupExtra*)extra);
            break;
        case PF_Cmd_GPU_DEVICE_SETDOWN:
            err = GPUDeviceSetdown(in_dataP, out_data, (PF_GPUDeviceSetdownExtra*)extra);
            break;
#endif
        default:
            break;
        }
    }
    catch (PF_Err& thrown_err)
    {
        err = thrown_err;
    }
    catch (...)
    {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return err;
}
