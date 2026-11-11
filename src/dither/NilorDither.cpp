// Nilor Dither — ordered Bayer dithering for After Effects.
// CPU path (8/16/32f) + DirectX 12 compute path (GPU, float32).
// Modeled on the AE SDK 27 SDK_Invert_ProcAmp sample, Windows/DirectX-only.

#include "NilorDither.h"

#include "AEFX_SuiteHandlerTemplate.h"
#include "AE_EffectGPUSuites.h"
#include "PrSDKPixelFormat.h"
#include "String_Utils.h"
#include "Smart_Utils.h"

#ifdef AE_OS_WIN
    #include "DirectXUtils.h"
    // sample keeps this in its .cpp under #if HAS_HLSL; we are HLSL-only on Windows
    inline PF_Err DXErr(bool inSuccess) { return inSuccess ? PF_Err_NONE : PF_Err_INTERNAL_STRUCT_DAMAGED; }
    #define DX_ERR(FUNC) ERR(DXErr(FUNC))
#endif

#include <math.h>
#include <stdlib.h>

// ---------------------------------------------------------------- helpers

static const float kBayer4[16] = {
     0,  8,  2, 10,
    12,  4, 14,  6,
     3, 11,  1,  9,
    15,  7, 13,  5
};

static const float kBayer8[64] = {
     0, 32,  8, 40,  2, 34, 10, 42,
    48, 16, 56, 24, 50, 18, 58, 26,
    12, 44,  4, 36, 14, 46,  6, 38,
    60, 28, 52, 20, 62, 30, 54, 22,
     3, 35, 11, 43,  1, 33,  9, 41,
    51, 19, 59, 27, 49, 17, 57, 25,
    15, 47,  7, 39, 13, 45,  5, 37,
    63, 31, 55, 23, 61, 29, 53, 21
};

static float BayerAt(int x, int y, int size, float phase)
{
    int n2 = size * size;
    int shift = (int)(phase * (float)n2 * 1.61803398875f);
    int idx = ((y % size) * size + (x % size) + shift) % n2;
    if (idx < 0) idx += n2;
    float v = (size == 4) ? kBayer4[idx] : kBayer8[idx];
    return v / (float)n2;
}

static inline float DitherChannel(float c, float threshold, float levels)
{
    float q = c * levels;
    float f = q - floorf(q);
    float base = floorf(q);
    return (base + (threshold > f ? 0.0f : 1.0f)) / levels;
}

static inline float Clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }
static inline float LerpF(float a, float b, float t) { return a + (b - a) * t; }

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

    out_data->out_flags = PF_OutFlag_PIX_INDEPENDENT | PF_OutFlag_DEEP_COLOR_AWARE;

    out_data->out_flags2 =
        PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

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
    PF_ADD_FLOAT_SLIDERX(
        "Levels", LEVELS_MIN, LEVELS_MAX, LEVELS_MIN, LEVELS_MAX, LEVELS_DFLT,
        1, PF_ValueDisplayFlag_NONE, 0, DITHER_LEVELS);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Strength", STRENGTH_MIN, STRENGTH_MAX, STRENGTH_MIN, STRENGTH_MAX, STRENGTH_DFLT,
        1, PF_ValueDisplayFlag_PERCENT, 0, DITHER_STRENGTH);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Matrix", 2, 2, "4x4|8x8", DITHER_MATRIX);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Animated", FALSE, 0, DITHER_ANIMATED);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Speed", SPEED_MIN, SPEED_MAX, SPEED_MIN, SPEED_MAX, SPEED_DFLT,
        1, PF_ValueDisplayFlag_NONE, 0, DITHER_SPEED);

    out_data->num_params = DITHER_NUM_PARAMS;
    return err;
}

// ---------------------------------------------------------------- GPU (DirectX 12)

#ifdef AE_OS_WIN

struct DitherGPUData
{
    DXContextPtr      mContext;
    ShaderObjectPtr   mDitherShader;
};

static PF_Err GPUDeviceSetup(PF_InData* in_dataP, PF_OutData* out_dataP, PF_GPUDeviceSetupExtra* extraP)
{
    PF_Err err = PF_Err_NONE;

    if (extraP->input->what_gpu != PF_GPU_Framework_DIRECTX)
    {
        return PF_Err_UNRECOGNIZED_PARAM_TYPE;
    }

    AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
        AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1, out_dataP);
    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpuDeviceSuite =
        AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_dataP);

    PF_GPUDeviceInfo device_info;
    AEFX_CLR_STRUCT(device_info);
    gpuDeviceSuite->GetDeviceInfo(in_dataP->effect_ref, extraP->input->device_index, &device_info);

    PF_Handle gpu_dataH = handle_suite->host_new_handle(sizeof(DitherGPUData));
    DitherGPUData* dx_gpu_data = reinterpret_cast<DitherGPUData*>(*gpu_dataH);
    memset(dx_gpu_data, 0, sizeof(DitherGPUData));

    dx_gpu_data->mContext = std::make_shared<DXContext>();
    dx_gpu_data->mDitherShader = std::make_shared<ShaderObject>();

    DX_ERR(dx_gpu_data->mContext->Initialize(
        (ID3D12Device*)device_info.devicePV, (ID3D12CommandQueue*)device_info.command_queuePV));

    std::wstring csoPath, sigPath;
    DX_ERR(GetShaderPath(L"DitherKernel", csoPath, sigPath));
    DX_ERR(dx_gpu_data->mContext->LoadShader(csoPath.c_str(), sigPath.c_str(), dx_gpu_data->mDitherShader));

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
        DitherGPUData* dx_gpu_dataP = reinterpret_cast<DitherGPUData*>(*gpu_dataH);
        if (dx_gpu_dataP)
        {
            dx_gpu_dataP->mContext->CloseWaitAndReset();
            dx_gpu_dataP->mContext.reset();
            dx_gpu_dataP->mDitherShader.reset();
        }

        AEFX_SuiteScoper<PF_HandleSuite1> handle_suite =
            AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1, out_dataP);
        handle_suite->host_dispose_handle(gpu_dataH);
    }
    return err;
}

#endif // AE_OS_WIN

// ---------------------------------------------------------------- CPU kernels

static PF_Err FilterImage8(void* refcon, A_long xL, A_long yL, PF_Pixel8* inP, PF_Pixel8* outP)
{
    DitherParams* p = (DitherParams*)refcon;
    float t = BayerAt(xL, yL, p->mMatrixSize, p->mPhase);
    float s = p->mStrength;
    float L = p->mLevels;

    float r = inP->red / 255.f, g = inP->green / 255.f, b = inP->blue / 255.f;
    outP->red   = (A_u_char)(Clamp01(LerpF(r, DitherChannel(r, t, L), s)) * 255.f + 0.5f);
    outP->green = (A_u_char)(Clamp01(LerpF(g, DitherChannel(g, t, L), s)) * 255.f + 0.5f);
    outP->blue  = (A_u_char)(Clamp01(LerpF(b, DitherChannel(b, t, L), s)) * 255.f + 0.5f);
    outP->alpha = inP->alpha;
    return PF_Err_NONE;
}

static PF_Err FilterImage16(void* refcon, A_long xL, A_long yL, PF_Pixel16* inP, PF_Pixel16* outP)
{
    DitherParams* p = (DitherParams*)refcon;
    float t = BayerAt(xL, yL, p->mMatrixSize, p->mPhase);
    float s = p->mStrength;
    float L = p->mLevels;
    const float MX = 32768.f;

    float r = inP->red / MX, g = inP->green / MX, b = inP->blue / MX;
    outP->red   = (A_u_short)(Clamp01(LerpF(r, DitherChannel(r, t, L), s)) * MX);
    outP->green = (A_u_short)(Clamp01(LerpF(g, DitherChannel(g, t, L), s)) * MX);
    outP->blue  = (A_u_short)(Clamp01(LerpF(b, DitherChannel(b, t, L), s)) * MX);
    outP->alpha = inP->alpha;
    return PF_Err_NONE;
}

static PF_Err FilterImage32(void* refcon, A_long xL, A_long yL, PF_PixelFloat* inP, PF_PixelFloat* outP)
{
    DitherParams* p = (DitherParams*)refcon;
    float t = BayerAt(xL, yL, p->mMatrixSize, p->mPhase);
    float s = p->mStrength;
    float L = p->mLevels;

    outP->red   = LerpF(inP->red,   DitherChannel(inP->red,   t, L), s);
    outP->green = LerpF(inP->green, DitherChannel(inP->green, t, L), s);
    outP->blue  = LerpF(inP->blue,  DitherChannel(inP->blue,  t, L), s);
    outP->alpha = inP->alpha;
    return PF_Err_NONE;
}

// ---------------------------------------------------------------- render

static PF_Err Render(PF_InData* in_dataP, PF_OutData* out_dataP, PF_ParamDef* params[], PF_LayerDef* output)
{
    // non-SmartRender path (legacy hosts): iterate over the input using the checked-out params
    PF_Err err = PF_Err_NONE;

    DitherParams p;
    p.mSrcPitch = 0;
    p.mDstPitch = 0;
    p.mWidth = output->width;
    p.mHeight = output->height;
    p.mLevels = (float)params[DITHER_LEVELS]->u.fs_d.value;
    p.mStrength = (float)(params[DITHER_STRENGTH]->u.fs_d.value / 100.0);
    p.mMatrixSize = (params[DITHER_MATRIX]->u.pd.value == 1) ? 4 : 8;
    p.mPhase = 0.0f;
    if (params[DITHER_ANIMATED]->u.bd.value)
    {
        double seconds = (double)in_dataP->current_time / (double)in_dataP->time_scale;
        double ph = seconds * (double)params[DITHER_SPEED]->u.fs_d.value;
        p.mPhase = (float)(ph - floor(ph));
    }

    AEFX_SuiteScoper<PF_WorldSuite2> world_suite =
        AEFX_SuiteScoper<PF_WorldSuite2>(in_dataP, kPFWorldSuite, kPFWorldSuiteVersion2, out_dataP);
    PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
    ERR(world_suite->PF_GetPixelFormat(output, &pixel_format));

    switch (pixel_format)
    {
    case PF_PixelFormat_ARGB32:
    {
        AEFX_SuiteScoper<PF_Iterate8Suite2> iterate8Suite =
            AEFX_SuiteScoper<PF_Iterate8Suite2>(in_dataP, kPFIterate8Suite, kPFIterate8SuiteVersion2, out_dataP);
        ERR(iterate8Suite->iterate(in_dataP, 0, output->height, &params[DITHER_INPUT]->u.ld, NULL, (void*)&p,
                                   FilterImage8, output));
        break;
    }
    case PF_PixelFormat_ARGB64:
    {
        AEFX_SuiteScoper<PF_iterate16Suite2> iterate16Suite =
            AEFX_SuiteScoper<PF_iterate16Suite2>(in_dataP, kPFIterate16Suite, kPFIterate16SuiteVersion2, out_dataP);
        ERR(iterate16Suite->iterate(in_dataP, 0, output->height, &params[DITHER_INPUT]->u.ld, NULL, (void*)&p,
                                    FilterImage16, output));
        break;
    }
    case PF_PixelFormat_ARGB128:
    {
        AEFX_SuiteScoper<PF_iterateFloatSuite2> iterateFloatSuite = AEFX_SuiteScoper<PF_iterateFloatSuite2>(
            in_dataP, kPFIterateFloatSuite, kPFIterateFloatSuiteVersion2, out_dataP);
        ERR(iterateFloatSuite->iterate(in_dataP, 0, output->height, &params[DITHER_INPUT]->u.ld, NULL, (void*)&p,
                                       FilterImage32, output));
        break;
    }
    default:
        err = PF_Err_BAD_CALLBACK_PARAM;
        break;
    }
    return err;
}

static void DisposePreRenderData(void* pre_render_dataPV)
{
    if (pre_render_dataPV) free(pre_render_dataPV);
}

static PF_Err PreRender(PF_InData* in_dataP, PF_OutData* out_dataP, PF_PreRenderExtra* extraP)
{
    PF_Err err = PF_Err_NONE;

    extraP->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;

    DitherParams* infoP = (DitherParams*)malloc(sizeof(DitherParams));
    if (!infoP) return PF_Err_OUT_OF_MEMORY;
    memset(infoP, 0, sizeof(DitherParams));

    // params are queried at PreRender time and carried through pre_render_data
    PF_ParamDef cur_param;

    ERR(PF_CHECKOUT_PARAM(in_dataP, DITHER_LEVELS, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) infoP->mLevels = (float)cur_param.u.fs_d.value;

    ERR(PF_CHECKOUT_PARAM(in_dataP, DITHER_STRENGTH, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) infoP->mStrength = (float)(cur_param.u.fs_d.value / 100.0);   // slider is 0..100, kernel wants 0..1

    ERR(PF_CHECKOUT_PARAM(in_dataP, DITHER_MATRIX, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) infoP->mMatrixSize = (cur_param.u.pd.value == 1) ? 4 : 8;

    ERR(PF_CHECKOUT_PARAM(in_dataP, DITHER_ANIMATED, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    PF_Boolean animated = FALSE;
    if (!err) animated = cur_param.u.bd.value;

    ERR(PF_CHECKOUT_PARAM(in_dataP, DITHER_SPEED, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    float speed = 0.0f;
    if (!err) speed = (float)cur_param.u.fs_d.value;

    infoP->mPhase = 0.0f;
    if (!err && animated)
    {
        double seconds = (double)in_dataP->current_time / (double)in_dataP->time_scale;
        double ph = seconds * (double)speed;
        infoP->mPhase = (float)(ph - floor(ph));
    }

    if (err)
    {
        free(infoP);
        return err;
    }

    extraP->output->pre_render_data = infoP;
    extraP->output->delete_pre_render_data_func = DisposePreRenderData;

    return err;
}

static PF_Err SmartRenderCPU(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    DitherParams* infoP)
{
    PF_Err err = PF_Err_NONE;

    switch (pixel_format)
    {
    case PF_PixelFormat_ARGB128: {
        AEFX_SuiteScoper<PF_iterateFloatSuite2> iterateFloatSuite = AEFX_SuiteScoper<PF_iterateFloatSuite2>(
            in_data, kPFIterateFloatSuite, kPFIterateFloatSuiteVersion2, out_data);
        ERR(iterateFloatSuite->iterate(in_data, 0, output_worldP->height, input_worldP, NULL, (void*)infoP,
                                       FilterImage32, output_worldP));
        break;
    }
    case PF_PixelFormat_ARGB64: {
        AEFX_SuiteScoper<PF_iterate16Suite2> iterate16Suite =
            AEFX_SuiteScoper<PF_iterate16Suite2>(in_data, kPFIterate16Suite, kPFIterate16SuiteVersion2, out_data);
        ERR(iterate16Suite->iterate(in_data, 0, output_worldP->height, input_worldP, NULL, (void*)infoP,
                                    FilterImage16, output_worldP));
        break;
    }
    case PF_PixelFormat_ARGB32: {
        AEFX_SuiteScoper<PF_Iterate8Suite2> iterate8Suite =
            AEFX_SuiteScoper<PF_Iterate8Suite2>(in_data, kPFIterate8Suite, kPFIterate8SuiteVersion2, out_data);
        ERR(iterate8Suite->iterate(in_data, 0, output_worldP->height, input_worldP, NULL, (void*)infoP,
                                   FilterImage8, output_worldP));
        break;
    }
    default:
        err = PF_Err_BAD_CALLBACK_PARAM;
        break;
    }
    return err;
}

#ifdef AE_OS_WIN
static PF_Err SmartRenderGPU(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    DitherParams* infoP)
{
    PF_Err err = PF_Err_NONE;

    if (pixel_format != PF_PixelFormat_GPU_BGRA128)
    {
        return PF_Err_UNRECOGNIZED_PARAM_TYPE;
    }

    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite =
        AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_dataP);

    A_long bytes_per_pixel = 16;

    DitherParams kp = *infoP;
    kp.mSrcPitch = input_worldP->rowbytes / bytes_per_pixel;
    kp.mDstPitch = output_worldP->rowbytes / bytes_per_pixel;
    kp.mWidth = input_worldP->width;
    kp.mHeight = input_worldP->height;

    void* src_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, input_worldP, &src_mem));

    void* dst_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, output_worldP, &dst_mem));

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_DIRECTX)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        DitherGPUData* dx_gpu_data = reinterpret_cast<DitherGPUData*>(*gpu_dataH);

        DXShaderExecution shaderExecution(dx_gpu_data->mContext, dx_gpu_data->mDitherShader, 3);
        DX_ERR(shaderExecution.SetParamBuffer(&kp, sizeof(DitherParams)));
        DX_ERR(shaderExecution.SetUnorderedAccessView((ID3D12Resource*)dst_mem, kp.mHeight * output_worldP->rowbytes));
        DX_ERR(shaderExecution.SetShaderResourceView((ID3D12Resource*)src_mem, kp.mHeight * input_worldP->rowbytes));
        DX_ERR(shaderExecution.Execute(
            (UINT)((kp.mWidth + 15) / 16), (UINT)((kp.mHeight + 15) / 16)));
    }
    else
    {
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
    }

    return err;
}
#endif

static PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extraP, bool isGPU)
{
    PF_Err err = PF_Err_NONE;
    PF_Err err2 = PF_Err_NONE;

    DitherParams* infoP = reinterpret_cast<DitherParams*>(extraP->input->pre_render_data);
    if (!infoP)
    {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    AEFX_SuiteScoper<PF_WorldSuite2> world_suite =
        AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);

    PF_EffectWorld* input_worldP = NULL;
    PF_EffectWorld* output_worldP = NULL;

    ERR(extraP->cb->checkout_layer_pixels(in_data->effect_ref, DITHER_INPUT, &input_worldP));
    ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

    if (!err && input_worldP && output_worldP)
    {
        PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
        ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

#ifdef AE_OS_WIN
        if (isGPU)
        {
            ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
        }
        else
#endif
        {
            ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
        }
    }

    if (input_worldP)
    {
        err2 = extraP->cb->checkin_layer_pixels(in_data->effect_ref, DITHER_INPUT);
    }
    if (!err)
    {
        err = err2;
    }
    return err;
}

// ---------------------------------------------------------------- entry

#ifdef AE_OS_WIN
    #define DllExport __declspec(dllexport)
#endif

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
        case PF_Cmd_SMART_PRE_RENDER:
            err = PreRender(in_dataP, out_data, (PF_PreRenderExtra*)extra);
            break;
        case PF_Cmd_SMART_RENDER:
            err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra*)extra, false);
            break;
        case PF_Cmd_SMART_RENDER_GPU:
            err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra*)extra, true);
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

extern "C" DllExport PF_Err PluginDataEntryFunction2(
    PF_PluginDataPtr inPtr,
    PF_PluginDataCB2 inPluginDataCallBackPtr,
    SPBasicSuite* inSPBasicSuitePtr,
    const char* inHostName,
    const char* inHostVersion)
{
    PF_Err result = PF_Err_INVALID_CALLBACK;
    PF_REGISTER_EFFECT_EXT2(
        inPtr,
        inPluginDataCallBackPtr,
        "Nilor Dither",          // name
        "Nilor Dither",          // match name
        "Nilor",                 // category
        AE_RESERVED_INFO,        // reserved info
        "EffectMain",            // entry point
        "https://github.com/nilor-corp/Nilor-AfterEffects-plugins"); // support URL
    return result;
}
