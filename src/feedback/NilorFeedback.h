// Nilor Feedback — true recursive frame feedback for After Effects (CPU + DirectX 12 GPU)
//
// TD feedbackTOP equivalent: output(t) = input(t) OVER transform(decay * output(t-1)).
// AE effects are stateless per frame; the recursion lives in per-instance sequence data
// (the previous output frame, kept on the GPU when the DirectX path is active).
//
// Continuity contract: feedback only accumulates when frames render sequentially forward.
// Any jump (scrub backwards, seek, size/device change) resets the accumulator to the
// current frame — same behaviour as TouchDesigner's feedback node on timeline jumps.
#pragma once

#include "AEConfig.h"
#include "entry.h"
#include "AEFX_SuiteHelper.h"
#include "PrSDKAESupport.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_Macros.h"
#include "AEGP_SuiteHandler.h"
#include "String_Utils.h"
#include "Param_Utils.h"
#include "Smart_Utils.h"
#include "PrSDKPixelFormat.h"

#ifdef AE_OS_WIN
    #include <Windows.h>
#endif

#define NAME        "Nilor Feedback"
#define DESCRIPTION "Recursive frame feedback: decay + transform the previous frame, composite under the current one."

#define MAJOR_VERSION    1
#define MINOR_VERSION    5
#define BUG_VERSION      0
#define STAGE_VERSION    PF_Stage_RELEASE
#define BUILD_VERSION    1

enum
{
    FB_INPUT = 0,
    FB_DECAY,
    FB_ZOOM,
    FB_ROTATE,
    FB_OFFSET_X,
    FB_OFFSET_Y,
    FB_SOFTEN,
    FB_RESET,
    FB_BAKE,    // keep LAST (append-only rule: no control-conversion warning on upgrade)
    FB_NUM_PARAMS
};

#define FB_DECAY_MIN     0
#define FB_DECAY_MAX     100
#define FB_DECAY_DFLT    90
#define FB_ZOOM_MIN      50
#define FB_ZOOM_MAX      150
#define FB_ZOOM_DFLT     100
#define FB_OFF_MIN      -2000
#define FB_OFF_MAX       2000
#define FB_OFF_DFLT      0
#define FB_SOFTEN_MIN    0
#define FB_SOFTEN_MAX    100
#define FB_SOFTEN_DFLT   0

// kernel parameter block — field order must match the cbuffer in FeedbackKernel.hlsl
typedef struct
{
    int   mSrcPitch;        // float4 elements per row, current input
    int   mDstPitch;        // float4 elements per row, output
    int   mPrevReadPitch;   // float4 elements per row, previous frame (read)
    int   mPrevWritePitch;  // float4 elements per row, previous frame (write)
    int   mWidth;
    int   mHeight;
    float mM00;             // inverse transform, row 0 (pixel space, about centre)
    float mM01;
    float mM02;
    float mM10;             // inverse transform, row 1
    float mM11;
    float mM12;
    float mDecay;           // 0..1
    int   mHasPrev;         // 0 = no valid previous frame (first frame / after reset)
} FeedbackParams;

// per-instance state (lives in sequence data)
typedef struct
{
    A_long      lastTime;       // current_time of the last frame we rendered
    A_long      timeStep;       // frame step at that render
    A_long      width, height;  // world size of the accumulator
    A_long      deviceIndex;    // GPU device the accumulator lives on (-1 = CPU)
    PF_Boolean  valid;          // accumulator holds a frame
    PF_Boolean  onGpu;          // accumulator is a pair of GPU worlds

    // CPU accumulator: PF_Handle to width*height*16 bytes of float RGBA
    PF_Handle   cpuPrevH;

    // GPU accumulator: ping-pong pair of PF_EffectWorld (read [cur], write [1-cur], then swap)
    PF_EffectWorld* gpuPrev[2];
    int             gpuCur;
} FeedbackSeqData;

#ifdef __cplusplus
extern "C" {
#endif

DllExport PF_Err EffectMain(
    PF_Cmd          cmd,
    PF_InData*      in_data,
    PF_OutData*     out_data,
    PF_ParamDef*    params[],
    PF_LayerDef*    output,
    void*           extra);

#ifdef __cplusplus
}
#endif
