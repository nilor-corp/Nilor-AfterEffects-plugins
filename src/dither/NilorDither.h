// Nilor Dither — ordered Bayer dithering for After Effects (CPU + DirectX 12 GPU)
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

#define NAME        "Nilor Dither"
#define DESCRIPTION "Ordered Bayer dithering (4x4/8x8), GPU-accelerated."

#define MAJOR_VERSION    1
#define MINOR_VERSION    0
#define BUG_VERSION      0
#define STAGE_VERSION    PF_Stage_RELEASE
#define BUILD_VERSION    1

enum
{
    DITHER_INPUT = 0,
    DITHER_LEVELS,
    DITHER_STRENGTH,
    DITHER_MATRIX,
    DITHER_ANIMATED,
    DITHER_SPEED,
    DITHER_NUM_PARAMS
};

// param ranges
#define LEVELS_MIN        2
#define LEVELS_MAX        32
#define LEVELS_DFLT       6
#define STRENGTH_MIN      0
#define STRENGTH_MAX      100
#define STRENGTH_DFLT     100
#define SPEED_MIN         0
#define SPEED_MAX         20
#define SPEED_DFLT        1

// kernel parameter block — field order must match the cbuffer in DitherKernel.hlsl
typedef struct
{
    int   mSrcPitch;
    int   mDstPitch;
    int   mWidth;
    int   mHeight;
    float mLevels;
    float mStrength;   // 0..1
    int   mMatrixSize; // 4 or 8
    float mPhase;      // 0..1
} DitherParams;

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
