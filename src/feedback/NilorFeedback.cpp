// Nilor Feedback v1.1 — recursive frame feedback for After Effects.
//
//   out(t) = input(t)  OVER  (decay * transform(out(t-1)))     [premultiplied 'over']
//
// ARCHITECTURE (why this looks nothing like v1.0):
// AE's multi-frame rendering replicates sequence data per render thread, so a per-instance
// "previous frame" world can never work: frame N and N+1 land on different threads with
// different clones. The supported mechanism for cross-frame state is AEGP_ComputeCacheSuite:
// every rendered frame is cached globally (all threads, all clones) keyed by
// (effect instance, time, params, input content hash, reset generation). Rendering frame T
// checks out frame T-step, computing it recursively on a cache miss, so feedback is correct
// in the viewer, RAM preview and render queue, under MFR or not.
//
// v1.1 composites on CPU. The compute kernel is trivial (~1 multiply + lerp per pixel);
// a GPU variant can return later, but the cache value must be a CPU world anyway (values
// outlive render calls and get freed on arbitrary threads, which GPU worlds must not).
//
// Continuity contract: feedback accumulates only over sequential frames. Scrubbing, seeking,
// or the Reset button breaks the chain — the next frame starts fresh. Identical to the
// TouchDesigner feedbackTOP behaviour this replaces.

#include "NilorFeedback.h"

#include "AEFX_SuiteHandlerTemplate.h"
#include "AEGP_SuiteHandler.h"
#include "AE_ComputeCacheSuite.h"
#include "PrSDKPixelFormat.h"
#include "String_Utils.h"
#include "Smart_Utils.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------- params / state

typedef struct
{
    float mDecay;    // 0..1
    float mZoom;     // percent (100 = identity)
    float mRotate;   // degrees
    float mOffX;     // pixels
    float mOffY;     // pixels
} FeedbackUserParams;

// everything a cache entry for one frame depends on
typedef struct
{
    const PF_InData* in_dataP;    // valid during the SmartRender call that drives the compute chain
    A_long           time;
    A_long           timeStep;
    A_u_long         timeScale;
    A_long           generation;  // bump on Reset
    FeedbackUserParams params;
    PF_State         inputState;  // content identity of the source layer
    int              depth;       // recursion guard
} FBOptions;

// cached value: one composited frame as a 16-bit ARGB CPU world (4x smaller than float;
// a 70 s 4K sequence of float frames would be ~277 GB — the cache would purge and the chain
// would recompute. 16-bit keeps it cacheable and trails don't need float precision.)
typedef struct
{
    PF_EffectWorld* world;        // owned (struct malloc'd by us; pixels owned by AE until dispose)
    const SPBasicSuite* pica;     // for disposal on purge (any thread)
    A_long width, height;
} FBFrameValue;

// global data: the reset generation counter (shared; only written on the UI thread)
typedef struct
{
    A_long generation;
} FBGlobal;

static const char* kCacheClass = "com.nilor.feedback.frame.v1";

#define FB_MAX_DEPTH 4000   // recursion guard (70 s @ 60 fps); a longer chain resets instead

// ---------------------------------------------------------------- helpers

static inline float Clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

static void BuildInverseTransform(
    float cx, float cy, float offX, float offY, float rotDeg, float zoomPct,
    float* m00, float* m01, float* m02, float* m10, float* m11, float* m12)
{
    double r = rotDeg * 3.14159265358979323846 / 180.0;
    double s = (zoomPct != 0.0f) ? (100.0 / zoomPct) : 1.0;
    double cs = cos(r) * s, sn = sin(r) * s;

    *m00 = (float)cs;  *m01 = (float)sn;
    *m10 = (float)-sn; *m11 = (float)cs;
    *m02 = (float)(cx - cs * (cx + offX) - sn * (cy + offY));
    *m12 = (float)(cy + sn * (cx + offX) - cs * (cy + offY));
}

// FNV-1a over raw bytes into 128 bits
static void HashBytes(AEGP_CCComputeKeyP key, const void* data, size_t len, A_long salt)
{
    const unsigned char* p = (const unsigned char*)data;
    unsigned long long h0 = 1469598103934665603ULL ^ (unsigned long long)salt;
    unsigned long long h1 = 1099511628211ULL ^ (unsigned long long)(salt * 2654435761u);
    for (size_t i = 0; i < len; i++)
    {
        h0 = (h0 ^ p[i]) * 1099511628211ULL;
        h1 = (h1 ^ (p[i] + (unsigned char)(i & 7))) * 14029467366897019727ULL;
    }
    key->bytes[0] = (A_long)(h0 & 0xFFFFFFFF);
    key->bytes[1] = (A_long)(h0 >> 32);
    key->bytes[2] = (A_long)(h1 & 0xFFFFFFFF);
    key->bytes[3] = (A_long)(h1 >> 32);
}

// ---------------------------------------------------------------- pixel conversion

static inline void ToFloat8(const PF_Pixel8* p, float* px)
{ px[0] = p->red / 255.f; px[1] = p->green / 255.f; px[2] = p->blue / 255.f; px[3] = p->alpha / 255.f; }
static inline void ToFloat16(const PF_Pixel16* p, float* px)
{ px[0] = p->red / 32768.f; px[1] = p->green / 32768.f; px[2] = p->blue / 32768.f; px[3] = p->alpha / 32768.f; }
static inline void ToFloat32(const PF_PixelFloat* p, float* px)
{ px[0] = p->red; px[1] = p->green; px[2] = p->blue; px[3] = p->alpha; }

// read a pixel from any supported world as float RGBA
static inline void ReadWorldPixel(const PF_EffectWorld* w, PF_PixelFormat fmt, A_long x, A_long y, float* px)
{
    const char* row = (const char*)w->data + (size_t)y * (size_t)w->rowbytes;
    switch (fmt)
    {
    case PF_PixelFormat_ARGB32:  ToFloat8(((const PF_Pixel8*)row) + x, px); break;
    case PF_PixelFormat_ARGB64:  ToFloat16(((const PF_Pixel16*)row) + x, px); break;
    case PF_PixelFormat_ARGB128: ToFloat32(((const PF_PixelFloat*)row) + x, px); break;
    default: px[0] = px[1] = px[2] = px[3] = 0; break;
    }
}

// ---------------------------------------------------------------- the composite

// one band of rows of the composite. out is ARGB64 (16-bit) — cached frames are 4x smaller
// than float and trails don't need float precision.
static void CompositeRows(
    A_long y0, A_long y1,
    const PF_EffectWorld* curW, PF_PixelFormat curFmt,
    const PF_EffectWorld* prevW, PF_Boolean hasPrev,
    PF_EffectWorld* outW,
    const FeedbackUserParams* up,
    float m00, float m01, float m02, float m10, float m11, float m12)
{
    A_long w = outW->width;
    for (A_long y = y0; y < y1; y++)
    {
        PF_Pixel16* dst = (PF_Pixel16*)((char*)outW->data + (size_t)y * (size_t)outW->rowbytes);
        for (A_long x = 0; x < w; x++, dst++)
        {
            float cur[4];
            ReadWorldPixel(curW, curFmt, x, y, cur);

            float oR = cur[0], oG = cur[1], oB = cur[2], oA = cur[3];

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
                A_long h = outW->height;
                if (y0 < 0) y0 = 0; if (y0 > h - 1) y0 = h - 1;
                if (y1 < 0) y1 = 0; if (y1 > h - 1) y1 = h - 1;

                float c00[4], c10[4], c01[4], c11[4];
                ReadWorldPixel(prevW, PF_PixelFormat_ARGB64, x0, y0, c00);
                ReadWorldPixel(prevW, PF_PixelFormat_ARGB64, x1, y0, c10);
                ReadWorldPixel(prevW, PF_PixelFormat_ARGB64, x0, y1, c01);
                ReadWorldPixel(prevW, PF_PixelFormat_ARGB64, x1, y1, c11);

                for (int c = 0; c < 4; c++)
                {
                    float top = c00[c] + (c10[c] - c00[c]) * tx;
                    float bot = c01[c] + (c11[c] - c01[c]) * tx;
                    float v = (top + (bot - top) * ty) * up->mDecay;
                    if (c == 3) oA = cur[3] + v * (1.f - cur[3]);
                    else
                    {
                        float outc = cur[c] + v * (1.f - cur[3]);
                        if (c == 0) oR = outc; else if (c == 1) oG = outc; else oB = outc;
                    }
                }
            }

            dst->red   = (A_u_short)(Clamp01(oR) * 32768.f);
            dst->green = (A_u_short)(Clamp01(oG) * 32768.f);
            dst->blue  = (A_u_short)(Clamp01(oB) * 32768.f);
            dst->alpha = (A_u_short)(Clamp01(oA) * 32768.f);
        }
    }
}

// out = cur OVER (decay * transform(prev)), row-parallel
static void CompositeFrame(
    const PF_EffectWorld* curW, PF_PixelFormat curFmt,
    const PF_EffectWorld* prevW, PF_Boolean hasPrev,
    PF_EffectWorld* outW,
    const FeedbackUserParams* up)
{
    A_long w = outW->width, h = outW->height;
    float m00, m01, m02, m10, m11, m12;
    BuildInverseTransform(w * 0.5f, h * 0.5f, up->mOffX, up->mOffY, up->mRotate, up->mZoom,
                          &m00, &m01, &m02, &m10, &m11, &m12);

    unsigned nth = std::thread::hardware_concurrency();
    if (nth == 0) nth = 4;
    nth = std::min(nth, 16u);
    if ((unsigned)h < nth * 64) nth = std::max(1u, (unsigned)(h / 64));

    if (nth <= 1)
    {
        CompositeRows(0, h, curW, curFmt, prevW, hasPrev, outW, up, m00, m01, m02, m10, m11, m12);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(nth);
    A_long band = (h + (A_long)nth - 1) / (A_long)nth;
    for (unsigned t = 0; t < nth; t++)
    {
        A_long y0 = (A_long)t * band;
        A_long y1 = std::min(h, y0 + band);
        if (y0 >= y1) break;
        workers.emplace_back(CompositeRows, y0, y1, curW, curFmt, prevW, hasPrev, outW, up,
                             m00, m01, m02, m10, m11, m12);
    }
    for (auto& t : workers) t.join();
}

// ---------------------------------------------------------------- cache callbacks

static A_Err FBGenerateKey(AEGP_CCComputeOptionsRefconP optionsP, AEGP_CCComputeKeyP out_keyP)
{
    FBOptions* o = (FBOptions*)optionsP;
    // the key must capture: which instance, which frame, which params, what content, which reset era
    HashBytes(out_keyP, &o->in_dataP->effect_ref, sizeof(o->in_dataP->effect_ref), 11);
    HashBytes(out_keyP, &o->time, sizeof(o->time), out_keyP->bytes[0]);
    HashBytes(out_keyP, &o->timeStep, sizeof(o->timeStep), out_keyP->bytes[1]);
    HashBytes(out_keyP, &o->generation, sizeof(o->generation), out_keyP->bytes[2]);
    HashBytes(out_keyP, &o->params, sizeof(o->params), out_keyP->bytes[3]);
    HashBytes(out_keyP, &o->inputState, sizeof(o->inputState), 77);
    return A_Err_NONE;
}

// forward decl for recursion
static PF_Err FBComputeFrame(FBOptions* o, PF_EffectWorld** outWorldPP);

static A_Err FBCompute(AEGP_CCComputeOptionsRefconP optionsP, AEGP_CCComputeValueRefconP* out_valuePP)
{
    FBOptions* o = (FBOptions*)optionsP;
    PF_Err err = PF_Err_NONE;

    PF_EffectWorld* world = NULL;
    err = FBComputeFrame(o, &world);
    if (err || !world) return (A_Err)err;

    FBFrameValue* v = (FBFrameValue*)malloc(sizeof(FBFrameValue));
    if (!v) return A_Err_ALLOC;
    v->world = world;
    v->pica = o->in_dataP->pica_basicP;
    v->width = world->width;
    v->height = world->height;
    *out_valuePP = v;
    return A_Err_NONE;
}

static size_t FBApproxSize(AEGP_CCComputeValueRefconP valueP)
{
    FBFrameValue* v = (FBFrameValue*)valueP;
    return v ? (size_t)v->width * (size_t)v->height * 8 : 0;   // ARGB64
}

static void FBDeleteValue(AEGP_CCComputeValueRefconP valueP)
{
    FBFrameValue* v = (FBFrameValue*)valueP;
    if (!v) return;
    if (v->world && v->pica)
    {
        AEFX_SuiteScoper<PF_WorldSuite2> ws(const_cast<SPBasicSuite*>(v->pica), kPFWorldSuite, kPFWorldSuiteVersion2);
        ws->PF_DisposeWorld(NULL, v->world);
        free(v->world);   // the struct itself is ours (PF_NewWorld fills, doesn't allocate it)
    }
    free(v);
}

// render one frame into a fresh ARGB128 world (checkout input at that time, pull the
// previous frame from the cache — computing it recursively on a miss)
static PF_Err FBComputeFrame(FBOptions* o, PF_EffectWorld** outWorldPP)
{
    PF_Err err = PF_Err_NONE;
    const PF_InData* in_dataP = o->in_dataP;

    AEFX_SuiteScoper<PF_WorldSuite2> ws(const_cast<SPBasicSuite*>(in_dataP->pica_basicP), kPFWorldSuite, kPFWorldSuiteVersion2);
    AEFX_SuiteScoper<AEGP_ComputeCacheSuite1> cache(
        const_cast<SPBasicSuite*>(in_dataP->pica_basicP), kAEGPComputeCacheSuite, kAEGPComputeCacheSuiteVersion1);

    // current input at time o->time (checkout by param index works for arbitrary times)
    PF_ParamDef cur_def;
    AEFX_CLR_STRUCT(cur_def);
    ERR((in_dataP->inter.checkout_param)(
        in_dataP->effect_ref, FB_INPUT, o->time, o->timeStep, o->timeScale, &cur_def));
    if (err) return err;

    PF_EffectWorld* curW = &cur_def.u.ld;
    PF_PixelFormat curFmt = PF_PixelFormat_INVALID;
    ERR(ws->PF_GetPixelFormat(curW, &curFmt));

    A_long w = curW->width, h = curW->height;
    if (w <= 0 || h <= 0)
    {
        (in_dataP->inter.checkin_param)(in_dataP->effect_ref, &cur_def);
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    // previous frame from the cache
    PF_EffectWorld* prevW = NULL;
    PF_Boolean hasPrev = FALSE;
    AEGP_CCCheckoutReceiptP receipt = NULL;

    A_long prevTime = o->time - o->timeStep;
    if (prevTime >= 0 && o->depth < FB_MAX_DEPTH)
    {
        FBOptions prevO = *o;
        prevO.time = prevTime;
        prevO.depth = o->depth + 1;
        A_Err cerr = cache->AEGP_ComputeIfNeededAndCheckout(kCacheClass, &prevO, true, &receipt);
        if (cerr == A_Err_NONE && receipt)
        {
            FBFrameValue* prevV = NULL;
            cache->AEGP_GetReceiptComputeValue(receipt, (AEGP_CCComputeValueRefconP*)&prevV);
            if (prevV && prevV->world && prevV->world->width == w && prevV->world->height == h)
            {
                prevW = prevV->world;
                hasPrev = TRUE;
            }
        }
    }

    // composite into a fresh world (struct is ours; AE owns the pixels until we dispose)
    PF_EffectWorld* outW = (PF_EffectWorld*)malloc(sizeof(PF_EffectWorld));
    if (!outW)
    {
        if (receipt) cache->AEGP_CheckinComputeReceipt(receipt);
        (in_dataP->inter.checkin_param)(in_dataP->effect_ref, &cur_def);
        return PF_Err_OUT_OF_MEMORY;
    }
    ERR(ws->PF_NewWorld(in_dataP->effect_ref, w, h, FALSE, PF_PixelFormat_ARGB64, outW));
    if (!err)
    {
        CompositeFrame(curW, curFmt, prevW, hasPrev, outW, &o->params);
    }

    if (receipt) cache->AEGP_CheckinComputeReceipt(receipt);
    (in_dataP->inter.checkin_param)(in_dataP->effect_ref, &cur_def);

    if (!err && outW)
    {
        *outWorldPP = outW;
        return PF_Err_NONE;
    }
    if (outW) ws->PF_DisposeWorld(in_dataP->effect_ref, outW);
    return err ? err : PF_Err_INTERNAL_STRUCT_DAMAGED;
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
    out_data->out_flags = PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_DEEP_COLOR_AWARE;
    out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE | PF_OutFlag2_SUPPORTS_SMART_RENDER;

    // global data: the reset generation counter
    AEFX_SuiteScoper<PF_HandleSuite1> hs =
        AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    PF_Handle gh = hs->host_new_handle(sizeof(FBGlobal));
    if (gh)
    {
        FBGlobal* g = (FBGlobal*)hs->host_lock_handle(gh);
        g->generation = 0;
        hs->host_unlock_handle(gh);
        out_data->global_data = gh;
    }
    else
    {
        err = PF_Err_OUT_OF_MEMORY;
    }

    // register the frame cache
    if (!err)
    {
        AEFX_SuiteScoper<AEGP_ComputeCacheSuite1> cache =
            AEFX_SuiteScoper<AEGP_ComputeCacheSuite1>(in_dataP, kAEGPComputeCacheSuite, kAEGPComputeCacheSuiteVersion1, out_data);
        AEGP_ComputeCacheCallbacks cb;
        cb.generate_key = FBGenerateKey;
        cb.compute = FBCompute;
        cb.approx_size_value = FBApproxSize;
        cb.delete_compute_value = FBDeleteValue;
        A_Err rerr = cache->AEGP_ClassRegister(kCacheClass, &cb);
        if (rerr != A_Err_NONE) err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}

static PF_Err GlobalSetdown(PF_InData* in_dataP, PF_OutData* out_data)
{
    PF_Err err = PF_Err_NONE;
    if (in_dataP->global_data)
    {
        AEFX_SuiteScoper<PF_HandleSuite1> hs =
            AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
        hs->host_dispose_handle(in_dataP->global_data);
        in_dataP->global_data = NULL;
    }
    AEFX_SuiteScoper<AEGP_ComputeCacheSuite1> cache =
        AEFX_SuiteScoper<AEGP_ComputeCacheSuite1>(in_dataP, kAEGPComputeCacheSuite, kAEGPComputeCacheSuiteVersion1, out_data);
    cache->AEGP_ClassUnregister(kCacheClass);
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

static PF_Err Render(PF_InData* in_dataP, PF_OutData* out_dataP, PF_ParamDef* params[], PF_LayerDef* output)
{
    // legacy non-smart path: passthrough
    PF_Err err = PF_Err_NONE;
    AEFX_SuiteScoper<PF_WorldTransformSuite1> wtP = AEFX_SuiteScoper<PF_WorldTransformSuite1>(
        in_dataP, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_dataP);
    ERR(wtP->copy_hq(in_dataP->effect_ref, &params[FB_INPUT]->u.ld, output, NULL, NULL));
    return err;
}

static PF_Err PreRender(PF_InData* in_dataP, PF_OutData* out_dataP, PF_PreRenderExtra* extraP)
{
    PF_Err err = PF_Err_NONE;

    // no GPU_RENDER_POSSIBLE on v1.1: cache values are CPU worlds (see header comment)

    FeedbackUserParams* up = (FeedbackUserParams*)malloc(sizeof(FeedbackUserParams));
    if (!up) return PF_Err_OUT_OF_MEMORY;

    PF_ParamDef cur_param;

    ERR(PF_CHECKOUT_PARAM(in_dataP, FB_DECAY, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) up->mDecay = (float)(cur_param.u.fs_d.value / 100.0);

    ERR(PF_CHECKOUT_PARAM(in_dataP, FB_ZOOM, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) up->mZoom = (float)cur_param.u.fs_d.value;

    ERR(PF_CHECKOUT_PARAM(in_dataP, FB_ROTATE, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) up->mRotate = (float)(cur_param.u.ad.value / 65536.0);

    ERR(PF_CHECKOUT_PARAM(in_dataP, FB_OFFSET_X, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) up->mOffX = (float)cur_param.u.fs_d.value;

    ERR(PF_CHECKOUT_PARAM(in_dataP, FB_OFFSET_Y, in_dataP->current_time, in_dataP->time_step,
                          in_dataP->time_scale, &cur_param));
    if (!err) up->mOffY = (float)cur_param.u.fs_d.value;

    if (err) { free(up); return err; }

    extraP->output->pre_render_data = up;
    extraP->output->delete_pre_render_data_func = [](void* p) { free(p); };

    // AE requires these at SMART_PRE_RENDER
    PF_RenderRequest req = extraP->input->output_request;
    PF_CheckoutResult in_result;
    AEFX_CLR_STRUCT(in_result);
    ERR(extraP->cb->checkout_layer(in_dataP->effect_ref, FB_INPUT, FB_INPUT, &req,
                                   in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &in_result));
    if (!err)
    {
        UnionLRect(&in_result.result_rect, &extraP->output->result_rect);
        UnionLRect(&in_result.max_result_rect, &extraP->output->max_result_rect);
    }
    return err;
}

static PF_Err SmartRender(PF_InData* in_dataP, PF_OutData* out_dataP, PF_SmartRenderExtra* extraP, bool isGPU)
{
    PF_Err err = PF_Err_NONE;
    PF_Err err2 = PF_Err_NONE;

    FeedbackUserParams* up = reinterpret_cast<FeedbackUserParams*>(extraP->input->pre_render_data);
    if (!up) return PF_Err_INTERNAL_STRUCT_DAMAGED;

    AEFX_SuiteScoper<PF_HandleSuite1> hs =
        AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1, out_dataP);
    AEFX_SuiteScoper<PF_WorldSuite2> ws =
        AEFX_SuiteScoper<PF_WorldSuite2>(in_dataP, kPFWorldSuite, kPFWorldSuiteVersion2, out_dataP);
    AEFX_SuiteScoper<AEGP_ComputeCacheSuite1> cache =
        AEFX_SuiteScoper<AEGP_ComputeCacheSuite1>(in_dataP, kAEGPComputeCacheSuite, kAEGPComputeCacheSuiteVersion1, out_dataP);
    AEFX_SuiteScoper<PF_ParamUtilsSuite3> paramUtils =
        AEFX_SuiteScoper<PF_ParamUtilsSuite3>(in_dataP, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3, out_dataP);

    PF_EffectWorld* output_worldP = NULL;
    ERR(extraP->cb->checkout_output(in_dataP->effect_ref, &output_worldP));
    if (err || !output_worldP) return err ? err : PF_Err_INTERNAL_STRUCT_DAMAGED;

    // build the frame options
    FBOptions o;
    memset(&o, 0, sizeof(o));
    o.in_dataP = in_dataP;
    o.time = in_dataP->current_time;
    o.timeStep = in_dataP->time_step;
    o.timeScale = in_dataP->time_scale;
    o.params = *up;
    o.depth = 0;

    if (in_dataP->global_data)
    {
        FBGlobal* g = (FBGlobal*)hs->host_lock_handle(in_dataP->global_data);
        o.generation = g->generation;
        hs->host_unlock_handle(in_dataP->global_data);
    }

    ERR(paramUtils->PF_GetCurrentState(in_dataP->effect_ref, FB_INPUT, NULL, NULL, &o.inputState));

    if (!err)
    {
        AEGP_CCCheckoutReceiptP receipt = NULL;
        A_Err cerr = cache->AEGP_ComputeIfNeededAndCheckout(kCacheClass, &o, true, &receipt);
        if (cerr == A_Err_NONE && receipt)
        {
            FBFrameValue* v = NULL;
            cache->AEGP_GetReceiptComputeValue(receipt, (AEGP_CCComputeValueRefconP*)&v);
            if (v && v->world)
            {
                // blit the cached 16-bit frame into the output world (any depth)
                PF_PixelFormat outFmt = PF_PixelFormat_INVALID;
                ERR(ws->PF_GetPixelFormat(output_worldP, &outFmt));
                A_long w = v->world->width < output_worldP->width ? v->world->width : output_worldP->width;
                A_long h = v->world->height < output_worldP->height ? v->world->height : output_worldP->height;
                for (A_long y = 0; y < h && !err; y++)
                {
                    const PF_Pixel16* srcRow =
                        (const PF_Pixel16*)((const char*)v->world->data + (size_t)y * (size_t)v->world->rowbytes);
                    char* dstRow = (char*)output_worldP->data + (size_t)y * (size_t)output_worldP->rowbytes;
                    for (A_long x = 0; x < w; x++)
                    {
                        const PF_Pixel16* s = srcRow + x;
                        float fr = s->red / 32768.f, fg = s->green / 32768.f, fb = s->blue / 32768.f, fa = s->alpha / 32768.f;
                        switch (outFmt)
                        {
                        case PF_PixelFormat_ARGB32: {
                            PF_Pixel8* d = ((PF_Pixel8*)dstRow) + x;
                            d->red = (A_u_char)(Clamp01(fr) * 255.f + 0.5f);
                            d->green = (A_u_char)(Clamp01(fg) * 255.f + 0.5f);
                            d->blue = (A_u_char)(Clamp01(fb) * 255.f + 0.5f);
                            d->alpha = (A_u_char)(Clamp01(fa) * 255.f + 0.5f);
                            break;
                        }
                        case PF_PixelFormat_ARGB64: {
                            PF_Pixel16* d = ((PF_Pixel16*)dstRow) + x;
                            *d = *s;
                            break;
                        }
                        case PF_PixelFormat_ARGB128: {
                            PF_PixelFloat* d = ((PF_PixelFloat*)dstRow) + x;
                            d->red = fr; d->green = fg; d->blue = fb; d->alpha = fa;
                            break;
                        }
                        default:
                            err = PF_Err_BAD_CALLBACK_PARAM;
                            break;
                        }
                    }
                }
            }
            cache->AEGP_CheckinComputeReceipt(receipt);
        }
        else if (!err)
        {
            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
        }
    }

    if (output_worldP)
    {
        err2 = PF_Err_NONE; // output world needs no checkin
    }
    if (!err) err = err2;
    return err;
}

static PF_Err UserChangedParam(PF_InData* in_dataP, PF_OutData* out_dataP, const PF_UserChangedParamExtra* extraP)
{
    PF_Err err = PF_Err_NONE;
    if (extraP->param_index == FB_RESET && in_dataP->global_data)
    {
        AEFX_SuiteScoper<PF_HandleSuite1> hs =
            AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP, kPFHandleSuite, kPFHandleSuiteVersion1, out_dataP);
        FBGlobal* g = (FBGlobal*)hs->host_lock_handle(in_dataP->global_data);
        g->generation++;
        hs->host_unlock_handle(in_dataP->global_data);
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
        case PF_Cmd_GLOBAL_SETDOWN:
            err = GlobalSetdown(in_dataP, out_data);
            break;
        case PF_Cmd_PARAMS_SETUP:
            err = ParamsSetup(in_dataP, out_data, params, output);
            break;
        case PF_Cmd_RENDER:
            err = Render(in_dataP, out_data, params, output);
            break;
        case PF_Cmd_USER_CHANGED_PARAM:
            err = UserChangedParam(in_dataP, out_data, (const PF_UserChangedParamExtra*)extra);
            break;
        case PF_Cmd_SMART_PRE_RENDER:
            err = PreRender(in_dataP, out_data, (PF_PreRenderExtra*)extra);
            break;
        case PF_Cmd_SMART_RENDER:
            err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra*)extra, false);
            break;
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
        "Nilor Feedback",        // name
        "Nilor Feedback",        // match name
        "Nilor",                 // category
        AE_RESERVED_INFO,        // reserved info
        "EffectMain",            // entry point
        "https://github.com/nilor-corp/Nilor-AfterEffects-plugins"); // support URL
    return result;
}
