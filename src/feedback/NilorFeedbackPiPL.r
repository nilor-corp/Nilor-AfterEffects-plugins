#include "AEConfig.h"
#include "AE_EffectVers.h"

#ifndef AE_OS_WIN
    #include "AE_General.r"
#endif

// flags: out_flags  = PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_DEEP_COLOR_AWARE = 0x2000004
//        out_flags2 = FLOAT_COLOR_AWARE | SMART_RENDER | MUTABLE_RENDER_SEQUENCE_DATA_SLOWER
//                     | GPU_RENDER_F32 | DIRECTX = 0x3A001400   (no THREADED: serial state)
resource 'PiPL' (16000) {
    {
        Kind { AEEffect },
        Name { "Nilor Feedback" },
        Category { "Nilor" },
#ifdef AE_OS_WIN
    #if defined(AE_PROC_INTELx64)
        CodeWin64X86 {"EffectMain"},
    #elif defined(AE_PROC_ARM64)
        CodeWinARM64 {"EffectMain"},
    #endif
#elif defined(AE_OS_MAC)
        CodeMacIntel64 {"EffectMain"},
        CodeMacARM64 {"EffectMain"},
#endif
        AE_PiPL_Version { 2, 0 },
        AE_Effect_Spec_Version { PF_PLUG_IN_VERSION, PF_PLUG_IN_SUBVERS },
        AE_Effect_Version { 525825 /* PF_VERSION(1,0,0,RELEASE,1), no VERS_HIGH bits */ },
        AE_Effect_Info_Flags { 0 },
        AE_Effect_Global_OutFlags { 0x2000004 },
        AE_Effect_Global_OutFlags_2 { 0x32001400 },
        AE_Effect_Match_Name { "Nilor Feedback" },
        AE_Reserved_Info { 0 },
        AE_Effect_Support_URL { "https://github.com/nilor-corp/Nilor-AfterEffects-plugins" }
    }
};
