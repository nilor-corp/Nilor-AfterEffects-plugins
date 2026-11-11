# Changelog

## Nilor Feedback 1.3 (in testing)
- Added: Trail Soften (in-loop separable gaussian on the accumulator, 0-100 px)
- Param order note for future versions: APPEND new params after FB_RESET (inserting
  mid-list triggers AE's "effect control conversion" warning on existing projects)

## Nilor Feedback 1.2
- Row-parallel composite (std::thread bands)
- 16-bit (ARGB64) cached frames — 4x smaller than float; long sequences stay cacheable

## Nilor Feedback 1.1
- Rewritten around AEGP_ComputeCacheSuite: frames cached globally keyed by
  (instance, time, params, input state, reset generation); MFR/viewer/RQ all correct
- Dropped per-instance sequence-data accumulator (broken by MFR design)

## Nilor Feedback 1.0 / Nilor Dither 1.0
- Initial CPU + DirectX 12 builds

## Parked (2026-11-11)
- Nilor Feedback uninstalled from AE 2027 (user call: iterating it took too long; using a
  different approach for the trails). Source remains in the repo; the last version (1.5)
  verified working in render-queue isolation tests. Nilor Dither stays installed and works.
