#ifndef DS4_ROCM_H
#define DS4_ROCM_H

/*
 * ROCm compatibility layer for the existing graph scheduler.
 *
 * The C graph code is currently written against the ds4_metal_* tensor/kernel
 * ABI.  The ROCm backend implements that ABI with HIP so the scheduler can be
 * shared while kernels are ported one by one.
 */
#include "ds4_metal.h"

#endif
