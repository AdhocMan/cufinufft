#ifndef __CUDECONVOLVE_H__
#define __CUDECONVOLVE_H__

#include <cufinufft_eitherprec.h>

int CUDECONVOLVE1D(CUFINUFFT_PLAN d_mem, int blksize);
int CUDECONVOLVE2D(CUFINUFFT_PLAN d_mem, int blksize);
int CUDECONVOLVE3D(CUFINUFFT_PLAN d_mem, int blksize);
#endif
