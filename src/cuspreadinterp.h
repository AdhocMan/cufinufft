#ifndef __CUSPREADINTERP_H__
#define __CUSPREADINTERP_H__

#include <cufinufft_eitherprec.h>

#include <iostream>
#include <math.h>
#include <helper_cuda.h>
#include "cuda_hip_wrapper.h"
#include <thrust/extrema.h>
#include "../../contrib/utils.h"
#include "../../contrib/utils_fp.h"
#include "../../include/utils.h"


static __forceinline__ __device__
FLT evaluate_kernel(FLT x, FLT es_c, FLT es_beta, int ns)
/* ES ("exp sqrt") kernel evaluation at single real argument:
   phi(x) = exp(beta.sqrt(1 - (2x/n_s)^2)),    for |x| < nspread/2
   related to an asymptotic approximation to the Kaiser--Bessel, itself an
   approximation to prolate spheroidal wavefunction (PSWF) of order 0.
   This is the "reference implementation", used by eg common/onedim_* 
    2/17/17 */
{
	return abs(x) < ns/2.0 ? exp(es_beta * (sqrt(1.0 - es_c*x*x))) : 0.0;
}

static __inline__ __device__
void eval_kernel_vec_Horner(FLT *ker, const FLT x, const int w, 
	const double upsampfac)
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   This is the current evaluation method, since it's faster (except i7 w=16).
   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
	FLT z = 2*x + w - 1.0;         // scale so local grid offset z in [-1,1]
	// insert the auto-generated code which expects z, w args, writes to ker...
	if (upsampfac==2.0) {     // floating point equality is fine here
#include "../contrib/ker_horner_allw_loop.c"
	}
}

static __inline__ __device__
void eval_kernel_vec(FLT *ker, const FLT x, const double w, const double es_c, 
					 const double es_beta)
{
	for(int i=0; i<w; i++){
		ker[i] = evaluate_kernel(abs(x+i), es_c, es_beta, w);		
	}
}

namespace {
/* ------------------------ 1d Spreading Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */

__global__
void Spread_1d_NUptsdriven(FLT *x, CUCPX *c, CUCPX *fw, int M, const int ns, 
	int nf1, FLT es_c, FLT es_beta, int *idxnupts, int pirange)
{
	int xstart,xend;
	int xx, ix;
	FLT ker1[MAX_NSPREAD];

	FLT x_rescaled;
	FLT kervalue1;
	CUCPX cnow;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		cnow = c[idxnupts[i]];

		xstart = ceil(x_rescaled - ns/2.0);
		xend  = floor(x_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		for(xx=xstart; xx<=xend; xx++){
			ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
			kervalue1=ker1[xx-xstart];
			atomicAdd(&fw[ix].x, cnow.x*kervalue1);
			atomicAdd(&fw[ix].y, cnow.y*kervalue1);
		}
	}

}

__global__
void Spread_1d_NUptsdriven_Horner(FLT *x, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, FLT sigma, int* idxnupts, int pirange)
{
	int xx, ix;
	FLT ker1[MAX_NSPREAD];

	FLT x_rescaled;
	CUCPX cnow;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		cnow = c[idxnupts[i]];
		int xstart = ceil(x_rescaled - ns/2.0);
		int xend  = floor(x_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		eval_kernel_vec_Horner(ker1,x1,ns,sigma);
		for(xx=xstart; xx<=xend; xx++){
			ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
			FLT kervalue=ker1[xx-xstart];
			atomicAdd(&fw[ix].x, cnow.x*kervalue);
			atomicAdd(&fw[ix].y, cnow.y*kervalue);
		}
	}
}

/* Kernels for SubProb Method */
// SubProb properties
__global__
void CalcBinSize_noghost_1d(int M, int nf1, int  bin_size_x, int nbinx, 
	int* bin_size, FLT *x, int* sortidx, int pirange)
{
	int binx;
	int oldidx;
	FLT x_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		binx = binx < 0 ? 0 : binx;
		oldidx = atomicAdd(&bin_size[binx], 1);
		sortidx[i] = oldidx;
		if(binx >= nbinx){
			sortidx[i] = -binx;
		}
	}
}

__global__
void CalcInvertofGlobalSortIdx_1d(int M, int bin_size_x, int nbinx, 
	int* bin_startpts, int* sortidx, FLT *x, int* index, int pirange, int nf1)
{
	int binx;
	FLT x_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		binx = binx < 0 ? 0 : binx;

		index[bin_startpts[binx]+sortidx[i]] = i;
	}
}


__global__
void Spread_1d_Subprob(FLT *x, CUCPX *c, CUCPX *fw, int M, const int ns,
	int nf1, FLT es_c, FLT es_beta, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int* subprob_to_bin,
	int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, 
	int* idxnupts, int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,xend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;

	int N = (bin_size_x+2*ceil(ns/2.0));
	FLT ker1[MAX_NSPREAD];
	
	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		cnow = c[idxnupts[idx]];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;

		FLT x1=(FLT)xstart+xoffset - x_rescaled;
		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);

		for(int xx=xstart; xx<=xend; xx++){
			ix = xx+ceil(ns/2.0);
			if(ix >= (bin_size_x + (int) ceil(ns/2.0)*2) || ix<0) break;
			atomicAdd(&fwshared[ix].x, cnow.x*ker1[xx-xstart]);
			atomicAdd(&fwshared[ix].y, cnow.y*ker1[xx-xstart]);
		}
	}
	__syncthreads();
	/* write to global memory */
	for(int k=threadIdx.x; k<N; k+=blockDim.x){
		ix = xoffset-ceil(ns/2.0)+k;
		if(ix < (nf1+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			atomicAdd(&fw[ix].x, fwshared[k].x);
			atomicAdd(&fw[ix].y, fwshared[k].y);
		}
	}
}

__global__
void Spread_1d_Subprob_Horner(FLT *x, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, FLT sigma, int* binstartpts, int* bin_size, 
	int bin_size_x, int* subprob_to_bin, int* subprobstartpts, 
	int* numsubprob, int maxsubprobsize, int nbinx, int* idxnupts, int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,xend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;

	int N = (bin_size_x+2*ceil(ns/2.0));
	
	FLT ker1[MAX_NSPREAD];

	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		cnow = c[idxnupts[idx]];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		xend  = floor(x_rescaled + ns/2.0)-xoffset;

		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);

		for(int xx=xstart; xx<=xend; xx++){
			ix = xx+ceil(ns/2.0);
			if(ix >= (bin_size_x + (int) ceil(ns/2.0)*2) || ix<0) break;
			atomicAdd(&fwshared[ix].x, cnow.x*ker1[xx-xstart]);
			atomicAdd(&fwshared[ix].y, cnow.y*ker1[xx-xstart]);
		}
	}
	__syncthreads();

	/* write to global memory */
	for(int k=threadIdx.x; k<N; k+=blockDim.x){
		ix = xoffset-ceil(ns/2.0)+k;
		if(ix < (nf1+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			atomicAdd(&fw[ix].x, fwshared[k].x);
			atomicAdd(&fw[ix].y, fwshared[k].y);
		}
	}
}

/* --------------------- 1d Interpolation Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */
__global__
void Interp_1d_NUptsdriven(FLT *x, CUCPX *c, CUCPX *fw, int M, const int ns,
		       int nf1, FLT es_c, FLT es_beta, int* idxnupts, int pirange)
{
	FLT ker1[MAX_NSPREAD];
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		FLT x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
        
		int xstart = ceil(x_rescaled - ns/2.0);
		int xend  = floor(x_rescaled + ns/2.0);
		CUCPX cnow;
		cnow.x = 0.0;
		cnow.y = 0.0;

		FLT x1=(FLT)xstart-x_rescaled;
		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		for(int xx=xstart; xx<=xend; xx++){
			int ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
			FLT kervalue1 = ker1[xx-xstart];
			cnow.x += fw[ix].x*kervalue1;
			cnow.y += fw[ix].y*kervalue1;
		}
		c[idxnupts[i]].x = cnow.x;
		c[idxnupts[i]].y = cnow.y;
	}
}

__global__
void Interp_1d_NUptsdriven_Horner(FLT *x, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, FLT sigma, int* idxnupts, int pirange)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		FLT x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);

		int xstart = ceil(x_rescaled - ns/2.0);
		int xend  = floor(x_rescaled + ns/2.0);

		CUCPX cnow;
		cnow.x = 0.0;
		cnow.y = 0.0;
		FLT ker1[MAX_NSPREAD];

		eval_kernel_vec_Horner(ker1,xstart-x_rescaled,ns,sigma);

		for(int xx=xstart; xx<=xend; xx++){
			int ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
			cnow.x += fw[ix].x*ker1[xx-xstart];
			cnow.y += fw[ix].y*ker1[xx-xstart];
		}
		c[idxnupts[i]].x = cnow.x;
		c[idxnupts[i]].y = cnow.y;
	}

}

/* ---------------------- 3d Spreading Kernels -------------------------------*/
/* Kernels for bin sort NUpts */
__global__
void CalcBinSize_noghost_3d(int M, int nf1, int nf2, int nf3, int  bin_size_x,
	int bin_size_y, int bin_size_z, int nbinx, int nbiny, int nbinz,
    int* bin_size, FLT *x, FLT *y, FLT *z, int* sortidx, int pirange)
{
	int binidx, binx, biny, binz;
	int oldidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		binx = binx < 0 ? 0 : binx;

		biny = floor(y_rescaled/bin_size_y);
		biny = biny >= nbiny ? biny-1 : biny;
		biny = biny < 0 ? 0 : biny;

		binz = floor(z_rescaled/bin_size_z);
		binz = binz >= nbinz ? binz-1 : binz;
		binz = binz < 0 ? 0 : binz;
		binidx = binx+biny*nbinx+binz*nbinx*nbiny;
		oldidx = atomicAdd(&bin_size[binidx], 1);
		sortidx[i] = oldidx;
	}
}

__global__
void CalcInvertofGlobalSortIdx_3d(int M, int bin_size_x, int bin_size_y,
	int bin_size_z, int nbinx, int nbiny, int nbinz, int* bin_startpts,
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index, int pirange, int nf1,
	int nf2, int nf3)
{
	int binx,biny,binz;
	int binidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		binx = binx < 0 ? 0 : binx;
		biny = floor(y_rescaled/bin_size_y);
		biny = biny >= nbiny ? biny-1 : biny;
		biny = biny < 0 ? 0 : biny;
		binz = floor(z_rescaled/bin_size_z);
		binz = binz >= nbinz ? binz-1 : binz;
		binz = binz < 0 ? 0 : binz;
		binidx = CalcGlobalIdx_V2(binx,biny,binz,nbinx,nbiny,nbinz);

		index[bin_startpts[binidx]+sortidx[i]] = i;
	}
}

/* Kernels for NUptsdriven method */
__global__
void Spread_3d_NUptsdriven_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw,
	int M, const int ns, int nf1, int nf2, int nf3, FLT sigma, int* idxnupts,
	int pirange)
{
	int xx, yy, zz, ix, iy, iz;
	int outidx;
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker3[MAX_NSPREAD];

	FLT ker1val, ker2val, ker3val;

	FLT x_rescaled, y_rescaled, z_rescaled;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);
		z_rescaled=RESCALE(z[idxnupts[i]], nf3, pirange);

		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int zstart = ceil(z_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);
		int zend = floor(z_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		FLT y1=(FLT)ystart-y_rescaled;
		FLT z1=(FLT)zstart-z_rescaled;

		eval_kernel_vec_Horner(ker1,x1,ns,sigma);
		eval_kernel_vec_Horner(ker2,y1,ns,sigma);
		eval_kernel_vec_Horner(ker3,z1,ns,sigma);
		for(zz=zstart; zz<=zend; zz++){
			ker3val=ker3[zz-zstart];
			for(yy=ystart; yy<=yend; yy++){
				ker2val=ker2[yy-ystart];
				for(xx=xstart; xx<=xend; xx++){
					ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
					iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
					iz = zz < 0 ? zz+nf3 : (zz>nf3-1 ? zz-nf3 : zz);
					outidx = ix+iy*nf1+iz*nf1*nf2;
					ker1val=ker1[xx-xstart];
					FLT kervalue=ker1val*ker2val*ker3val;
					atomicAdd(&fw[outidx].x, c[idxnupts[i]].x*kervalue);
					atomicAdd(&fw[outidx].y, c[idxnupts[i]].y*kervalue);
				}
			}
		}
	}
}
__global__
void Spread_3d_NUptsdriven(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta,
	int* idxnupts, int pirange)
{
	int xx, yy, zz, ix, iy, iz;
	int outidx;
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker3[MAX_NSPREAD];

	FLT x_rescaled, y_rescaled, z_rescaled;
	FLT ker1val, ker2val, ker3val;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);
		z_rescaled=RESCALE(z[idxnupts[i]], nf3, pirange);

		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int zstart = ceil(z_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);
		int zend = floor(z_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		FLT y1=(FLT)ystart-y_rescaled;
		FLT z1=(FLT)zstart-z_rescaled;

		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		eval_kernel_vec(ker2,y1,ns,es_c,es_beta);
		eval_kernel_vec(ker3,z1,ns,es_c,es_beta);
		for(zz=zstart; zz<=zend; zz++){
			ker3val=ker3[zz-zstart];
			for(yy=ystart; yy<=yend; yy++){
				ker2val=ker2[yy-ystart];
				for(xx=xstart; xx<=xend; xx++){
					ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
					iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
					iz = zz < 0 ? zz+nf3 : (zz>nf3-1 ? zz-nf3 : zz);
					outidx = ix+iy*nf1+iz*nf1*nf2;

					ker1val=ker1[xx-xstart];
					FLT kervalue=ker1val*ker2val*ker3val;

					atomicAdd(&fw[outidx].x, c[idxnupts[i]].x*kervalue);
					atomicAdd(&fw[outidx].y, c[idxnupts[i]].y*kervalue);
				}
			}
		}
	}
}

/* Kernels for Subprob method */

__global__
void Spread_3d_Subprob_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int bin_size_z,
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob,
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts,
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend,zstart,zend;
	int bidx=subprob_to_bin[blockIdx.x];
	int binsubp_idx=blockIdx.x-subprobstartpts[bidx];
	int ix,iy,iz,outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=((bidx / nbinx)%nbiny)*bin_size_y;
	int zoffset=(bidx/ (nbinx*nbiny))*bin_size_z;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*
		(bin_size_z+2*ceil(ns/2.0));


	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();
	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;

	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		FLT ker1[MAX_NSPREAD];
		FLT ker2[MAX_NSPREAD];
		FLT ker3[MAX_NSPREAD];

		int nuptsidx = idxnupts[ptstart+i];
		x_rescaled = RESCALE(x[nuptsidx],nf1,pirange);
		y_rescaled = RESCALE(y[nuptsidx],nf2,pirange);
		z_rescaled = RESCALE(z[nuptsidx],nf3,pirange);
		cnow = c[nuptsidx];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;

		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;

		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart+yoffset-y_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker3,zstart+zoffset-z_rescaled,ns,sigma);

		for (int zz=zstart; zz<=zend; zz++){
			FLT kervalue3 = ker3[zz-zstart];
			iz = zz+ceil(ns/2.0);
			if(iz >= (bin_size_z + (int) ceil(ns/2.0)*2) || iz<0) break;
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				iy = yy+ceil(ns/2.0);
				if(iy >= (bin_size_y + (int) ceil(ns/2.0)*2) || iy<0) break;
				for(int xx=xstart; xx<=xend; xx++){
					ix = xx+ceil(ns/2.0);
					if(ix >= (bin_size_x + (int) ceil(ns/2.0)*2) || ix<0) break;
					outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2)+
						iz*(bin_size_x+ceil(ns/2.0)*2)*
						   (bin_size_y+ceil(ns/2.0)*2);
					FLT kervalue1 = ker1[xx-xstart];
					atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2*
						kervalue3);
					atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2*
						kervalue3);
				}
			}
		}
	}
	__syncthreads();
	/* write to global memory */
	for(int n=threadIdx.x; n<N; n+=blockDim.x){
		int i = n % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = (int) (n /(bin_size_x+2*ceil(ns/2.0))) %
				(int) (bin_size_y+2*ceil(ns/2.0));
		int k = n / ((bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0)));

		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		iz = zoffset-ceil(ns/2.0)+k;

		if(ix<(nf1+ceil(ns/2.0)) &&
		   iy<(nf2+ceil(ns/2.0)) &&
		   iz<(nf3+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			iz = iz < 0 ? iz+nf3 : (iz>nf3-1 ? iz-nf3 : iz);
			outidx = ix+iy*nf1+iz*nf1*nf2;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2)+
				k*(bin_size_x+ceil(ns/2.0)*2)*(bin_size_y+ceil(ns/2.0)*2);
			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
		}
	}
}

__global__
void Spread_3d_Subprob(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int bin_size_z,
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob, int maxsubprobsize,
	int nbinx, int nbiny, int nbinz, int* idxnupts, int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend,zstart,zend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix, iy, iz, outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=((bidx / nbinx)%nbiny)*bin_size_y;
	int zoffset=(bidx/ (nbinx*nbiny))*bin_size_z;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*
		(bin_size_z+2*ceil(ns/2.0));

	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();
	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		FLT ker1[MAX_NSPREAD];
		FLT ker2[MAX_NSPREAD];
		FLT ker3[MAX_NSPREAD];
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);
		z_rescaled=RESCALE(z[idxnupts[idx]], nf3, pirange);
		cnow = c[idxnupts[idx]];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;

		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;

		FLT x1=(FLT)xstart+xoffset-x_rescaled;
		FLT y1=(FLT)ystart+yoffset-y_rescaled;
		FLT z1=(FLT)zstart+zoffset-z_rescaled;

		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		eval_kernel_vec(ker2,y1,ns,es_c,es_beta);
		eval_kernel_vec(ker3,z1,ns,es_c,es_beta);
#if 1
		for(int zz=zstart; zz<=zend; zz++){
			FLT kervalue3 = ker3[zz-zstart];
			iz = zz+ceil(ns/2.0);
			if(iz >= (bin_size_z + (int) ceil(ns/2.0)*2) || iz<0) break;
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				iy = yy+ceil(ns/2.0);
				if(iy >= (bin_size_y + (int) ceil(ns/2.0)*2) || iy<0) break;
				for(int xx=xstart; xx<=xend; xx++){
					FLT kervalue1 = ker1[xx-xstart];
					ix = xx+ceil(ns/2.0);
					if(ix >= (bin_size_x + (int) ceil(ns/2.0)*2) || ix<0) break;
					outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2)+
							 iz*(bin_size_x+ceil(ns/2.0)*2)*
						        (bin_size_y+ceil(ns/2.0)*2);
#if 1
					atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2*
						kervalue3);
					atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2*
						kervalue3);
#endif
				}
			}
		}
#endif
	}
	__syncthreads();
	/* write to global memory */
	for(int n=threadIdx.x; n<N; n+=blockDim.x){
		int i = n % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = (int) (n /(bin_size_x+2*ceil(ns/2.0))) % (int) (bin_size_y+2*ceil(ns/2.0));
		int k = n / ((bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0)));

		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		iz = zoffset-ceil(ns/2.0)+k;
		if(ix<(nf1+ceil(ns/2.0)) && iy<(nf2+ceil(ns/2.0)) && iz<(nf3+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			iz = iz < 0 ? iz+nf3 : (iz>nf3-1 ? iz-nf3 : iz);
			outidx = ix+iy*nf1+iz*nf1*nf2;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2)+
				k*(bin_size_x+ceil(ns/2.0)*2)*(bin_size_y+ceil(ns/2.0)*2);
			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
		}
	}
}
/* Kernels for Block BlockGather Method */
__global__
void LocateNUptstoBins_ghost(int M, int  bin_size_x, int bin_size_y,
	int bin_size_z, int nobinx, int nobiny, int nobinz, int binsperobinx,
	int binsperobiny, int binsperobinz, int* bin_size, FLT *x, FLT *y, FLT *z,
	int* sortidx, int pirange, int nf1, int nf2, int nf3)
{
	int binidx,binx,biny,binz;
	int oldidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binz = floor(z_rescaled/bin_size_z);
		binx = binx/(binsperobinx-2)*binsperobinx + (binx%(binsperobinx-2)+1);
		biny = biny/(binsperobiny-2)*binsperobiny + (biny%(binsperobiny-2)+1);
		binz = binz/(binsperobinz-2)*binsperobinz + (binz%(binsperobinz-2)+1);

		binidx = CalcGlobalIdx(binx,biny,binz,nobinx,nobiny,nobinz,binsperobinx,
			binsperobiny, binsperobinz);
		oldidx = atomicAdd(&bin_size[binidx], 1);
		sortidx[i] = oldidx;
	}
}

__global__
void CalcInvertofGlobalSortIdx_ghost(int M, int  bin_size_x,
	int bin_size_y, int bin_size_z, int nobinx, int nobiny, int nobinz,
	int binsperobinx, int binsperobiny, int binsperobinz, int* bin_startpts,
	int* sortidx, FLT *x, FLT *y, FLT *z, int* index, int pirange, int nf1,
	int nf2, int nf3)
{
	int binx,biny,binz;
	int binidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		biny = floor(y_rescaled/bin_size_y);
		binz = floor(z_rescaled/bin_size_z);
		binx = binx/(binsperobinx-2)*binsperobinx + (binx%(binsperobinx-2)+1);
		biny = biny/(binsperobiny-2)*binsperobiny + (biny%(binsperobiny-2)+1);
		binz = binz/(binsperobinz-2)*binsperobinz + (binz%(binsperobinz-2)+1);

		binidx = CalcGlobalIdx(binx,biny,binz,nobinx,nobiny,nobinz,binsperobinx,
			binsperobiny, binsperobinz);
		index[bin_startpts[binidx]+sortidx[i]] = i;
	}
}


__global__
void Spread_3d_BlockGather(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, FLT sigma,
	int* binstartpts, int obin_size_x, int obin_size_y, int obin_size_z,
	int binsperobin, int* subprob_to_bin, int* subprobstartpts,
	int maxsubprobsize, int nobinx, int nobiny, int nobinz, int* idxnupts,
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,zstart,xend,yend,zend;
	int subpidx=blockIdx.x;
	int obidx=subprob_to_bin[subpidx];
	int bidx = obidx*binsperobin;

	int obinsubp_idx=subpidx-subprobstartpts[obidx];
	int ix, iy, iz;
	int outidx;
	int ptstart=binstartpts[bidx]+obinsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, binstartpts[bidx+binsperobin]-binstartpts[bidx]
			-obinsubp_idx*maxsubprobsize);

	int xoffset=(obidx % nobinx)*obin_size_x;
	int yoffset=(obidx / nobinx)%nobiny*obin_size_y;
	int zoffset=(obidx / (nobinx*nobiny))*obin_size_z;

	int N = obin_size_x*obin_size_y*obin_size_z;

	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();
	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		int b = idxnupts[idx]/M;
		int box[3];
		for(int d=0;d<3;d++){
			box[d] = b%3;
			if(box[d] == 1)
				box[d] = -1;
			if(box[d] == 2)
				box[d] = 1;
			b=b/3;
		}
		int ii = idxnupts[idx]%M;
		x_rescaled = RESCALE(x[ii],nf1,pirange) + box[0]*nf1;
		y_rescaled = RESCALE(y[ii],nf2,pirange) + box[1]*nf2;
		z_rescaled = RESCALE(z[ii],nf3,pirange) + box[2]*nf3;
		cnow = c[ii];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		xstart = xstart < 0 ? 0 : xstart;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		ystart = ystart < 0 ? 0 : ystart;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;
		zstart = zstart < 0 ? 0 : zstart;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		xend   = xend >= obin_size_x ? obin_size_x-1 : xend;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		yend   = yend >= obin_size_y ? obin_size_y-1 : yend;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;
		zend   = zend >= obin_size_z ? obin_size_z-1 : zend;

		for(int zz=zstart; zz<=zend; zz++){
			FLT disz=abs(z_rescaled-(zz+zoffset));
			FLT kervalue3 = evaluate_kernel(disz, es_c, es_beta, ns);
			for(int yy=ystart; yy<=yend; yy++){
				FLT disy=abs(y_rescaled-(yy+yoffset));
				FLT kervalue2 = evaluate_kernel(disy, es_c, es_beta, ns);
				for(int xx=xstart; xx<=xend; xx++){
					outidx = xx+yy*obin_size_x+zz*obin_size_y*obin_size_x;
					FLT disx=abs(x_rescaled-(xx+xoffset));
					FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta, ns);
					atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2*
						kervalue3);
					atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2*
						kervalue3);
				}
			}
		}
	}
	__syncthreads();
	/* write to global memory */
	for(int n=threadIdx.x; n<N; n+=blockDim.x){
		int i = n%obin_size_x;
		int j = (n/obin_size_x)%obin_size_y;
		int k = n/(obin_size_x*obin_size_y);

		ix = xoffset+i;
		iy = yoffset+j;
		iz = zoffset+k;
		outidx = ix+iy*nf1+iz*nf1*nf2;
		atomicAdd(&fw[outidx].x, fwshared[n].x);
		atomicAdd(&fw[outidx].y, fwshared[n].y);
	}
}

__global__
void Spread_3d_BlockGather_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta, FLT sigma,
	int* binstartpts, int obin_size_x, int obin_size_y, int obin_size_z,
	int binsperobin, int* subprob_to_bin, int* subprobstartpts,
	int maxsubprobsize, int nobinx, int nobiny, int nobinz, int* idxnupts,
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,zstart,xend,yend,zend;
	int xstartnew,ystartnew,zstartnew,xendnew,yendnew,zendnew;
	int subpidx=blockIdx.x;
	int obidx=subprob_to_bin[subpidx];
	int bidx = obidx*binsperobin;

	int obinsubp_idx=subpidx-subprobstartpts[obidx];
	int ix, iy, iz;
	int outidx;
	int ptstart=binstartpts[bidx]+obinsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, binstartpts[bidx+binsperobin]-binstartpts[bidx]
			-obinsubp_idx*maxsubprobsize);

	int xoffset=(obidx%nobinx)*obin_size_x;
	int yoffset=(obidx/nobinx)%nobiny*obin_size_y;
	int zoffset=(obidx/(nobinx*nobiny))*obin_size_z;

	int N = obin_size_x*obin_size_y*obin_size_z;

	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker3[MAX_NSPREAD];

	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int nidx = idxnupts[ptstart+i];
		int b = nidx/M;
		int box[3];
		for(int d=0;d<3;d++){
			box[d] = b%3;
			if(box[d] == 1)
				box[d] = -1;
			if(box[d] == 2)
				box[d] = 1;
			b=b/3;
		}
		int ii = nidx%M;
		x_rescaled = RESCALE(x[ii],nf1,pirange) + box[0]*nf1;
		y_rescaled = RESCALE(y[ii],nf2,pirange) + box[1]*nf2;
		z_rescaled = RESCALE(z[ii],nf3,pirange) + box[2]*nf3;
		cnow = c[ii];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;

		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart+yoffset-y_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker3,zstart+zoffset-z_rescaled,ns,sigma);

		xstartnew = xstart < 0 ? 0 : xstart;
		ystartnew = ystart < 0 ? 0 : ystart;
		zstartnew = zstart < 0 ? 0 : zstart;
		xendnew   = xend >= obin_size_x ? obin_size_x-1 : xend;
		yendnew   = yend >= obin_size_y ? obin_size_y-1 : yend;
		zendnew   = zend >= obin_size_z ? obin_size_z-1 : zend;

		for(int zz=zstartnew; zz<=zendnew; zz++){
			FLT kervalue3 = ker3[zz-zstart];
			for(int yy=ystartnew; yy<=yendnew; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				for(int xx=xstartnew; xx<=xendnew; xx++){
					outidx = xx+yy*obin_size_x+zz*obin_size_y*obin_size_x;
					FLT kervalue1 = ker1[xx-xstart];
					atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2*kervalue3);
					atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2*kervalue3);
				}
			}
		}
	}
	__syncthreads();
	/* write to global memory */
	for(int n=threadIdx.x; n<N; n+=blockDim.x){
		int i = n%obin_size_x;
		int j = (n/obin_size_x)%obin_size_y;
		int k = n/(obin_size_x*obin_size_y);

		ix = xoffset+i;
		iy = yoffset+j;
		iz = zoffset+k;
		outidx = ix+iy*nf1+iz*nf1*nf2;
		atomicAdd(&fw[outidx].x, fwshared[n].x);
		atomicAdd(&fw[outidx].y, fwshared[n].y);
	}
}

/* ---------------------- 3d Interpolation Kernels ---------------------------*/
/* Kernels for NUptsdriven Method */
__global__
void Interp_3d_NUptsdriven(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw, int M,
	const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta,
	int *idxnupts, int pirange)
{
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker3[MAX_NSPREAD];

	FLT ker1val, ker2val, ker3val;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		FLT x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		FLT y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);
		FLT z_rescaled=RESCALE(z[idxnupts[i]], nf3, pirange);
		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int zstart = ceil(z_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);
		int zend = floor(z_rescaled + ns/2.0);
		CUCPX cnow;
		cnow.x = 0.0;
		cnow.y = 0.0;
		FLT x1=(FLT)xstart-x_rescaled;
		FLT y1=(FLT)ystart-y_rescaled;
		FLT z1=(FLT)zstart-z_rescaled;

		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		eval_kernel_vec(ker2,y1,ns,es_c,es_beta);
		eval_kernel_vec(ker3,z1,ns,es_c,es_beta);
		for(int zz=zstart; zz<=zend; zz++){
			ker3val=ker3[zz-zstart];
			for(int yy=ystart; yy<=yend; yy++){
				ker2val=ker2[yy-ystart];
				for(int xx=xstart; xx<=xend; xx++){
					int ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
					int iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
					int iz = zz < 0 ? zz+nf3 : (zz>nf3-1 ? zz-nf3 : zz);

					int inidx = ix+iy*nf1+iz*nf2*nf1;

					ker1val=ker1[xx-xstart];
					cnow.x += fw[inidx].x*ker1val*ker2val*ker3val;
					cnow.y += fw[inidx].y*ker1val*ker2val*ker3val;
				}
			}
		}
		c[idxnupts[i]].x = cnow.x;
		c[idxnupts[i]].y = cnow.y;
	}

}

__global__
void Interp_3d_NUptsdriven_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw,
	int M, const int ns, int nf1, int nf2, int nf3, FLT sigma, int *idxnupts,
	int pirange)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		FLT x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		FLT y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);
		FLT z_rescaled=RESCALE(z[idxnupts[i]], nf3, pirange);

		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int zstart = ceil(z_rescaled - ns/2.0);

		int xend   = floor(x_rescaled + ns/2.0);
		int yend   = floor(y_rescaled + ns/2.0);
		int zend   = floor(z_rescaled + ns/2.0);

		CUCPX cnow;
		cnow.x = 0.0;
		cnow.y = 0.0;

		FLT ker1[MAX_NSPREAD];
		FLT ker2[MAX_NSPREAD];
		FLT ker3[MAX_NSPREAD];

		eval_kernel_vec_Horner(ker1,xstart-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart-y_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker3,zstart-z_rescaled,ns,sigma);

		for(int zz=zstart; zz<=zend; zz++){
			FLT kervalue3 = ker3[zz-zstart];
			int iz = zz < 0 ? zz+nf3 : (zz>nf3-1 ? zz-nf3 : zz);
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				int iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				for(int xx=xstart; xx<=xend; xx++){
					int ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
					int inidx = ix+iy*nf1+iz*nf2*nf1;
					FLT kervalue1 = ker1[xx-xstart];
					cnow.x += fw[inidx].x*kervalue1*kervalue2*kervalue3;
					cnow.y += fw[inidx].y*kervalue1*kervalue2*kervalue3;
				}
			}
		}
		c[idxnupts[i]].x = cnow.x;
		c[idxnupts[i]].y = cnow.y;
	}

}

/* Kernels for SubProb Method */
__global__
void Interp_3d_Subprob(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw,
	int M, const int ns, int nf1, int nf2, int nf3, FLT es_c, FLT es_beta,
	int* binstartpts, int* bin_size, int bin_size_x, int bin_size_y,
	int bin_size_z, int* subprob_to_bin, int* subprobstartpts, int* numsubprob,
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts,
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend,zstart,zend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix, iy, iz;
	int outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=((bidx / nbinx)%nbiny)*bin_size_y;
	int zoffset=(bidx/ (nbinx*nbiny))*bin_size_z;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*
			(bin_size_z+2*ceil(ns/2.0));

#if 1
	for(int n=threadIdx.x;n<N; n+=blockDim.x){
		int i = n % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = (int) (n /(bin_size_x+2*ceil(ns/2.0))) % (int) (bin_size_y+2*ceil(ns/2.0));
		int k = n / ((bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0)));

		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		iz = zoffset-ceil(ns/2.0)+k;
		if(ix<(nf1+ceil(ns/2.0)) && iy<(nf2+ceil(ns/2.0)) && iz<(nf3+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			iz = iz < 0 ? iz+nf3 : (iz>nf3-1 ? iz-nf3 : iz);
			outidx = ix+iy*nf1+iz*nf1*nf2;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2)+
				k*(bin_size_x+ceil(ns/2.0)*2)*(bin_size_y+ceil(ns/2.0)*2);
			fwshared[sharedidx].x = fw[outidx].x;
			fwshared[sharedidx].y = fw[outidx].y;
		}
	}
#endif
	__syncthreads();
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker3[MAX_NSPREAD];

	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);
		z_rescaled=RESCALE(z[idxnupts[idx]], nf3, pirange);
		cnow.x = 0.0;
		cnow.y = 0.0;
		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;

		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;

		FLT x1=(FLT)xstart+xoffset-x_rescaled;
		FLT y1=(FLT)ystart+yoffset-y_rescaled;
		FLT z1=(FLT)zstart+zoffset-z_rescaled;

		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		eval_kernel_vec(ker2,y1,ns,es_c,es_beta);
		eval_kernel_vec(ker3,z1,ns,es_c,es_beta);

		for (int zz=zstart; zz<=zend; zz++){
			FLT kervalue3 = ker3[zz-zstart];
			iz = zz+ceil(ns/2.0);
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				iy = yy+ceil(ns/2.0);
				for(int xx=xstart; xx<=xend; xx++){
					ix = xx+ceil(ns/2.0);
					outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2)+
						    iz*(bin_size_x+ceil(ns/2.0)*2)*
						       (bin_size_y+ceil(ns/2.0)*2);

					FLT kervalue1 = ker1[xx-xstart];
					cnow.x += fwshared[outidx].x*kervalue1*kervalue2*kervalue3;
					cnow.y += fwshared[outidx].y*kervalue1*kervalue2*kervalue3;
				}
			}
		}
		c[idxnupts[idx]].x = cnow.x;
		c[idxnupts[idx]].y = cnow.y;
	}
}
__global__
void Interp_3d_Subprob_Horner(FLT *x, FLT *y, FLT *z, CUCPX *c, CUCPX *fw,
	int M, const int ns, int nf1, int nf2, int nf3, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int bin_size_z,
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob,
	int maxsubprobsize, int nbinx, int nbiny, int nbinz, int* idxnupts,
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend,zstart,zend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix, iy, iz;
	int outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=((bidx / nbinx)%nbiny)*bin_size_y;
	int zoffset=(bidx/ (nbinx*nbiny))*bin_size_z;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0))*
			(bin_size_z+2*ceil(ns/2.0));

	for(int n=threadIdx.x;n<N; n+=blockDim.x){
		int i = n % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = (int) (n /(bin_size_x+2*ceil(ns/2.0))) % (int) (bin_size_y+2*ceil(ns/2.0));
		int k = n / ((bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0)));

		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		iz = zoffset-ceil(ns/2.0)+k;
		if(ix<(nf1+ceil(ns/2.0)) && iy<(nf2+ceil(ns/2.0)) && iz<(nf3+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			iz = iz < 0 ? iz+nf3 : (iz>nf3-1 ? iz-nf3 : iz);
			outidx = ix+iy*nf1+iz*nf1*nf2;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2)+
				k*(bin_size_x+ceil(ns/2.0)*2)*(bin_size_y+ceil(ns/2.0)*2);
			fwshared[sharedidx].x = fw[outidx].x;
			fwshared[sharedidx].y = fw[outidx].y;
		}
	}
	__syncthreads();
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker3[MAX_NSPREAD];
	FLT x_rescaled, y_rescaled, z_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);
		z_rescaled=RESCALE(z[idxnupts[idx]], nf3, pirange);
		cnow.x = 0.0;
		cnow.y = 0.0;

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		zstart = ceil(z_rescaled - ns/2.0)-zoffset;

		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;
		zend   = floor(z_rescaled + ns/2.0)-zoffset;

		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart+yoffset-y_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker3,zstart+zoffset-z_rescaled,ns,sigma);
		for (int zz=zstart; zz<=zend; zz++){
			FLT kervalue3 = ker3[zz-zstart];
			iz = zz+ceil(ns/2.0);
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2 = ker2[yy-ystart];
				iy = yy+ceil(ns/2.0);
				for(int xx=xstart; xx<=xend; xx++){
					ix = xx+ceil(ns/2.0);
					outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2)+
							 iz*(bin_size_x+ceil(ns/2.0)*2)*
							    (bin_size_y+ceil(ns/2.0)*2);
					FLT kervalue1 = ker1[xx-xstart];
					cnow.x += fwshared[outidx].x*kervalue1*kervalue2*kervalue3;
					cnow.y += fwshared[outidx].y*kervalue1*kervalue2*kervalue3;
				}
			}
		}
		c[idxnupts[idx]].x = cnow.x;
		c[idxnupts[idx]].y = cnow.y;
	}
}


/* ------------------------ 2d Spreading Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */

__global__
void Spread_2d_NUptsdriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
		const int ns, int nf1, int nf2, FLT es_c, FLT es_beta, int *idxnupts, 
		int pirange)
{
	int xstart,ystart,xend,yend;
	int xx, yy, ix, iy;
	int outidx;
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];

	FLT x_rescaled, y_rescaled;
	FLT kervalue1, kervalue2;
	CUCPX cnow;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);
		cnow = c[idxnupts[i]];

		xstart = ceil(x_rescaled - ns/2.0);
		ystart = ceil(y_rescaled - ns/2.0);
		xend = floor(x_rescaled + ns/2.0);
		yend = floor(y_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		FLT y1=(FLT)ystart-y_rescaled;
		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		eval_kernel_vec(ker2,y1,ns,es_c,es_beta);
		for(yy=ystart; yy<=yend; yy++){
			for(xx=xstart; xx<=xend; xx++){
				ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
				iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				outidx = ix+iy*nf1;
				kervalue1=ker1[xx-xstart];
				kervalue2=ker2[yy-ystart];
				atomicAdd(&fw[outidx].x, cnow.x*kervalue1*kervalue2);
				atomicAdd(&fw[outidx].y, cnow.y*kervalue1*kervalue2);
			}
		}

	}

}

__global__
void Spread_2d_NUptsdriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT sigma, int* idxnupts, int pirange)
{
	int xx, yy, ix, iy;
	int outidx;
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	FLT ker1val, ker2val;

	FLT x_rescaled, y_rescaled;
	CUCPX cnow;
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);
		cnow = c[idxnupts[i]];
		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);

		FLT x1=(FLT)xstart-x_rescaled;
		FLT y1=(FLT)ystart-y_rescaled;
		eval_kernel_vec_Horner(ker1,x1,ns,sigma);
		eval_kernel_vec_Horner(ker2,y1,ns,sigma);
		for(yy=ystart; yy<=yend; yy++){
			for(xx=xstart; xx<=xend; xx++){
				ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
				iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				outidx = ix+iy*nf1;
				ker1val=ker1[xx-xstart];
				ker2val=ker2[yy-ystart];
				FLT kervalue=ker1val*ker2val;
				atomicAdd(&fw[outidx].x, cnow.x*kervalue);
				atomicAdd(&fw[outidx].y, cnow.y*kervalue);
			}
		}
	}
}

/* Kernels for SubProb Method */
// SubProb properties
__global__
void CalcBinSize_noghost_2d(int M, int nf1, int nf2, int  bin_size_x, 
	int bin_size_y, int nbinx, int nbiny, int* bin_size, FLT *x, FLT *y, 
	int* sortidx, int pirange)
{
	int binidx, binx, biny;
	int oldidx;
	FLT x_rescaled,y_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		binx = binx < 0 ? 0 : binx;
		biny = floor(y_rescaled/bin_size_y);
		biny = biny >= nbiny ? biny-1 : biny;
		biny = biny < 0 ? 0 : biny;
		binidx = binx+biny*nbinx;
		oldidx = atomicAdd(&bin_size[binidx], 1);
		sortidx[i] = oldidx;
		if(binx >= nbinx || biny >= nbiny){
			sortidx[i] = -biny;
		}
	}
}

__global__
void CalcInvertofGlobalSortIdx_2d(int M, int bin_size_x, int bin_size_y, 
	int nbinx,int nbiny, int* bin_startpts, int* sortidx, FLT *x, FLT *y, 
	int* index, int pirange, int nf1, int nf2)
{
	int binx, biny;
	int binidx;
	FLT x_rescaled, y_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		binx = binx < 0 ? 0 : binx;
		biny = floor(y_rescaled/bin_size_y);
		biny = biny >= nbiny ? biny-1 : biny;
		biny = biny < 0 ? 0 : biny;
		binidx = binx+biny*nbinx;

		index[bin_startpts[binidx]+sortidx[i]] = i;
	}
}


__global__
void Spread_2d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
	int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
	int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, 
	int nbiny, int* idxnupts, int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix, iy;
	int outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=(bidx / nbinx)*bin_size_y;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0));
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];
	
	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled, y_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);
		cnow = c[idxnupts[idx]];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;

		FLT x1=(FLT)xstart+xoffset - x_rescaled;
		FLT y1=(FLT)ystart+yoffset - y_rescaled;
		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		eval_kernel_vec(ker2,y1,ns,es_c,es_beta);

		for(int yy=ystart; yy<=yend; yy++){
			iy = yy+ceil(ns/2.0);
			if(iy >= (bin_size_y + (int) ceil(ns/2.0)*2) || iy<0) break;
			for(int xx=xstart; xx<=xend; xx++){
				ix = xx+ceil(ns/2.0);
				if(ix >= (bin_size_x + (int) ceil(ns/2.0)*2) || ix<0) break;
				outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2);
				FLT kervalue1 = ker1[xx-xstart];
				FLT kervalue2 = ker2[yy-ystart];
				atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2);
				atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2);
			}
		}
	}
	__syncthreads();
	/* write to global memory */
	for(int k=threadIdx.x; k<N; k+=blockDim.x){
		int i = k % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = k /( bin_size_x+2*ceil(ns/2.0) );
		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		if(ix < (nf1+ceil(ns/2.0)) && iy < (nf2+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			outidx = ix+iy*nf1;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
		}
	}
}

__global__
void Spread_2d_Subprob_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT sigma, int* binstartpts, int* bin_size, 
	int bin_size_x, int bin_size_y, int* subprob_to_bin, int* subprobstartpts, 
	int* numsubprob, int maxsubprobsize, int nbinx, int nbiny, int* idxnupts, 
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix, iy, outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=(bidx / nbinx)*bin_size_y;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0));
	
	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];


	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled, y_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);
		cnow = c[idxnupts[idx]];

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;

		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart+yoffset-y_rescaled,ns,sigma);

		for(int yy=ystart; yy<=yend; yy++){
			iy = yy+ceil(ns/2.0);
			if(iy >= (bin_size_y + (int) ceil(ns/2.0)*2) || iy<0) break;
			FLT kervalue2 = ker2[yy-ystart];
			for(int xx=xstart; xx<=xend; xx++){
				ix = xx+ceil(ns/2.0);
				if(ix >= (bin_size_x + (int) ceil(ns/2.0)*2) || ix<0) break;
				outidx = ix+iy*(bin_size_x+ (int) ceil(ns/2.0)*2);
				FLT kervalue1 = ker1[xx-xstart];
				atomicAdd(&fwshared[outidx].x, cnow.x*kervalue1*kervalue2);
				atomicAdd(&fwshared[outidx].y, cnow.y*kervalue1*kervalue2);
			}
		}
	}
	__syncthreads();

	/* write to global memory */
	for(int k=threadIdx.x; k<N; k+=blockDim.x){
		int i = k % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = k /( bin_size_x+2*ceil(ns/2.0) );
		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		if(ix < (nf1+ceil(ns/2.0)) && iy < (nf2+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			outidx = ix+iy*nf1;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
		}
	}
}

/* Kernels for Paul's Method */
__global__
void LocateFineGridPos_Paul(int M, int nf1, int nf2, int  bin_size_x, 
	int bin_size_y, int nbinx, int nbiny, int* bin_size, int ns, FLT *x, FLT *y, 
	int* sortidx, int* finegridsize, int pirange)
{
	int binidx, binx, biny;
	int oldidx;
	int xidx, yidx, finegrididx;
	FLT x_rescaled,y_rescaled;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		if(ns%2 == 0){
			x_rescaled=RESCALE(x[i], nf1, pirange);
			y_rescaled=RESCALE(y[i], nf2, pirange);
			binx = floor(floor(x_rescaled)/bin_size_x);
			biny = floor(floor(y_rescaled)/bin_size_y);
			binidx = binx+biny*nbinx;
			xidx = floor(x_rescaled) - binx*bin_size_x;
			yidx = floor(y_rescaled) - biny*bin_size_y;
			finegrididx = binidx*bin_size_x*bin_size_y + xidx + yidx*bin_size_x;
		}else{
			x_rescaled=RESCALE(x[i], nf1, pirange);
			y_rescaled=RESCALE(y[i], nf2, pirange);
			xidx = ceil(x_rescaled - 0.5);
			yidx = ceil(y_rescaled - 0.5);
			
			//xidx = (xidx == nf1) ? (xidx-nf1) : xidx;
			//yidx = (yidx == nf2) ? (yidx-nf2) : yidx;

			binx = floor(xidx/(float) bin_size_x);
			biny = floor(yidx/(float) bin_size_y);
			binidx = binx+biny*nbinx;

			xidx = xidx - binx*bin_size_x;
			yidx = yidx - biny*bin_size_y;
			finegrididx = binidx*bin_size_x*bin_size_y + xidx + yidx*bin_size_x;
		}
		oldidx = atomicAdd(&finegridsize[finegrididx], 1);
		sortidx[i] = oldidx;
	}
}

__global__
void CalcInvertofGlobalSortIdx_Paul(int nf1, int nf2, int M, int bin_size_x, 
		int bin_size_y, int nbinx,int nbiny,int ns, FLT *x, FLT *y, 
		int* finegridstartpts, int* sortidx, int* index, int pirange)
{
	FLT x_rescaled, y_rescaled;
	int binx, biny, binidx, xidx, yidx, finegrididx;
	for(int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x){
		if(ns%2 == 0){
			x_rescaled=RESCALE(x[i], nf1, pirange);
			y_rescaled=RESCALE(y[i], nf2, pirange);
			binx = floor(floor(x_rescaled)/bin_size_x);
			biny = floor(floor(y_rescaled)/bin_size_y);
			binidx = binx+biny*nbinx;
			xidx = floor(x_rescaled) - binx*bin_size_x;
			yidx = floor(y_rescaled) - biny*bin_size_y;
			finegrididx = binidx*bin_size_x*bin_size_y + xidx + yidx*bin_size_x;
		}else{
			x_rescaled=RESCALE(x[i], nf1, pirange);
			y_rescaled=RESCALE(y[i], nf2, pirange);
			xidx = ceil(x_rescaled - 0.5);
			yidx = ceil(y_rescaled - 0.5);
			
			xidx = (xidx == nf1) ? xidx - nf1 : xidx;
			yidx = (yidx == nf2) ? yidx - nf2 : yidx;

			binx = floor(xidx/(float) bin_size_x);
			biny = floor(yidx/(float) bin_size_y);
			binidx = binx+biny*nbinx;

			xidx = xidx - binx*bin_size_x;
			yidx = yidx - biny*bin_size_y;
			finegrididx = binidx*bin_size_x*bin_size_y + xidx + yidx*bin_size_x;
		}
		index[finegridstartpts[finegrididx]+sortidx[i]] = i;
	}
}


__global__
void Spread_2d_Subprob_Paul(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, 
	int* binstartpts, int* bin_size, int bin_size_x, int bin_size_y, 
	int* subprob_to_bin, int* subprobstartpts, int* numsubprob, 
	int maxsubprobsize, int nbinx, int nbiny, int* idxnupts, int* fgstartpts,
	int* finegridsize, int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];

	int ix,iy,outidx;

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=(bidx / nbinx)*bin_size_y;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0));
#if 0
	FLT ker1[MAX_NSPREAD*10];
    FLT ker2[MAX_NSPREAD*10];
#endif
	for(int i=threadIdx.x; i<N; i+=blockDim.x){
		fwshared[i].x = 0.0;
		fwshared[i].y = 0.0;
	}
	__syncthreads();

	FLT x_rescaled, y_rescaled;
	for(int i=threadIdx.x; i<bin_size_x*bin_size_y; i+=blockDim.x){
		int fineidx = bidx*bin_size_x*bin_size_y+i;
		int idxstart = fgstartpts[fineidx]+binsubp_idx*maxsubprobsize;
		int nupts = min(maxsubprobsize,finegridsize[fineidx]-binsubp_idx*
			maxsubprobsize);
		if(nupts > 0){
			x_rescaled = x[idxnupts[idxstart]];
			y_rescaled = y[idxnupts[idxstart]];

			xstart = ceil(x_rescaled - ns/2.0)-xoffset;
			ystart = ceil(y_rescaled - ns/2.0)-yoffset;
			xend   = floor(x_rescaled + ns/2.0)-xoffset;
			yend   = floor(y_rescaled + ns/2.0)-yoffset;
#if 0
			for(int m=0; m<nupts; m++){
				int idx = idxstart+m;
				x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
				y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);

				eval_kernel_vec_Horner(ker1+m*MAX_NSPREAD,xstart+xoffset-
					x_rescaled,ns,sigma);
				eval_kernel_vec_Horner(ker2+m*MAX_NSPREAD,ystart+yoffset-
					y_rescaled,ns,sigma);
			}
#endif
			for(int yy=ystart; yy<=yend; yy++){
				FLT kervalue2[10];
				for(int m=0; m<nupts; m++){
					int idx = idxstart+m;
#if 1 
					y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);
					FLT disy = abs(y_rescaled-(yy+yoffset));
					kervalue2[m] = evaluate_kernel(disy, es_c, es_beta, ns);
#else
					kervalue2[m] = ker2[m*MAX_NSPREAD+yy-ystart];
#endif
				}
				for(int xx=xstart; xx<=xend; xx++){
					ix = xx+ceil(ns/2.0);
					iy = yy+ceil(ns/2.0);
					outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2);
					CUCPX updatevalue;
					updatevalue.x = 0.0;
					updatevalue.y = 0.0;
					for(int m=0; m<nupts; m++){
						int idx = idxstart+m;
#if 1
						x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
						FLT disx = abs(x_rescaled-(xx+xoffset));
						FLT kervalue1 = evaluate_kernel(disx, es_c, es_beta, ns);

						updatevalue.x += kervalue2[m]*kervalue1*
										 c[idxnupts[idx]].x;
						updatevalue.y += kervalue2[m]*kervalue1*
										 c[idxnupts[idx]].y;
#else
						FLT kervalue1 = ker1[m*MAX_NSPREAD+xx-xstart];
						updatevalue.x += kervalue1*kervalue2[m]*
							c[idxnupts[idx]].x;
						updatevalue.y += kervalue1*kervalue2[m]*
							c[idxnupts[idx]].y;
#endif
					}
					atomicAdd(&fwshared[outidx].x, updatevalue.x);
					atomicAdd(&fwshared[outidx].y, updatevalue.y);
				}
			}
		}
	}
	__syncthreads();

	/* write to global memory */
	for(int k=threadIdx.x; k<N; k+=blockDim.x){
		int i = k % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = k /( bin_size_x+2*ceil(ns/2.0) );
		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		if(ix < (nf1+ceil(ns/2.0)) && iy < (nf2+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			outidx = ix+iy*nf1;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
			atomicAdd(&fw[outidx].x, fwshared[sharedidx].x);
			atomicAdd(&fw[outidx].y, fwshared[sharedidx].y);
		}
	}
}
/* --------------------- 2d Interpolation Kernels ----------------------------*/
/* Kernels for NUptsdriven Method */
__global__
void Interp_2d_NUptsdriven(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
		       int nf1, int nf2, FLT es_c, FLT es_beta, int* idxnupts, int pirange)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		FLT x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		FLT y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);
        
		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);
		CUCPX cnow;
		cnow.x = 0.0;
		cnow.y = 0.0;
		FLT ker1[MAX_NSPREAD];
		FLT ker2[MAX_NSPREAD];

		FLT x1=(FLT)xstart-x_rescaled;
		FLT y1=(FLT)ystart-y_rescaled;
		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		eval_kernel_vec(ker2,y1,ns,es_c,es_beta);

		for(int yy=ystart; yy<=yend; yy++){
			FLT kervalue2 = ker2[yy-ystart];
			for(int xx=xstart; xx<=xend; xx++){
				int ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
				int iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				int inidx = ix+iy*nf1;
				FLT kervalue1 = ker1[xx-xstart];
				cnow.x += fw[inidx].x*kervalue1*kervalue2;
				cnow.y += fw[inidx].y*kervalue1*kervalue2;
			}
		}
		c[idxnupts[i]].x = cnow.x;
		c[idxnupts[i]].y = cnow.y;
	}

}

__global__
void Interp_2d_NUptsdriven_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT sigma, int* idxnupts, int pirange)
{
	for(int i=blockDim.x*blockIdx.x+threadIdx.x; i<M; i+=blockDim.x*gridDim.x){
		FLT x_rescaled=RESCALE(x[idxnupts[i]], nf1, pirange);
		FLT y_rescaled=RESCALE(y[idxnupts[i]], nf2, pirange);

		int xstart = ceil(x_rescaled - ns/2.0);
		int ystart = ceil(y_rescaled - ns/2.0);
		int xend = floor(x_rescaled + ns/2.0);
		int yend = floor(y_rescaled + ns/2.0);

		CUCPX cnow;
		cnow.x = 0.0;
		cnow.y = 0.0;
		FLT ker1[MAX_NSPREAD];
		FLT ker2[MAX_NSPREAD];

		eval_kernel_vec_Horner(ker1,xstart-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart-y_rescaled,ns,sigma);

		for(int yy=ystart; yy<=yend; yy++){
			FLT kervalue2 = ker2[yy-ystart];
			for(int xx=xstart; xx<=xend; xx++){
				int ix = xx < 0 ? xx+nf1 : (xx>nf1-1 ? xx-nf1 : xx);
				int iy = yy < 0 ? yy+nf2 : (yy>nf2-1 ? yy-nf2 : yy);
				int inidx = ix+iy*nf1;
				FLT kervalue1 = ker1[xx-xstart];
				cnow.x += fw[inidx].x*kervalue1*kervalue2;
				cnow.y += fw[inidx].y*kervalue1*kervalue2;
			}
		}
		c[idxnupts[i]].x = cnow.x;
		c[idxnupts[i]].y = cnow.y;
	}

}

/* Kernels for Subprob Method */
__global__
void Interp_2d_Subprob(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, const int ns,
	int nf1, int nf2, FLT es_c, FLT es_beta, FLT sigma, int* binstartpts,
	int* bin_size, int bin_size_x, int bin_size_y, int* subprob_to_bin,
	int* subprobstartpts, int* numsubprob, int maxsubprobsize, int nbinx, 
	int nbiny, int* idxnupts, int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix, iy;
	int outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=(bidx / nbinx)*bin_size_y;
	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0));

	for(int k=threadIdx.x;k<N; k+=blockDim.x){
		int i = k % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = k /( bin_size_x+2*ceil(ns/2.0) );
		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		if(ix < (nf1+ceil(ns/2.0)) && iy < (nf2+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			outidx = ix+iy*nf1;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
			fwshared[sharedidx].x = fw[outidx].x;
			fwshared[sharedidx].y = fw[outidx].y;
		}
	}
	__syncthreads();

	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];

	FLT x_rescaled, y_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);
		cnow.x = 0.0;
		cnow.y = 0.0;

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;

		FLT x1=(FLT)xstart+xoffset-x_rescaled;
		FLT y1=(FLT)ystart+yoffset-y_rescaled;

		eval_kernel_vec(ker1,x1,ns,es_c,es_beta);
		eval_kernel_vec(ker2,y1,ns,es_c,es_beta);
		for(int yy=ystart; yy<=yend; yy++){
			FLT kervalue2 = ker2[yy-ystart];
			for(int xx=xstart; xx<=xend; xx++){
				ix = xx+ceil(ns/2.0);
				iy = yy+ceil(ns/2.0);
				outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2);
				FLT kervalue1 = ker1[xx-xstart];
				cnow.x += fwshared[outidx].x*kervalue1*kervalue2;
				cnow.y += fwshared[outidx].y*kervalue1*kervalue2;
			}
		}
		c[idxnupts[idx]] = cnow;
	}
}

__global__
void Interp_2d_Subprob_Horner(FLT *x, FLT *y, CUCPX *c, CUCPX *fw, int M, 
	const int ns, int nf1, int nf2, FLT sigma, int* binstartpts, int* bin_size, 
	int bin_size_x, int bin_size_y, int* subprob_to_bin, int* subprobstartpts, 
	int* numsubprob, int maxsubprobsize, int nbinx, int nbiny, int* idxnupts, 
	int pirange)
{
	extern __shared__ CUCPX fwshared[];

	int xstart,ystart,xend,yend;
	int subpidx=blockIdx.x;
	int bidx=subprob_to_bin[subpidx];
	int binsubp_idx=subpidx-subprobstartpts[bidx];
	int ix, iy;
	int outidx;
	int ptstart=binstartpts[bidx]+binsubp_idx*maxsubprobsize;
	int nupts=min(maxsubprobsize, bin_size[bidx]-binsubp_idx*maxsubprobsize);

	int xoffset=(bidx % nbinx)*bin_size_x;
	int yoffset=(bidx / nbinx)*bin_size_y;

	int N = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*ceil(ns/2.0));


	for(int k=threadIdx.x;k<N; k+=blockDim.x){
		int i = k % (int) (bin_size_x+2*ceil(ns/2.0) );
		int j = k /( bin_size_x+2*ceil(ns/2.0) );
		ix = xoffset-ceil(ns/2.0)+i;
		iy = yoffset-ceil(ns/2.0)+j;
		if(ix < (nf1+ceil(ns/2.0)) && iy < (nf2+ceil(ns/2.0))){
			ix = ix < 0 ? ix+nf1 : (ix>nf1-1 ? ix-nf1 : ix);
			iy = iy < 0 ? iy+nf2 : (iy>nf2-1 ? iy-nf2 : iy);
			outidx = ix+iy*nf1;
			int sharedidx=i+j*(bin_size_x+ceil(ns/2.0)*2);
			fwshared[sharedidx].x = fw[outidx].x;
			fwshared[sharedidx].y = fw[outidx].y;
		}
	}
	__syncthreads();

	FLT ker1[MAX_NSPREAD];
	FLT ker2[MAX_NSPREAD];

	FLT x_rescaled, y_rescaled;
	CUCPX cnow;
	for(int i=threadIdx.x; i<nupts; i+=blockDim.x){
		int idx = ptstart+i;
		x_rescaled=RESCALE(x[idxnupts[idx]], nf1, pirange);
		y_rescaled=RESCALE(y[idxnupts[idx]], nf2, pirange);
		cnow.x = 0.0;
		cnow.y = 0.0;

		xstart = ceil(x_rescaled - ns/2.0)-xoffset;
		ystart = ceil(y_rescaled - ns/2.0)-yoffset;
		xend   = floor(x_rescaled + ns/2.0)-xoffset;
		yend   = floor(y_rescaled + ns/2.0)-yoffset;

		eval_kernel_vec_Horner(ker1,xstart+xoffset-x_rescaled,ns,sigma);
		eval_kernel_vec_Horner(ker2,ystart+yoffset-y_rescaled,ns,sigma);
		
		for(int yy=ystart; yy<=yend; yy++){
			FLT kervalue2 = ker2[yy-ystart];
			for(int xx=xstart; xx<=xend; xx++){
				ix = xx+ceil(ns/2.0);
				iy = yy+ceil(ns/2.0);
				outidx = ix+iy*(bin_size_x+ceil(ns/2.0)*2);
		
				FLT kervalue1 = ker1[xx-xstart];
				cnow.x += fwshared[outidx].x*kervalue1*kervalue2;
				cnow.y += fwshared[outidx].y*kervalue1*kervalue2;
			}
		}
		c[idxnupts[idx]] = cnow;
	}
}


}




/* C wrapper for calling CUDA kernels */
// Wrapper for testing spread, interpolation only
int CUFINUFFT_SPREAD1D(int nf1, CUCPX* d_fw, int M,
	FLT *d_kx, CUCPX* d_c, CUFINUFFT_PLAN d_plan);
int CUFINUFFT_INTERP1D(int nf1, CUCPX* d_fw, int M,
	FLT *d_kx, CUCPX* d_c, CUFINUFFT_PLAN d_plan);
int CUFINUFFT_SPREAD2D(int nf1, int nf2, CUCPX* d_fw, int M,
	FLT *d_kx, FLT *d_ky, CUCPX* d_c, CUFINUFFT_PLAN d_plan);
int CUFINUFFT_INTERP2D(int nf1, int nf2, CUCPX* d_fw, int M,
	FLT *d_kx, FLT *d_ky, CUCPX* d_c, CUFINUFFT_PLAN d_plan);
int CUFINUFFT_SPREAD3D(int nf1, int nf2, int nf3,
	CUCPX* d_fw, int M, FLT *d_kx, FLT *d_ky, FLT* d_kz,
	CUCPX* d_c, CUFINUFFT_PLAN dplan);
int CUFINUFFT_INTERP3D(int nf1, int nf2, int nf3,
	CUCPX* d_fw, int M, FLT *d_kx, FLT *d_ky, FLT *d_kz, 
    CUCPX* d_c, CUFINUFFT_PLAN dplan);

// Functions for calling different methods of spreading & interpolation
int CUSPREAD1D(CUFINUFFT_PLAN d_plan, int blksize);
int CUINTERP1D(CUFINUFFT_PLAN d_plan, int blksize);
int CUSPREAD2D(CUFINUFFT_PLAN d_plan, int blksize);
int CUINTERP2D(CUFINUFFT_PLAN d_plan, int blksize);
int CUSPREAD3D(CUFINUFFT_PLAN d_plan, int blksize, CUCPX* d_c, CUCPX* d_fw);
int CUINTERP3D(CUFINUFFT_PLAN d_plan, int blksize);

// Wrappers for methods of spreading
int CUSPREAD1D_NUPTSDRIVEN_PROP(int nf1, int M, CUFINUFFT_PLAN d_plan);
int CUSPREAD1D_NUPTSDRIVEN(int nf1, int M, CUFINUFFT_PLAN d_plan, int blksize);
int CUSPREAD1D_SUBPROB_PROP(int nf1, int M, CUFINUFFT_PLAN d_plan);
int CUSPREAD1D_SUBPROB(int nf1, int M, CUFINUFFT_PLAN d_plan, int blksize);

int CUSPREAD2D_NUPTSDRIVEN_PROP(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan);
int CUSPREAD2D_NUPTSDRIVEN(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
int CUSPREAD2D_SUBPROB_PROP(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan);
int CUSPREAD2D_PAUL_PROP(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan);
int CUSPREAD2D_SUBPROB(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
int CUSPREAD2D_PAUL(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan,
	int blksize);

int CUSPREAD3D_NUPTSDRIVEN_PROP(int nf1, int nf2, int nf3, int M,
	CUFINUFFT_PLAN d_plan);
int CUSPREAD3D_NUPTSDRIVEN(int nf1, int nf2, int nf3, CUCPX* d_c, int M, CUCPX* d_fw,
	CUFINUFFT_PLAN d_plan, int blksize);
int CUSPREAD3D_BLOCKGATHER_PROP(int nf1, int nf2, int nf3, int M,
	CUFINUFFT_PLAN d_plan);
int CUSPREAD3D_BLOCKGATHER(int nf1, int nf2, int nf3, CUCPX* d_c, int M, CUCPX* d_fw,
	CUFINUFFT_PLAN d_plan, int blksize);
int CUSPREAD3D_SUBPROB_PROP(int nf1, int nf2, int nf3, int M,
	CUFINUFFT_PLAN d_plan);
int CUSPREAD3D_SUBPROB(int nf1, int nf2, int nf3, CUCPX* d_c, int M, CUCPX* d_fw, CUFINUFFT_PLAN d_plan,
	int blksize);

// Wrappers for methods of interpolation
int CUINTERP1D_NUPTSDRIVEN(int nf1, int M, CUFINUFFT_PLAN d_plan, int blksize);
int CUINTERP2D_NUPTSDRIVEN(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
int CUINTERP2D_SUBPROB(int nf1, int nf2, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
int CUINTERP3D_NUPTSDRIVEN(int nf1, int nf2, int nf3, int M,
	CUFINUFFT_PLAN d_plan, int blksize);
int CUINTERP3D_SUBPROB(int nf1, int nf2, int nf3, int M, CUFINUFFT_PLAN d_plan,
	int blksize);
#endif
