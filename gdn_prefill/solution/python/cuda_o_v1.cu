// cuda_o_v1.cu — CUDA O-kernel v3: 4 warps, XOR swizzle, v_new transpose
// SM100a: ldmatrix<4> required for A, XOR swizzle for bank-conflict-free

#include <cuda_bf16.h>
#include <cstdint>
#include "cuda_utils.h"

constexpr int BT=64, KD=128, VD=128, BV=64, TB=128;

__device__ __forceinline__
uint32_t pk2(float lo, float hi) {
    uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;":"=r"(r):"f"(hi),"f"(lo)); return r;
}
__device__ __forceinline__
int xor_idx(int row, int col, int stride) {
    return row*stride+(((col>>3)^(row&7))<<3)+(col&7);
}
__device__ __forceinline__
void xor_st16(nv_bfloat16 *base, int row, int grp, int stride, uint4 val) {
    *(uint4*)(base+row*stride+((grp^(row&7))<<3))=val;
}
__device__ __forceinline__
void ld_A_xor(uint32_t a[4], uint32_t sb, int stride, int mrow, int kc, int lane) {
    int lr=(lane%8)+((lane&8)?8:0), lc=(lane>=16)?8:0;
    int row=mrow+lr, col=kc+lc;
    ldmatrix<4>(a, sb+row*stride*2+(((col>>3)^(row&7))<<3)*2+(col&7)*2);
}
__device__ __forceinline__
void ld_B_xor(uint32_t b[2], const nv_bfloat16 *base, int stride, int n, int k, int gid, int thr) {
    int row=n+gid;
    b[0]=*(const uint32_t*)(base+xor_idx(row,k+thr*2,stride));
    b[1]=*(const uint32_t*)(base+xor_idx(row,k+thr*2+8,stride));
}

__global__ void __launch_bounds__(TB)
o_kernel_cuda(
    const nv_bfloat16 *__restrict__ q_ptr,
    const nv_bfloat16 *__restrict__ k_ptr,
    const nv_bfloat16 *__restrict__ v_new_ptr,
    const nv_bfloat16 *__restrict__ h_ptr,
    const float *__restrict__ g_cu_ptr,
    nv_bfloat16 *__restrict__ o_ptr,
    const int64_t *__restrict__ cu_seqlens,
    const int32_t *__restrict__ chunk_idx,
    int total_chunks, float scale, int H, int Hg)
{
    const int cid=blockIdx.x, hid=blockIdx.y;
    if(cid>=total_chunks) return;
    const int sid=chunk_idx[cid*2], cloc=chunk_idx[cid*2+1];
    const int bos=(int)cu_seqlens[sid], seqlen=(int)cu_seqlens[sid+1]-bos;
    const int tbase=cloc*BT;
    const int tid=threadIdx.x, warp=tid/32, lane=tid%32;
    const int gid=lane/4, thr=lane%4, mrow=warp*16;

    extern __shared__ char smem[];
    auto *s_q  =(nv_bfloat16*)smem;         // [BT, KD]
    auto *s_buf=s_q+BT*KD;                  // [max(BT,BV), KD] for k then h
    auto *s_A  =s_buf+BT*KD;                // [BT, BT]
    auto *s_vT =s_A+BT*BT;                  // [BV, BT] transposed v_new
    auto *s_g  =(float*)(s_vT+BV*BT);       // [BT]

    const uint32_t sq_a=__cvta_generic_to_shared(s_q);
    const uint32_t sa_a=__cvta_generic_to_shared(s_A);
    const int kh=hid/(H/Hg);
    const auto *qp=q_ptr+(int64_t)bos*Hg*KD+kh*KD;
    const auto *kp=k_ptr+(int64_t)bos*Hg*KD+kh*KD;
    const auto *gp=g_cu_ptr+(int64_t)bos*H+hid;
    const auto *hp=h_ptr+((int64_t)cid*H+hid)*VD*KD;
    const auto *vp=v_new_ptr+((int64_t)cid*BT*H+hid)*VD;
    auto *op=o_ptr+((int64_t)bos*H+hid)*VD;

    // ═══════ Phase 1: Load q, k, g_cu ═══════
    for(int i=tid;i<BT*KD/8;i+=TB){
        int r=i/(KD/8),g=i%(KD/8);int gt=tbase+r;
        uint4 v={0,0,0,0};
        if(gt<seqlen)v=*(const uint4*)(qp+(int64_t)gt*Hg*KD+g*8);
        xor_st16(s_q,r,g,KD,v);
    }
    for(int i=tid;i<BT*KD/8;i+=TB){
        int r=i/(KD/8),g=i%(KD/8);int gt=tbase+r;
        uint4 v={0,0,0,0};
        if(gt<seqlen)v=*(const uint4*)(kp+(int64_t)gt*Hg*KD+g*8);
        xor_st16(s_buf,r,g,KD,v);
    }
    for(int i=tid;i<BT;i+=TB)
        s_g[i]=(tbase+i<seqlen)?gp[(int64_t)(tbase+i)*H]:0.f;
    __syncthreads();

    // ═══════ Phase 2: A = causal(q @ k^T × exp_gate) ═══════
    float acc[8][4]={};
    for(int kt=0;kt<KD/16;kt++){
        int kc=kt*16;
        uint32_t qa[4]; ld_A_xor(qa,sq_a,KD,mrow,kc,lane);
        for(int nt=0;nt<8;nt++){
            uint32_t kb[2]; ld_B_xor(kb,s_buf,KD,nt*8,kc,gid,thr);
            mma_m16n8k16_bf16(acc[nt][0],acc[nt][1],acc[nt][2],acc[nt][3],
                qa[0],qa[1],qa[2],qa[3],kb[0],kb[1],
                acc[nt][0],acc[nt][1],acc[nt][2],acc[nt][3]);
        }
    }
    float eg0=__expf(s_g[mrow+gid]), eg1=__expf(s_g[mrow+gid+8]);
    for(int nt=0;nt<8;nt++){
        int c0=nt*8+thr*2,c1=c0+1;
        float ng0=__expf(-s_g[c0]),ng1=__expf(-s_g[c1]);
        int r0=mrow+gid,r1=r0+8;
        float a00=acc[nt][0]*eg0*ng0,a01=acc[nt][1]*eg0*ng1;
        if(c0>r0)a00=0;if(c1>r0)a01=0;
        *(uint32_t*)(s_A+xor_idx(r0,c0,BT))=pk2(a00,a01);
        float a10=acc[nt][2]*eg1*ng0,a11=acc[nt][3]*eg1*ng1;
        if(c0>r1)a10=0;if(c1>r1)a11=0;
        *(uint32_t*)(s_A+xor_idx(r1,c0,BT))=pk2(a10,a11);
    }
    __syncthreads();

    // ═══════ Phase 3: BV tile loop ═══════
    for(int bv=0;bv<VD/BV;bv++){
        int bv0=bv*BV;
        // Load h [BV,KD] → s_buf (XOR, vectorized)
        for(int i=tid;i<BV*KD/8;i+=TB){
            int r=i/(KD/8),g=i%(KD/8);
            xor_st16(s_buf,r,g,KD,*(const uint4*)(hp+(int64_t)(bv0+r)*KD+g*8));
        }
        // Load v_new [BT,BV] → s_vT (XOR, row-major — NOT transposed)
        for(int i=tid;i<BT*BV/8;i+=TB){
            int r=i/(BV/8),c8=i%(BV/8);int gt=tbase+r;
            uint4 v={0,0,0,0};
            if(gt<seqlen)v=*(const uint4*)(vp+(int64_t)r*H*VD+bv0+c8*8);
            xor_st16(s_vT,r,c8,BV,v);
        }
        __syncthreads();

        // q @ h^T
        float oa[8][4]={};
        for(int kt=0;kt<KD/16;kt++){
            int kc=kt*16;
            uint32_t qa[4]; ld_A_xor(qa,sq_a,KD,mrow,kc,lane);
            for(int nt=0;nt<8;nt++){
                uint32_t hb[2]; ld_B_xor(hb,s_buf,KD,nt*8,kc,gid,thr);
                mma_m16n8k16_bf16(oa[nt][0],oa[nt][1],oa[nt][2],oa[nt][3],
                    qa[0],qa[1],qa[2],qa[3],hb[0],hb[1],
                    oa[nt][0],oa[nt][1],oa[nt][2],oa[nt][3]);
            }
        }
        for(int nt=0;nt<8;nt++){
            oa[nt][0]*=eg0;oa[nt][1]*=eg0;oa[nt][2]*=eg1;oa[nt][3]*=eg1;
        }

        // A @ v_new (causal skip, v_new transposed → contiguous B loads)
        for(int kt=0;kt<=warp;kt++){
            int kc=kt*16;
            uint32_t aa[4]; ld_A_xor(aa,sa_a,BT,mrow,kc,lane);
            for(int nt=0;nt<8;nt++){
                // v_new row-major [BT,BV] → scattered B loads
                uint32_t vb[2];
                {auto ld=[&](int k,int n)->uint16_t{return *(const uint16_t*)(s_vT+xor_idx(k,n,BV));};
                int n=nt*8+gid;
                vb[0]=(uint32_t)ld(kc+thr*2,n)|((uint32_t)ld(kc+thr*2+1,n)<<16);
                vb[1]=(uint32_t)ld(kc+thr*2+8,n)|((uint32_t)ld(kc+thr*2+9,n)<<16);}
                mma_m16n8k16_bf16(oa[nt][0],oa[nt][1],oa[nt][2],oa[nt][3],
                    aa[0],aa[1],aa[2],aa[3],vb[0],vb[1],
                    oa[nt][0],oa[nt][1],oa[nt][2],oa[nt][3]);
            }
        }

        // Store
        for(int nt=0;nt<8;nt++){
            int col=bv0+nt*8+thr*2;
            int gt0=tbase+mrow+gid;
            if(gt0<seqlen)
                *(uint32_t*)(op+(int64_t)gt0*H*VD+col)=pk2(oa[nt][0]*scale,oa[nt][1]*scale);
            int gt1=gt0+8;
            if(gt1<seqlen)
                *(uint32_t*)(op+(int64_t)gt1*H*VD+col)=pk2(oa[nt][2]*scale,oa[nt][3]*scale);
        }
        __syncthreads();
    }
}

void o_v1(
    TensorView q, TensorView k, TensorView v_new, TensorView h,
    TensorView g_cu, TensorView o, TensorView cu_seqlens,
    TensorView chunk_indices, int total_num_chunks, double scale_d)
{
    int H=(int)o.size(1), Hg=(int)q.size(1);
    // smem: q[BT*KD] + buf[BT*KD] + A[BT*BT] + vT[BV*BT] + g[BT]
    constexpr int SMEM=BT*KD*2+BT*KD*2+BT*BT*2+BV*BT*2+BT*4;
    auto kern=o_kernel_cuda;
    cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
    dim3 grid(total_num_chunks, H);
    kern<<<grid,TB,SMEM>>>(
        (const nv_bfloat16*)q.data_ptr(),(const nv_bfloat16*)k.data_ptr(),
        (const nv_bfloat16*)v_new.data_ptr(),(const nv_bfloat16*)h.data_ptr(),
        (const float*)g_cu.data_ptr(),(nv_bfloat16*)o.data_ptr(),
        (const int64_t*)cu_seqlens.data_ptr(),(const int32_t*)chunk_indices.data_ptr(),
        total_num_chunks,(float)scale_d,H,Hg);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(o_v1, o_v1);
