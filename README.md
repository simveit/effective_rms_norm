# effective_rms_norm
effective_rms_norm

## Profile with NVIDA NSight

Please outcomment the benchmarking code in each kernel and run `make compile_all` and than `make profile_all` to create profiles for the kernels.

## Performance Comparison

| Kernel | Bandwidth (GB/s) | % of Max Bandwidth | Implementation |
|--------|------------------|-------------------|----------------|
| rms_naive | 3015.30 | 91.37% | Custom |
| rms_smem | 3044.99 | 92.27% | Custom |
| rms_warp | 3043.23 | 92.22% | Custom |
| rms_smem_float4 | 3062.86 | 92.81% | Custom |
| rms_warp_float4 | 3062.63 | 92.81% | Custom |