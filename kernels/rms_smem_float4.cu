#include <cuda_runtime.h>

#include <iostream>
#include <random>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const *func, char const *file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const *file, int line) {
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <int hiddenDim, int threadsPerBlock>
__global__ void rmsNormKernelSmemFloat4(float4 *x, float4 *w, float eps,
                                        float4 *y) {
  __shared__ float squaredPerThread[threadsPerBlock];
  __shared__ float4 xShared[hiddenDim >> 2];
  __shared__ float rms_;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  float sum = 0.0f;

  for (int i = tid; i < hiddenDim >> 2; i += threadsPerBlock) {
    int index = bid * (hiddenDim >> 2) + i;
    float4 x_ = x[index];
    xShared[i] = x_;
    sum += (x_.x * x_.x) + (x_.y * x_.y) + (x_.z * x_.z) + (x_.w * x_.w);
  }
  squaredPerThread[tid] = sum;
  __syncthreads();

  for (int activeThreads = threadsPerBlock >> 1; activeThreads > 0;
       activeThreads >>= 1) {
    if (tid < activeThreads) {
      squaredPerThread[tid] += squaredPerThread[tid + activeThreads];
    }
    __syncthreads();
  }

  if (tid == 0) {
    rms_ = rsqrtf(squaredPerThread[tid] / hiddenDim + eps);
  }
  __syncthreads();

  for (int i = tid; i < hiddenDim >> 2; i += threadsPerBlock) {
    float4 w_ = w[i];
    float4 x_ = xShared[i];
    float4 val = make_float4(x_.x * rms_ * w_.x, x_.y * rms_ * w_.y,
                             x_.z * rms_ * w_.z, x_.w * rms_ * w_.w);
    y[bid * (hiddenDim >> 2) + i] = val;
  }
}

template <int numTokens, int hiddenDim, int threadsPerBlock>
void launchRmsNormSmemFloat4(float *x, float *w, float eps, float *y) {
  float4 *x_ = reinterpret_cast<float4 *>(x);
  float4 *w_ = reinterpret_cast<float4 *>(w);
  float4 *y_ = reinterpret_cast<float4 *>(y);
  rmsNormKernelSmemFloat4<hiddenDim, threadsPerBlock>
      <<<numTokens, threadsPerBlock>>>(x_, w_, eps, y_);
}

template <int numTokens, int hiddenDim>
void launchRmsNormCpu(float *x, float *w, float eps, float *y) {
  float rms;
  for (int token = 0; token < numTokens; token++) {
    rms = 0;
    for (int hidden = 0; hidden < hiddenDim; hidden++) {
      rms += x[token * hiddenDim + hidden] * x[token * hiddenDim + hidden];
    }
    rms = sqrt(rms / hiddenDim + eps);
    for (int hidden = 0; hidden < hiddenDim; hidden++) {
      y[token * hiddenDim + hidden] =
          x[token * hiddenDim + hidden] / rms * w[hidden];
    }
  }
}

int main() {
  const int numTokens = 1 << 18;
  const int hiddenDim = 1 << 12;
  const size_t size = numTokens * hiddenDim * sizeof(float);
  const int threadsPerBlock = 1 << 9;

  float *xHost = new float[numTokens * hiddenDim];
  float *wHost = new float[hiddenDim];
  float eps = 1e-5f;
  float *yHost = new float[numTokens * hiddenDim];
  float *yReference = new float[numTokens * hiddenDim];

  std::default_random_engine generator(42);
  std::normal_distribution<float> distribution(0.0, 1.0);

  for (int i = 0; i < numTokens * hiddenDim; i++) {
    xHost[i] = distribution(generator);
    if (i < hiddenDim) {
      wHost[i] = 1.0f;
    }
  }

  float *xDevice;
  float *wDevice;
  float *yDevice;

  CHECK_CUDA_ERROR(cudaMalloc(&xDevice, size));
  CHECK_CUDA_ERROR(cudaMalloc(&wDevice, size / numTokens));
  CHECK_CUDA_ERROR(cudaMalloc(&yDevice, size));

  CHECK_CUDA_ERROR(cudaMemcpy(xDevice, xHost, size, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(wDevice, wHost, size / numTokens, cudaMemcpyHostToDevice));

  launchRmsNormSmemFloat4<numTokens, hiddenDim, threadsPerBlock>(
      xDevice, wDevice, eps, yDevice);

  CHECK_CUDA_ERROR(cudaMemcpy(yHost, yDevice, size, cudaMemcpyDeviceToHost));
  CHECK_LAST_CUDA_ERROR();

  launchRmsNormCpu<numTokens, hiddenDim>(xHost, wHost, eps, yReference);

  for (int token = 0; token < numTokens; token++) {
    for (int hidden = 0; hidden < hiddenDim; hidden++) {
      float y = yHost[token * hiddenDim + hidden];
      float yR = yReference[token * hiddenDim + hidden];

      if (fabs(y - yR) > 1e-3) {
        std::cout << "Error at token = " << token << " , hidden = " << hidden
                  << std::endl;
        std::cout << "y = " << y << " , yR = " << yR << std::endl;
        return 1;
      }
    }
  }
  std::cout << "Verification successfull" << std::endl;

  int numRounds = 10000;
  size_t numCrossMemoryBound = 2 * size;
  cudaEvent_t start, stop;
  float time;

  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

  CHECK_CUDA_ERROR(cudaEventRecord(start));
  for (int i = 0; i < numRounds; i++) {
    launchRmsNormSmemFloat4<numTokens, hiddenDim, threadsPerBlock>(
        xDevice, wDevice, eps, yDevice);
  }
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
  CHECK_LAST_CUDA_ERROR();

  float latency = time / numRounds;
  float bandwidth = (numCrossMemoryBound / latency) / 1e6;

  std::cout << "Latency = " << latency << " ms" << std::endl;
  std::cout << "Bandwidth = " << bandwidth << " GB/s" << std::endl;
  std::cout << "% of max = " << bandwidth / 3300 * 100 << " %" << std::endl;

  CHECK_CUDA_ERROR(cudaFree(xDevice));
  CHECK_CUDA_ERROR(cudaFree(wDevice));
  CHECK_CUDA_ERROR(cudaFree(yDevice));

  free(xHost);
  free(wHost);
  free(yHost);

  return 0;
}