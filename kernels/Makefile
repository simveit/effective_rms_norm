NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
OUT_DIR = out
PROFILE_DIR = profile

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCC_FLAGS += -arch=sm_90a

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

rms_naive: rms_naive.cu
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

rms_naive_profile: rms_naive
	$(NCU_COMMAND) -o $(PROFILE_DIR)/$@ -f $(OUT_DIR)/$^

rms_smem: rms_smem.cu
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

rms_smem_profile: rms_smem
	$(NCU_COMMAND) -o $(PROFILE_DIR)/$@ -f $(OUT_DIR)/$^

rms_warp: rms_warp.cu
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

rms_warp_profile: rms_warp
	$(NCU_COMMAND) -o $(PROFILE_DIR)/$@ -f $(OUT_DIR)/$^

rms_smem_float4: rms_smem_float4.cu
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

rms_smem_float4_profile: rms_smem_float4
	$(NCU_COMMAND) -o $(PROFILE_DIR)/$@ -f $(OUT_DIR)/$^

rms_warp_float4: rms_warp_float4.cu
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

rms_warp_float4_profile: rms_smem_float4
	$(NCU_COMMAND) -o $(PROFILE_DIR)/$@ -f $(OUT_DIR)/$^


compile_all: 
	make rms_naive
	make rms_smem
	make rms_warp
	make rms_smem_float4
	make rms_warp_float4

run_all: 
	./$(OUT_DIR)/rms_naive
	./$(OUT_DIR)/rms_smem
	./$(OUT_DIR)/rms_warp
	./$(OUT_DIR)/rms_smem_float4
	./$(OUT_DIR)/rms_warp_float4

profile_all: 
	make rms_naive_profile
	make rms_smem_profile
	make rms_warp_profile
	make rms_smem_float4_profile
	make rms_warp_float4_profile
	
clean:
	rm $(OUT_DIR)/* $(PROFILE_DIR)/*

clean_out:
	rm $(OUT_DIR)/* 

clean_profile:
	rm $(PROFILE_DIR)/*
