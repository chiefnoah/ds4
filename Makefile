CC ?= cc
CFLAGS ?= -O3 -ffast-math -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -Wall -Wextra -fobjc-arc
DBGFLAGS ?=

PORTABLE_DEFS := -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE
CFLAGS += $(PORTABLE_DEFS) $(DBGFLAGS)

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 amd64))
CFLAGS += -march=native
OBJCFLAGS += -march=native
else
CFLAGS += -mcpu=native
OBJCFLAGS += -mcpu=native
endif

LDLIBS ?= -lm -pthread
UNAME_S := $(shell uname -s)
NATIVE_LDLIBS := $(LDLIBS)
METAL_SRCS := $(wildcard metal/*.metal)
DS4_ROCM ?= 0
ROCMCC ?= hipcc
HIP_PATH ?= $(shell hipconfig --path 2>/dev/null)
DS4_ROCM_ARCH ?= $(shell rocm-smi --showproductname 2>/dev/null | sed -n 's/.*GFX Version:[[:space:]]*//p' | head -n 1)
ROCM_ARCH_FLAGS := $(if $(DS4_ROCM_ARCH),--offload-arch=$(DS4_ROCM_ARCH))
# ROCm 7 hipcc no longer falls back to a system path for amdgcn bitcode; locate
# it via the loaded clang's resource dir or, as a fallback, the nearest device
# library shipped with rocm-device-libs.
ROCM_DEVICE_LIB_PATH ?= $(shell d=$$(dirname $$(dirname $$(realpath $$(command -v hipcc 2>/dev/null) 2>/dev/null) 2>/dev/null) 2>/dev/null)/amdgcn/bitcode; [ -d "$$d" ] && echo $$d)
ifeq ($(ROCM_DEVICE_LIB_PATH),)
ROCM_DEVICE_LIB_PATH := $(shell ls -d /nix/store/*-rocm-device-libs-*/amdgcn/bitcode 2>/dev/null | tail -n 1)
endif
ROCM_DEVICE_LIB_FLAGS := $(if $(ROCM_DEVICE_LIB_PATH),--rocm-device-lib-path=$(ROCM_DEVICE_LIB_PATH))
ROCMCFLAGS ?= -O3 -ffast-math -Wall -Wextra -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE -D__HIP_PLATFORM_AMD__ $(if $(HIP_PATH),-I$(HIP_PATH)/include) $(ROCM_ARCH_FLAGS) $(ROCM_DEVICE_LIB_FLAGS) $(DBGFLAGS)
ROCMLDLIBS ?= $(LDLIBS)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o
NATIVE_CORE_OBJS = ds4_native.o
else
ifeq ($(DS4_ROCM),1)
CFLAGS += -DDS4_USE_ROCM
CORE_OBJS = ds4.o ds4_rocm.o
NATIVE_CORE_OBJS = ds4_native.o
METAL_LDLIBS := $(ROCMLDLIBS)
else
CFLAGS += -DDS4_NO_METAL
CORE_OBJS = ds4.o
NATIVE_CORE_OBJS = ds4_native.o
METAL_LDLIBS := $(LDLIBS)
endif
endif

.PHONY: all clean test

all: ds4 ds4-server

ifeq ($(UNAME_S),Darwin)
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(NATIVE_LDLIBS)
else
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(if $(filter 1,$(DS4_ROCM)),$(ROCMCC),$(CC)) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

ds4-server: ds4_server.o rax.o $(CORE_OBJS)
	$(if $(filter 1,$(DS4_ROCM)),$(ROCMCC),$(CC)) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(NATIVE_LDLIBS)
endif

ds4.o: ds4.c ds4.h ds4_metal.h ds4_rocm.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_native.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4.c

ds4_cli_native.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4_cli.c

ds4_metal.o: ds4_metal.m ds4_metal.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_rocm.o: ds4_rocm.c ds4_rocm.h ds4_metal.h
	$(ROCMCC) $(ROCMCFLAGS) -x hip -c -o $@ ds4_rocm.c

ds4_test: ds4_test.o rax.o $(CORE_OBJS)
	$(if $(filter 1,$(DS4_ROCM)),$(ROCMCC),$(CC)) $(CFLAGS) -o $@ ds4_test.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4_rocm_kernel_test: tests/ds4_rocm_kernel_test.c ds4_rocm.o ds4_metal.h
	$(CC) $(CFLAGS) -c -o ds4_rocm_kernel_test.o tests/ds4_rocm_kernel_test.c
	$(ROCMCC) $(CFLAGS) -o $@ ds4_rocm_kernel_test.o ds4_rocm.o $(ROCMLDLIBS)

ds4_rocm_matmul_bench: tests/ds4_rocm_matmul_bench.c ds4_rocm.o ds4_metal.h
	$(CC) $(CFLAGS) -c -o ds4_rocm_matmul_bench.o tests/ds4_rocm_matmul_bench.c
	$(ROCMCC) $(CFLAGS) -o $@ ds4_rocm_matmul_bench.o ds4_rocm.o $(ROCMLDLIBS)

ds4_rocm_moe_bench: tests/ds4_rocm_moe_bench.c ds4_rocm.o ds4_metal.h
	$(CC) $(CFLAGS) -c -o ds4_rocm_moe_bench.o tests/ds4_rocm_moe_bench.c
	$(ROCMCC) $(CFLAGS) -o $@ ds4_rocm_moe_bench.o ds4_rocm.o $(ROCMLDLIBS)

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4_native ds4_server_test ds4_test ds4_rocm_kernel_test *.o
