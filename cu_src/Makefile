DEBUG ?= 0
ACCTIME ?= 1
PRINT_PROGRESS ?= 0
MPICC ?= mpicxx
MPIEXEC ?= mpiexec 

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-10.1

##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
    $(info WARNING - x86_64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=x86_64 instead)
    TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
    $(info WARNING - ARMv7 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=armv7l instead)
    TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
    $(info WARNING - aarch64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=aarch64 instead)
    TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
    $(info WARNING - ppc64le variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=ppc64le instead)
    TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
    $(info WARNING - GCC variable has been deprecated)
    $(info WARNING - please use HOST_COMPILER=$(GCC) instead)
    HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
    $(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le armv7l))
    ifneq ($(TARGET_ARCH),$(HOST_ARCH))
        ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
            TARGET_SIZE := 64
        else ifneq (,$(filter $(TARGET_ARCH),armv7l))
            TARGET_SIZE := 32
        endif
    else
        TARGET_SIZE := $(shell getconf LONG_BIT)
    endif
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# When on native aarch64 system with userspace of 32-bit, change TARGET_ARCH to armv7l
ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_SIZE),aarch64-aarch64-32)
    TARGET_ARCH = armv7l
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifeq ($(TARGET_OS),darwin)
    ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
        HOST_COMPILER ?= clang++
    endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        ifeq ($(TARGET_OS),linux)
            HOST_COMPILER ?= arm-linux-gnueabihf-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
        else ifeq ($(TARGET_OS),android)
            HOST_COMPILER ?= arm-linux-androideabi-g++
        endif
    else ifeq ($(TARGET_ARCH),aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/aarch64-unknown-nto-qnx7.0.0-g++
        else ifeq ($(TARGET_OS), android)
            HOST_COMPILER ?= aarch64-linux-android-clang++
        endif
    else ifeq ($(TARGET_ARCH),ppc64le)
        HOST_COMPILER ?= powerpc64le-linux-gnu-g++
    endif
endif


HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE} -dc
CCFLAGS     :=
LDFLAGS     :=

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib -L $(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib -L $(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/aarch64-linux-gnu -L $(TARGET_FS)/usr/lib/aarch64-linux-gnu
            LDFLAGS += --unresolved-symbols=ignore-in-shared-libs
            CCFLAGS += -isystem=$(TARGET_FS)/usr/include
            CCFLAGS += -isystem=$(TARGET_FS)/usr/include/aarch64-linux-gnu
        endif
    endif
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
        CCFLAGS += -DWIN_INTERFACE_CUSTOM -I/usr/include/aarch64-qnx-gnu
        LDFLAGS += -lsocket
        LDFLAGS += -rpath=/usr/lib/aarch64-qnx-gnu -L/usr/lib/aarch64-qnx-gnu
    endif
endif

# Install directory of different arch
CUDA_INSTALL_TARGET_DIR :=
ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-gnueabihf/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-android)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-android)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/ARMv7-linux-QNX/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-qnx/
else ifeq ($(TARGET_ARCH),ppc64le)
    CUDA_INSTALL_TARGET_DIR = targets/ppc64le-linux/
endif
# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS := -DDEBUG=$(DEBUG) -DPRINT_PROGRESS=$(PRINT_PROGRESS) -DACCTIME=$(ACCTIME)
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
#ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I../cu_inc
LIBRARIES :=

################################################################################

# Gencode arguments
SMS ?= 60 61 70 75

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

################################################################################

# Target rules
all: build

build: TestBFS TestScan DeltaFast DeltaFastCuda

DeltaFast.o:DeltaFast.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

TestBFS.o:TestBFS.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

TestScan.o:TestScan.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

DeltaFastCuda.o:DeltaFastCuda.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

DeltaFastMultiCuda.o:DeltaFastMultiCuda.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

bfs.o:bfs.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

#scan.o:scan.cu
#	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

graph.o:graph.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

kernels.o:kernels.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

deltafast.o:deltafast.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

utils.o:utils.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

TestBFS: TestBFS.o bfs.o graph.o kernels.o utils.o
	$(NVCC) $(ALL_LDFLAGS) -m64 $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) -lcudadevrt

TestScan: TestScan.o deltafast.o utils.o kernels.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

DeltaFast: DeltaFast.o graph.o utils.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

DeltaFastCuda: DeltaFastCuda.o graph.o deltafast.o kernels.o bfs.o utils.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

DeltaFastMultiCuda: DeltaFastMultiCuda.o graph.o deltafast.o kernels.o bfs.o utils.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

test_bfs: TestBFS
	./TestBFS ../dat/real/class.V 34 ../dat/real/class.E 78 33

test_bfs_large: TestBFS
	./TestBFS ../dat/real/oregon.V 10671 ../dat/real/oregon.E 44004 5001

test_bfs_itdk: TestBFS
	./TestBFS ../dat/real/itdk0304.V 192244 ../dat/real/itdk0304.E 636643 5001

test_scan: TestScan
	./TestScan

test_deltafast: DeltaFast
	./$< ../dat/real/class.V 34 ../dat/real/class.E 78 ../dat/real/class.A 6

test_deltafast_large: DeltaFast
	./$< ../dat/real/oregon.V 10671 ../dat/real/oregon.E 44004 ../dat/real/oregon.A 30

test_deltafastcuda: DeltaFastCuda
	./$< ../dat/real/class.V 34 ../dat/real/class.E 78 ../dat/real/class.A 6

test_deltafastcuda_large: DeltaFastCuda
	./$< ../dat/real/oregon.V 10671 ../dat/real/oregon.E 44004 ../dat/real/oregon.A 30

test_deltafastmulticuda: DeltaFastMultiCuda
	./$< ../dat/real/class.V 34 ../dat/real/class.E 78 3 ../dat/real/class.A 6

test_deltafastmulticuda_large: DeltaFastMultiCuda
	./$< ../dat/real/oregon.V 10671 ../dat/real/oregon.E 44004 3 ../dat/real/oregon.A 30

clean:
	rm -f TestBFS TestScan DeltaFast DeltaFastCuda DeltaFastMultiCuda
	rm -f *.o
