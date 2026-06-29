#  -----------------------------------------------------------------------------
#
#  This file is part of the PicoNut project.
#
#  Copyright (C) 2026 Johannes Hofmann <johannes.hofmann1@tha.de>
#      Technische Hochschule Augsburg, Technical University of Applied Sciences Augsburg
#
#  Description:
#    Makefile for LiteRT Micro (TFLM) library with optimized RVV suppport (muRISCV-NN).
#
#  --------------------- LICENSE -----------------------------------------------
#  Redistribution and use in source and binary forms, with or without modification,
#  are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation and/or
#     other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
#  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  -----------------------------------------------------------------------------





################################################################################
#                                                                              #
#   Configuration                                                              #
#                                                                              #
################################################################################


# Environment ...
PN_SW_OS = bare
# PN_MARCH = rv32im_zve32x
# PN_MARCH = rv32im

# Muriscv-NN ...
USE_VEXT = OFF
VLEN = 1024
# ELEN = 64
ELEN = 32

USE_PORTABLE = OFF
USE_PEXT = OFF
TOOLCHAIN = GCC

ENABLE_UNIT_TESTS = OFF
SIMULATOR = Spike

# TFLM ...
TFLM_REF = master



# Include PicoNut setup ...
include ../../../piconut.mk





################################################################################
#                                                                              #
#   Prerequisite                                                               #
#                                                                              #
################################################################################


TFLM_DIR = $(PN_MODULE_SOURCE_DIR)/Integration/tflm
TFLM_FILE_NAMES := signal tensorflow tflite-micro third_party LICENSE
TFLM_FILES := $(addprefix $(TFLM_DIR)/, $(TFLM_FILE_NAMES))

.PHONY: download-tflm
download-tflm: $(TFLM_FILES)

$(TFLM_FILES) &:
	@cd $(TFLM_DIR) && env -i PATH="$(PATH)" ./download_tflm.sh

	@# Fix source tree creation: copy reference kernels that cmsis-nn implements but muriscv-nn doesnt't.
	@src="$(TFLM_DIR)/tflite-micro/tensorflow/lite/micro/kernels"; \
	dst="$(TFLM_DIR)/tensorflow/lite/micro/kernels"; \
	/bin/bash -c "cp $$src/{transpose.cc,pad.cc,maximum_minimum.cc,batch_matmul.cc} $$dst/"


.PHONY: clean-tflm
clean-tflm:
	rm -rf $(TFLM_FILES)





################################################################################
#                                                                              #
#   Building                                                                   #
#                                                                              #
################################################################################


# Define the library ...
MODULE := pn_tflm

LIB := $(PN_MODULE_BUILD_DIR)/Integration/tflm/lib$(MODULE).a

# TBD+: Install headers
#
# TFLM_PREFIX := muriscv-nn/Integration/tflm
# 
# HEADERS := \
# 	tensorflow/compiler/mlir/lite/core/api/error_reporter.h \
# 	tensorflow/lite/core/api/error_reporter.h \
# 	tensorflow/lite/micro/compatibility.h \
# 	tensorflow/lite/micro/micro_interpreter.h \
# 	tensorflow/lite/micro/micro_mutable_op_resolver.h \
# 	tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h \
# 	tensorflow/lite/schema/schema_generated.h
# 

	
$(PN_MODULE_BUILD_DIR)/Makefile: $(TFLM_FILES)
ifneq (0,$(VERBOSE))
	cmake \
		-DENABLE_INTG_TESTS=ON \
		-DTOOLCHAIN=$(TOOLCHAIN) \
		-DUSE_PORTABLE=$(USE_PORTABLE) \
		-DUSE_VEXT=$(USE_VEXT) \
		-DUSE_PEXT=$(USE_PEXT) \
		-DVLEN=$(VLEN) \
		-DELEN=$(ELEN) \
		\
		-DDISABLE_TVM_INTG_TESTS=ON \
		-DENABLE_UNIT_TESTS=$(ENABLE_UNIT_TESTS) \
		-DDISABLE_TFLM_AWW_INTG_TESTS=ON \
		-DDISABLE_TFLM_IC_INTG_TESTS=ON \
		-DDISABLE_TFLM_TOY_INTG_TESTS=ON \
		-DDISABLE_TFLM_VWW_INTG_TESTS=ON \
		\
		-DSIMULATOR=$(SIMULATOR) \
		-DPN_SOURCE_DIR=$(PN_SOURCE_DIR) \
		-DPN_CFG_CPU_RESET_ADR=$(PN_CFG_CPU_RESET_ADR) \
		-DPN_CFG_SYS_CODE_SIZE=$(PN_CFG_SYS_CODE_SIZE) \
		-DPN_CFG_SYS_RAM_SIZE=$(PN_CFG_SYS_RAM_SIZE) \
		-DPN_CFG_SYS_STACK_SIZE=$(PN_CFG_SYS_STACK_SIZE) \
		-DPN_CFG_SYS_HEAP_SIZE=$(PN_CFG_SYS_HEAP_SIZE) \
		\
		-B$(PN_MODULE_BUILD_DIR)
else
	@echo $(PN_BUILD_PREFIX)SW-CMAKE $(patsubst $(PN_MODULE_BUILD_DIR)/%, %, $(LIB)): VEXT=$(USE_VEXT), VLEN=$(VLEN), ELEN=$(ELEN); \
	cmake \
		-DENABLE_INTG_TESTS=ON \
		-DTOOLCHAIN=$(TOOLCHAIN) \
		-DUSE_PORTABLE=$(USE_PORTABLE) \
		-DUSE_VEXT=$(USE_VEXT) \
		-DUSE_PEXT=$(USE_PEXT) \
		-DVLEN=$(VLEN) \
		-DELEN=$(ELEN) \
		\
		-DDISABLE_TVM_INTG_TESTS=ON \
		-DENABLE_UNIT_TESTS=$(ENABLE_UNIT_TESTS) \
		-DDISABLE_TFLM_AWW_INTG_TESTS=ON \
		-DDISABLE_TFLM_IC_INTG_TESTS=ON \
		-DDISABLE_TFLM_TOY_INTG_TESTS=ON \
		-DDISABLE_TFLM_VWW_INTG_TESTS=ON \
		\
		-DSIMULATOR=$(SIMULATOR) \
		-DPN_SOURCE_DIR=$(PN_SOURCE_DIR) \
		-DPN_CFG_CPU_RESET_ADR=$(PN_CFG_CPU_RESET_ADR) \
		-DPN_CFG_SYS_CODE_SIZE=$(PN_CFG_SYS_CODE_SIZE) \
		-DPN_CFG_SYS_RAM_SIZE=$(PN_CFG_SYS_RAM_SIZE) \
		-DPN_CFG_SYS_STACK_SIZE=$(PN_CFG_SYS_STACK_SIZE) \
		-DPN_CFG_SYS_HEAP_SIZE=$(PN_CFG_SYS_HEAP_SIZE) \
		\
		-B$(PN_MODULE_BUILD_DIR) \
		--log-level=WARNING > /dev/null
endif

$(LIB): $(PN_MODULE_BUILD_DIR)/Makefile
ifneq (0,$(VERBOSE))
	$(MAKE) -C $(PN_MODULE_BUILD_DIR) all
else
	@echo $(PN_BUILD_PREFIX)SW-MAKE $(patsubst $(PN_MODULE_BUILD_DIR)/%, %, $(LIB)); \
	$(MAKE) -C $(PN_MODULE_BUILD_DIR) all > /dev/null
endif


################################################################################
#                                                                              #
#   Exported Targets                                                           #
#                                                                              #
################################################################################

# Exported targets ...
.PHONY: build-all
build-all: $(LIB)

.PHONY: install-all
install-all:
	$(PN_INSTALL_SW_LIB) $(LIB)
	# TBD+: Why is muriscv-nn not compiled into libtflm?
	$(PN_INSTALL_SW_LIB) $(PN_MODULE_BUILD_DIR)/Source/libmuriscvnn.a
	# Install headers
	# $(foreach header,$(HEADERS), \
	# 	$(PN_INSTALL_TREE) sw/include/$(header) $(TFLM_PREFIX)/$(header); \
	# )
	# $(PN_INSTALL_SW_INCLUDE) muriscv-nn/Integration/tflm/third_party/ruy
	# $(PN_INSTALL_SW_INCLUDE) muriscv-nn/Integration/tflm/third_party/gemmlowp
	# $(PN_INSTALL_SW_INCLUDE) muriscv-nn/Integration/tflm/third_party/kissfft
	# $(PN_INSTALL_SW_INCLUDE) muriscv-nn/Integration/tflm/third_party/flatbuffers
	# $(PN_INSTALL_SW_INCLUDE) muriscv-nn/Integration/tflm/third_party/flatbuffers/include
	# $(PN_INSTALL_SW_INCLUDE) muriscv-nn/Include
	# $(PN_INSTALL_SW_INCLUDE) muriscv-nn/Include/CMSIS/NN
