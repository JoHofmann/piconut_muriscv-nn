#
# Copyright (C) 2021-2022 Chair of Electronic Design Automation, TUM.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Contains toolchain configurations and settings for using GCC

set(RISCV_GCC_BASENAME "riscv64-unknown-elf-" CACHE STRING "Base name of the toolchain executables.")

set(CMAKE_C_COMPILER ${RISCV_GCC_BASENAME}gcc)
set(CMAKE_CXX_COMPILER ${RISCV_GCC_BASENAME}g++)
set(CMAKE_ASM_COMPILER ${RISCV_GCC_BASENAME}gcc)
set(CMAKE_LINKER ${RISCV_GCC_BASENAME}ld)
set(CMAKE_OBJCOPY ${RISCV_GCC_BASENAME}objcopy)
set(CMAKE_OBJDUMP ${RISCV_GCC_BASENAME}objdump)
set(CMAKE_AR ${RISCV_GCC_BASENAME}ar)
set(CMAKE_RANLIB ${RISCV_GCC_BASENAME}ranlib)
set(CMAKE_STRIP ${RISCV_GCC_BASENAME}strip)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=${RISCV_ARCH} -mabi=${RISCV_ABI} -mcmodel=${RISCV_CMODEL}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${RISCV_ARCH} -mabi=${RISCV_ABI} -mcmodel=${RISCV_CMODEL}")
set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -march=${RISCV_ARCH} -mabi=${RISCV_ABI} -mcmodel=${RISCV_CMODEL}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -march=${RISCV_ARCH} -mabi=${RISCV_ABI} -mcmodel=${RISCV_CMODEL}")
