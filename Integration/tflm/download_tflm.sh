#!/bin/bash
#
# Copyright (C) 2021-2023 Chair of Electronic Design Automation, TUM.
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

# Prevent silent failures
set -euo pipefail

VERBOSE=${VERBOSE:-0}

# Path to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TFLM_REF=${TFLM_REF:-""}

log() {
  echo "[sw/libs/piconut_muriscvnn] $@"
}

# Configure quiet flags for external tools
Q_FLAG=""
if [ "$VERBOSE" -eq 0 ]; then
  Q_FLAG="-q"
fi

# Create and activate a virtual environment
VENV_DIR="${SCRIPT_DIR}/venv"
if [ ! -d "$VENV_DIR" ]; then
  log "Creating Python virtual environment at ${VENV_DIR}"
  python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment and install dependencies
source "$VENV_DIR/bin/activate"
pip install $Q_FLAG --upgrade pip
pip install $Q_FLAG numpy Pillow


log "Download TFLM sources"
if [ ! -d "tflite-micro" ]; then
  git clone $Q_FLAG https://github.com/tensorflow/tflite-micro.git
  cd tflite-micro
  if [[ "$TFLM_REF" != "" ]]
  then
    git checkout $Q_FLAG $TFLM_REF
  fi
else
  cd tflite-micro
  git pull $Q_FLAG
fi

log "Generate TFLM source tree"
if [ "$VERBOSE" -eq 0 ]; then
  python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py .. --makefile_options="OPTIMIZED_KERNEL_DIR=cmsis_nn" > /dev/null
else
  python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py .. --makefile_options="OPTIMIZED_KERNEL_DIR=cmsis_nn"
fi

# Use this command to build the source tree without the CMSIS-NN wrapper (also disable USE_CMSIS_NN_WRAPPER in CMakeLists.txt!)
# python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py ..
