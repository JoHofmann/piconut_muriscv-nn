/*************************************************************************

  This file is part of the PicoNut project.

  Copyright (C) 2026 Johannes Hofmann <johannes.hofmann1@tha.de>
      Technische Hochschule Augsburg, Technical University of Applied Sciences Augsburg

  Description:
    Profiler for RISC-V TFLM applications.

  --------------------- LICENSE -----------------------------------------------
  Redistribution and use in source and binary forms, with or without modification,
  are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
     list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation and/or
     other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*************************************************************************/

#include <cstdio>
#include <pn_riscv_defs.h>
#include "tensorflow/lite/micro/micro_profiler_interface.h"

class pn_profiler : public tflite::MicroProfilerInterface {
  public:
    uint32_t BeginEvent(const char *tag) override
    {
        if (num_events_ >= kMaxEvents)
        {
            num_events_ = 0; // wrap
        }
        tags_[num_events_] = tag;
        start_cycle_[num_events_] = read_cycle();
        start_instret_[num_events_] = read_instret();
        start_data_[num_events_] = read_data();
        return num_events_++;
    }

    void EndEvent(uint32_t event_handle) override
    {
        end_cycle_[event_handle] = read_cycle();
        end_instret_[event_handle] = read_instret();
        end_data_[event_handle] = read_data();
    }

    void report() const
    {
        printf("   %-20s %10s %10s %10s\n", "Tag", "Cycles", "Instructions", "Data");
        for (size_t i = 0; i < num_events_; ++i)
        {
            uint64_t delta_cycles = end_cycle_[i] - start_cycle_[i];
            uint64_t delta_instret = end_instret_[i] - start_instret_[i];
            uint64_t delta_data = end_data_[i] - start_data_[i];

            printf("%3d: %-20s %10llu %10llu %10llu\n", i, tags_[i], delta_cycles, delta_instret, delta_data);
        }
    }

  private:
    static constexpr size_t kMaxEvents = 256;
    const char *tags_[kMaxEvents];
    uint64_t start_cycle_[kMaxEvents];
    uint64_t end_cycle_[kMaxEvents];
    uint64_t start_instret_[kMaxEvents];
    uint64_t end_instret_[kMaxEvents];
    uint64_t start_data_[kMaxEvents];
    uint64_t end_data_[kMaxEvents];
    size_t num_events_ = 0;

    static uint64_t read_cycle()
    {
        uint32_t lo, hi1, hi2;
        do
        {
            hi1 = read_csr(cycleh);
            lo = read_csr(cycle);
            hi2 = read_csr(cycleh);
        } while (hi1 != hi2);
        return (((uint64_t)hi1) << 32) | lo;
    }

    static uint64_t read_instret()
    {
        uint32_t lo, hi1, hi2;
        do
        {
            hi1 = read_csr(instreth);
            lo = read_csr(instret);
            hi2 = read_csr(instreth);
        } while (hi1 != hi2);
        return (((uint64_t)hi1) << 32) | lo;
    }

    static uint64_t read_data()
    {
        // data accesses are stored in dscratch0 and dscratch1.
        uint32_t lo, hi1, hi2;
        do
        {
            hi1 = read_csr(dscratch1);
            lo = read_csr(dscratch0);
            hi2 = read_csr(dscratch1);
        } while (hi1 != hi2);
        return (((uint64_t)hi1) << 32) | lo;
    }
};
