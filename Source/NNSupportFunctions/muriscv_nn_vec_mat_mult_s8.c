
/*
 * Copyright (C) 2026 Johannes Hofmann, THA <johannes.hofmann1@tha.de>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if defined(USE_VEXT)
#include <riscv_vector.h>
#endif

#if defined(USE_PEXT)
#include <rvp_intrinsic.h>
#endif

#include "muriscv_nn_functions.h"
#include "muriscv_nn_support_functions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup NNBasicMath
 * @{
 */

/*
 * s8 vector(lhs) by matrix multiplication
 *
 * rhs is stored input-major: rhs[input_index * rhs_rows + output_index].
 * For compatibility with the transposed helper's argument convention,
 * rhs_cols is the input count and rhs_rows is the output count.
 * The physical matrix shape is therefore [rhs_cols][rhs_rows].
 * Bias has rhs_rows entries; dst uses address_offset between outputs.
 * kernel_sum is unused because offsets are applied directly.
 *
 */
muriscv_nn_status muriscv_nn_vec_mat_mult_s8(const q7_t *lhs,
                                               const q7_t *rhs,
                                               const int32_t *kernel_sum,
                                               const q31_t *bias,
                                               q7_t *dst,
                                               const int32_t lhs_offset,
                                               const int32_t dst_offset,
                                               const int32_t dst_multiplier,
                                               const int32_t dst_shift,
                                               const int32_t rhs_cols,
                                               const int32_t rhs_rows,
                                               const int32_t activation_min,
                                               const int32_t activation_max,
                                               const int32_t address_offset,
                                               const int32_t rhs_offset)
{
    (void)kernel_sum;

#if defined(USE_VEXT)
    if (address_offset == 1) {
    
    // Depending on the application this code could benefit from a smaller output
    // vector per batch and apply loop unrolling.

    // Each vector lane accumulates one output neuron. Process as many outputs
    // as fit in m8, then restart the input loop for the next output block.
    int32_t rows_remaining = rhs_rows;
    while (rows_remaining > 0)
    {
        const size_t vl = __riscv_vsetvl_e32m8(rows_remaining);
        vint32m8_t output = __riscv_vmv_v_x_i32m8(0, vl);
        const int32_t first_output = rhs_rows - rows_remaining;
    
        for (int i_loop_cnt = 0; i_loop_cnt < rhs_cols; i_loop_cnt++)
        {
            const int32_t lhs_value = (int32_t)lhs[i_loop_cnt] + lhs_offset;
            const q7_t *rhs_ptr = rhs + (size_t)i_loop_cnt * rhs_rows + first_output;
    
            vint32m8_t rhs_value = __riscv_vsext_vf4_i32m8(
                __riscv_vle8_v_i8m2(rhs_ptr, vl), vl);
            rhs_value = __riscv_vadd_vx_i32m8(rhs_value, rhs_offset, vl);

            output = __riscv_vmacc_vx_i32m8(output, lhs_value, rhs_value, vl);
    
        }
    
        // Bias is contiguous, with one int32 value per output row.
        if (bias)
        {
            vint32m8_t vbias = __riscv_vle32_v_i32m8(bias, vl);
            output = __riscv_vadd_vv_i32m8(output, vbias, vl);
            bias += vl;
        }
    
        // Apply positive shifts to the accumulator before the multiply.
        // The existing m8 helper handles multiplication and right shifts.
        if (dst_shift > 0)
        {
            output = __riscv_vsll_vx_i32m8(output, dst_shift, vl);
        }
        output = muriscv_nn_requantize_vec(output, dst_multiplier, MIN(dst_shift, 0), vl);
        output = __riscv_vadd_vx_i32m8(output, dst_offset, vl);
        output = __riscv_vmax_vx_i32m8(output, activation_min, vl);
        output = __riscv_vmin_vx_i32m8(output, activation_max, vl);
    
        // Narrow the completed outputs, preserving the number of lanes.
        vint16m4_t output_16 = __riscv_vncvt_x_x_w_i16m4(output, vl);
        vint8m2_t output_8 = __riscv_vncvt_x_x_w_i8m2(output_16, vl);
        __riscv_vse8_v_i8m2(dst, output_8, vl);
    
        dst += vl;
        rows_remaining -= vl;
    }
        return MURISCV_NN_SUCCESS;
    }
#endif

    // Scalar fallback for non-vector builds, strided outputs, and single rounding.
    // Every output walks one column of the input-major weight matrix.
    for (int32_t output_index = 0; output_index < rhs_rows; output_index++)
    {
        q31_t result = bias ? bias[output_index] : 0;

        for (int32_t input_index = 0; input_index < rhs_cols; input_index++)
        {
            const q31_t lhs_value = (q31_t)lhs[input_index] + lhs_offset;
            const q31_t rhs_value = (q31_t)rhs[(size_t)input_index * rhs_rows + output_index] + rhs_offset;
            result += lhs_value * rhs_value;
        }

        result = muriscv_nn_requantize(result, dst_multiplier, dst_shift);
        result += dst_offset;
        result = MAX(result, activation_min);
        result = MIN(result, activation_max);

        *dst = (q7_t)result;
        dst += address_offset;
    }

    return MURISCV_NN_SUCCESS;
}

/**
 * @} end of NNBasicMath group
 */
