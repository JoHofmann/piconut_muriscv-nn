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
muriscv_nn_status muriscv_nn_vec_mat_mult_per_ch_s8(const int8_t *lhs,
                                                    const int8_t *rhs,
                                                    const int32_t *kernel_sum,
                                                    const int32_t *bias,
                                                    int8_t *dst,
                                                    const int32_t lhs_offset,
                                                    const int32_t dst_offset,
                                                    const int32_t *dst_multiplier,
                                                    const int32_t *dst_shift,
                                                    const int32_t rhs_cols,
                                                    const int32_t rhs_rows,
                                                    const int32_t activation_min,
                                                    const int32_t activation_max,
                                                    const int32_t address_offset,
                                                    const int32_t rhs_offset)
{
    (void)kernel_sum;

#if defined(USE_VEXT)
    // Keep the scalar path for spaced outputs, as in vec_mat_mult_s8.
    // Single-rounding builds also use the scalar requantization below.
    if (address_offset == 1)
    {
        // Each m8 lane accumulates one output row. Restart the input loop
        // for each output block, including a partial final block.
        int32_t rows_remaining = rhs_rows;
        while (rows_remaining > 0)
        {
            const size_t vl = __riscv_vsetvl_e32m8(rows_remaining);
            vint32m8_t output = __riscv_vmv_v_x_i32m8(0, vl);
            const int32_t first_output = rhs_rows - rows_remaining;

            for (int32_t col = 0; col < rhs_cols; col++)
            {
                const int32_t lhs_value = (int32_t)lhs[col] + lhs_offset;
                const int8_t *rhs_ptr = rhs + (size_t)col * rhs_rows + first_output;
                vint32m8_t rhs_value = __riscv_vsext_vf4_i32m8(
                    __riscv_vle8_v_i8m2(rhs_ptr, vl), vl);
                rhs_value = __riscv_vadd_vx_i32m8(rhs_value, rhs_offset, vl);
                output = __riscv_vmacc_vx_i32m8(output, lhs_value, rhs_value, vl);
            }

            if (bias)
            {
                output = __riscv_vadd_vv_i32m8(output, __riscv_vle32_v_i32m8(bias, vl), vl);
                bias += vl;
            }

            output = muriscv_nn_requantize_vint32m8(output,
                                                    __riscv_vle32_v_i32m8(dst_multiplier, vl),
                                                    __riscv_vle32_v_i32m8(dst_shift, vl),
                                                    vl);

            output = __riscv_vadd_vx_i32m8(output, dst_offset, vl);
            output = __riscv_vmax_vx_i32m8(output, activation_min, vl);
            output = __riscv_vmin_vx_i32m8(output, activation_max, vl);

            const vint16m4_t output_16 = __riscv_vncvt_x_x_w_i16m4(output, vl);
            const vint8m2_t output_8 = __riscv_vncvt_x_x_w_i8m2(output_16, vl);
            __riscv_vse8_v_i8m2(dst, output_8, vl);

            dst += vl;
            dst_multiplier += vl;
            dst_shift += vl;
            rows_remaining -= vl;
        }

        return MURISCV_NN_SUCCESS;
    }
#endif

    // Scalar fallback for non-vector builds, spaced outputs and single rounding.
    for (int32_t row = 0; row < rhs_rows; row++)
    {
        int32_t output = bias ? bias[row] : 0;
        for (int32_t col = 0; col < rhs_cols; col++)
        {
            const int32_t lhs_value = (int32_t)lhs[col] + lhs_offset;
            const int32_t rhs_value = (int32_t)rhs[(size_t)col * rhs_rows + row] + rhs_offset;
            output += lhs_value * rhs_value;
        }

        output = muriscv_nn_requantize(output, dst_multiplier[row], dst_shift[row]);
        output += dst_offset;
        output = MAX(output, activation_min);
        output = MIN(output, activation_max);
        *dst = (int8_t)output;
        dst += address_offset;
    }

    return MURISCV_NN_SUCCESS;
}

/**
 * @} end of NNBasicMath group
 */
