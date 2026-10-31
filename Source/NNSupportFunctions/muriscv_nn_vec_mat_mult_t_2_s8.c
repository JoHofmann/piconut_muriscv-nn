/*
 * Copyright (C) 2010-2022 Arm Limited or its affiliates.
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
 *
 * Modifications copyright (C) 2021-2024 Chair of Electronic Design Automation, TUM
 */

#if defined(USE_VEXT)
#include <riscv_vector.h>
#endif

#if defined(USE_PEXT)
#include <rvp_intrinsic.h>
#endif

#include "muriscv_nn_functions.h"
#include "muriscv_nn_support_functions.h"

#include "pn_profiler.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup NNBasicMath
 * @{
 */

/*
 * s8 vector(lhs) by matrix (transposed) multiplication
 *
 * Refer header file for details.
 *
 * Used by muriscv_nn_fully_connected_s8() and muriscv_nn_svdf_s8().  Kernel sum has been moved into here, functional changes probably required
 *
 */
muriscv_nn_status muriscv_nn_vec_mat_mult_t_s8(const q7_t *lhs,
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
                                               const int32_t rhs_offset) // Not yet supported by PEXT
{
#if defined(USE_VEXT)
    if (address_offset == 1) { // TODO: remove this after #81 is fixed
    
    // Depending on the application this code could benefit from a smaller output
    // vector per batch and apply loop unrolling.

    // Each vector lane accumulates one output row. Process as many rows
    // as fit in m8, then restart the input loop for the next output block.
    int32_t rows_remaining = rhs_rows;
    while (rows_remaining > 0)
    {
        const size_t vl = __riscv_vsetvl_e32m8(rows_remaining);
        vint32m8_t output = __riscv_vmv_v_x_i32m8(0, vl);
        const q7_t *rhs_ptr = rhs;
    
        for (int i_loop_cnt = 0; i_loop_cnt < rhs_cols; i_loop_cnt++)
        {
            const int32_t lhs_value = (int32_t)lhs[i_loop_cnt] + lhs_offset;
    
            vint32m8_t rhs_value = __riscv_vsext_vf4_i32m8(
                __riscv_vlse8_v_i8m2(rhs_ptr, (ptrdiff_t)rhs_cols, vl), vl);
            rhs_value = __riscv_vadd_vx_i32m8(rhs_value, rhs_offset, vl);

            output = __riscv_vmacc_vx_i32m8(output, lhs_value, rhs_value, vl);
    
            rhs_ptr++;
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
        rhs += (size_t)rhs_cols * vl;
        rows_remaining -= vl;
    }
    } else { // TODO: remove this after #81 is fixed
    /* Uses 5x loop unrolling in order to expose more ILP */
    const int32_t row_loop_cnt = rhs_rows / 5;
    for (int i_row_loop_cnt = 0; i_row_loop_cnt < row_loop_cnt; i_row_loop_cnt++)
    {

        const q7_t *lhs_ptr = lhs;
        const q7_t *rhs_ptr_0 = &rhs[0];
        const q7_t *rhs_ptr_1 = &rhs[rhs_cols];
        const q7_t *rhs_ptr_2 = &rhs[rhs_cols * 2];
        const q7_t *rhs_ptr_3 = &rhs[rhs_cols * 3];
        const q7_t *rhs_ptr_4 = &rhs[rhs_cols * 4];

        q31_t res00 = 0;
        q31_t res01 = 0;
        q31_t res02 = 0;
        q31_t res03 = 0;
        q31_t res04 = 0;
        if (bias)
        {
            res00 = *bias++;
            res01 = *bias++;
            res02 = *bias++;
            res03 = *bias++;
            res04 = *bias++;
        }
        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            const q31_t rhs_value0 = (int8_t)*rhs_ptr_0 + rhs_offset;
            const q31_t rhs_value1 = (int8_t)*rhs_ptr_1 + rhs_offset;
            const q31_t rhs_value2 = (int8_t)*rhs_ptr_2 + rhs_offset;
            const q31_t rhs_value3 = (int8_t)*rhs_ptr_3 + rhs_offset;
            const q31_t rhs_value4 = (int8_t)*rhs_ptr_4 + rhs_offset;
            const q31_t lhs_value = (int8_t)*lhs_ptr + lhs_offset;

            res00 += lhs_value * rhs_value0;
            res01 += lhs_value * rhs_value1;
            res02 += lhs_value * rhs_value2;
            res03 += lhs_value * rhs_value3;
            res04 += lhs_value * rhs_value4;

            ++rhs_ptr_0;
            ++rhs_ptr_1;
            ++rhs_ptr_2;
            ++rhs_ptr_3;
            ++rhs_ptr_4;
            ++lhs_ptr;
        }
        // Quantize down
        res00 = muriscv_nn_requantize(res00, dst_multiplier, dst_shift);
        res01 = muriscv_nn_requantize(res01, dst_multiplier, dst_shift);
        res02 = muriscv_nn_requantize(res02, dst_multiplier, dst_shift);
        res03 = muriscv_nn_requantize(res03, dst_multiplier, dst_shift);
        res04 = muriscv_nn_requantize(res04, dst_multiplier, dst_shift);

        // Add offset
        res00 += dst_offset;
        res01 += dst_offset;
        res02 += dst_offset;
        res03 += dst_offset;
        res04 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);
        res01 = MAX(res01, activation_min);
        res01 = MIN(res01, activation_max);
        res02 = MAX(res02, activation_min);
        res02 = MIN(res02, activation_max);
        res03 = MAX(res03, activation_min);
        res03 = MIN(res03, activation_max);
        res04 = MAX(res04, activation_min);
        res04 = MIN(res04, activation_max);

        *dst = (q7_t)res00;
        *(dst + address_offset) = (q7_t)res01;
        *(dst + 2 * address_offset) = (q7_t)res02;
        *(dst + 3 * address_offset) = (q7_t)res03;
        *(dst + 4 * address_offset) = (q7_t)res04;
        dst += 5 * address_offset;

        rhs += 5 * rhs_cols;

    }

    /* Handle the leftover part from 5x loop unrolling */
    const int loop_cnt = rhs_rows % 5;
    for (int i_loop_cnt = 0; i_loop_cnt < loop_cnt; i_loop_cnt++)
    {

        const q7_t *lhs_ptr = &lhs[0];
        const q7_t *rhs_ptr = &rhs[0];

        q31_t res00 = 0;
        if (bias)
        {
            res00 = *bias++;
        }

        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            q31_t rhs_value0 = (int8_t)rhs_ptr[0] + rhs_offset;
            q31_t lhs_value = (int8_t)lhs_ptr[0] + lhs_offset;

            res00 += lhs_value * rhs_value0;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = muriscv_nn_requantize(res00, dst_multiplier, dst_shift);

        // Add offset
        res00 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);

        *dst = (int8_t)res00;
        dst += address_offset;
        rhs += rhs_cols;
        
    }
    } // TODO: remove this after #81 is fixed

#else /* defined(USE_VEXT) */

    /* Uses 5x loop unrolling in order to expose more ILP */
    const int32_t row_loop_cnt = rhs_rows / 5;
    for (int i_row_loop_cnt = 0; i_row_loop_cnt < row_loop_cnt; i_row_loop_cnt++)
    {
        
        const q7_t *lhs_ptr = lhs;
        const q7_t *rhs_ptr_0 = &rhs[0];
        const q7_t *rhs_ptr_1 = &rhs[rhs_cols];
        const q7_t *rhs_ptr_2 = &rhs[rhs_cols * 2];
        const q7_t *rhs_ptr_3 = &rhs[rhs_cols * 3];
        const q7_t *rhs_ptr_4 = &rhs[rhs_cols * 4];
        
        q31_t res00 = 0;
        q31_t res01 = 0;
        q31_t res02 = 0;
        q31_t res03 = 0;
        q31_t res04 = 0;
        if (bias)
        {
            res00 = *bias++;
            res01 = *bias++;
            res02 = *bias++;
            res03 = *bias++;
            res04 = *bias++;
        }
        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            const q31_t rhs_value0 = (int8_t)*rhs_ptr_0 + rhs_offset;
            const q31_t rhs_value1 = (int8_t)*rhs_ptr_1 + rhs_offset;
            const q31_t rhs_value2 = (int8_t)*rhs_ptr_2 + rhs_offset;
            const q31_t rhs_value3 = (int8_t)*rhs_ptr_3 + rhs_offset;
            const q31_t rhs_value4 = (int8_t)*rhs_ptr_4 + rhs_offset;
            const q31_t lhs_value = (int8_t)*lhs_ptr + lhs_offset;

            res00 += lhs_value * rhs_value0;
            res01 += lhs_value * rhs_value1;
            res02 += lhs_value * rhs_value2;
            res03 += lhs_value * rhs_value3;
            res04 += lhs_value * rhs_value4;

            ++rhs_ptr_0;
            ++rhs_ptr_1;
            ++rhs_ptr_2;
            ++rhs_ptr_3;
            ++rhs_ptr_4;
            ++lhs_ptr;
        }
        // Quantize down
        res00 = muriscv_nn_requantize(res00, dst_multiplier, dst_shift);
        res01 = muriscv_nn_requantize(res01, dst_multiplier, dst_shift);
        res02 = muriscv_nn_requantize(res02, dst_multiplier, dst_shift);
        res03 = muriscv_nn_requantize(res03, dst_multiplier, dst_shift);
        res04 = muriscv_nn_requantize(res04, dst_multiplier, dst_shift);

        // Add offset
        res00 += dst_offset;
        res01 += dst_offset;
        res02 += dst_offset;
        res03 += dst_offset;
        res04 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);
        res01 = MAX(res01, activation_min);
        res01 = MIN(res01, activation_max);
        res02 = MAX(res02, activation_min);
        res02 = MIN(res02, activation_max);
        res03 = MAX(res03, activation_min);
        res03 = MIN(res03, activation_max);
        res04 = MAX(res04, activation_min);
        res04 = MIN(res04, activation_max);

        *dst = (q7_t)res00;
        *(dst + address_offset) = (q7_t)res01;
        *(dst + 2 * address_offset) = (q7_t)res02;
        *(dst + 3 * address_offset) = (q7_t)res03;
        *(dst + 4 * address_offset) = (q7_t)res04;
        dst += 5 * address_offset;

        rhs += 5 * rhs_cols;
        
    }

    /* Handle the leftover part from 5x loop unrolling */
    const int loop_cnt = rhs_rows % 5;
    for (int i_loop_cnt = 0; i_loop_cnt < loop_cnt; i_loop_cnt++)
    {
        
        const q7_t *lhs_ptr = &lhs[0];
        const q7_t *rhs_ptr = &rhs[0];

        q31_t res00 = 0;
        if (bias)
        {
            res00 = *bias++;
        }

        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            q31_t rhs_value0 = (int8_t)rhs_ptr[0] + rhs_offset;
            q31_t lhs_value = (int8_t)lhs_ptr[0] + lhs_offset;

            res00 += lhs_value * rhs_value0;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = muriscv_nn_requantize(res00, dst_multiplier, dst_shift);

        // Add offset
        res00 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);

        *dst = (int8_t)res00;
        dst += address_offset;
        rhs += rhs_cols;
        
    }

#endif /* defined(USE_PEXT) */

    return MURISCV_NN_SUCCESS;
}

/**
 * @} end of NNBasicMath group
 */
