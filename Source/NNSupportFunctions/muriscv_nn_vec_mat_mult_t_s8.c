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
 * Used by muriscv_nn_fully_connected_s8() and muriscv_nn_svdf_s8().  Kernel sum has been moved into here, functional
 * changes probably required
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
    if (address_offset == 1)
    { // TODO: remove this after #81 is fixed


        // Fill one m1 output register across groups of four dot products.
        // At VLEN=128 it holds four sums; larger VLENs collect multiple groups.
        const size_t output_capacity = __riscv_vsetvlmax_e32m1();
        // Reduce one initialized zero lane even when the dot product is empty.
        const size_t accum_vl = __riscv_vsetvl_e32m4(MAX(rhs_cols, 1));

        int32_t rows_remaining = rhs_rows;

        while (rows_remaining > 0)
        {
            const size_t output_vl = MIN((size_t)rows_remaining, output_capacity);
            vint32m1_t output = __riscv_vmv_v_x_i32m1(0, output_vl);
            size_t filled = 0;

            // Four independent rows share each loaded input vector.
            while (filled + 4 <= output_vl)
            {
                const q7_t *lhs_ptr = lhs;
                const q7_t *rhs_ptr_0 = rhs + rhs_cols * 0;
                vint32m4_t result_0 = __riscv_vmv_v_x_i32m4(0, accum_vl);
                const q7_t *rhs_ptr_1 = rhs + rhs_cols * 1;
                vint32m4_t result_1 = __riscv_vmv_v_x_i32m4(0, accum_vl);
                const q7_t *rhs_ptr_2 = rhs + rhs_cols * 2;
                vint32m4_t result_2 = __riscv_vmv_v_x_i32m4(0, accum_vl);
                const q7_t *rhs_ptr_3 = rhs + rhs_cols * 3;
                vint32m4_t result_3 = __riscv_vmv_v_x_i32m4(0, accum_vl);

                //uint32_t stage = pn_profiler_begin_event(&profiler, "macc");

                int32_t cols_remaining = rhs_cols;
                while (cols_remaining > 0)
                {

                    const size_t vl = __riscv_vsetvl_e32m4(cols_remaining);

                    vint32m4_t lhs_value = __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(lhs_ptr, vl), vl);
                    lhs_value = __riscv_vadd_vx_i32m4(lhs_value, lhs_offset, vl);

                    // The final chunk must preserve partial sums in inactive lanes.

                    vint32m4_t rhs_value_0 = __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(rhs_ptr_0, vl), vl);
                    rhs_value_0 = __riscv_vadd_vx_i32m4(rhs_value_0, rhs_offset, vl);
                    result_0 = __riscv_vmacc_vv_i32m4_tu(result_0, lhs_value, rhs_value_0, vl);
                    rhs_ptr_0 += vl;

                    vint32m4_t rhs_value_1 = __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(rhs_ptr_1, vl), vl);
                    rhs_value_1 = __riscv_vadd_vx_i32m4(rhs_value_1, rhs_offset, vl);
                    result_1 = __riscv_vmacc_vv_i32m4_tu(result_1, lhs_value, rhs_value_1, vl);
                    rhs_ptr_1 += vl;

                    vint32m4_t rhs_value_2 = __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(rhs_ptr_2, vl), vl);
                    rhs_value_2 = __riscv_vadd_vx_i32m4(rhs_value_2, rhs_offset, vl);
                    result_2 = __riscv_vmacc_vv_i32m4_tu(result_2, lhs_value, rhs_value_2, vl);
                    rhs_ptr_2 += vl;

                    vint32m4_t rhs_value_3 = __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(rhs_ptr_3, vl), vl);
                    rhs_value_3 = __riscv_vadd_vx_i32m4(rhs_value_3, rhs_offset, vl);
                    result_3 = __riscv_vmacc_vv_i32m4_tu(result_3, lhs_value, rhs_value_3, vl);
                    rhs_ptr_3 += vl;

                    lhs_ptr += vl;
                    cols_remaining -= vl;
                }

                //pn_profiler_end_event(&profiler, stage);
                //stage = pn_profiler_begin_event(&profiler, "reduce");

                const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);

                vint32m1_t reduced_0 = __riscv_vredsum_vs_i32m4_i32m1(result_0, zero, accum_vl);
                output = __riscv_vslideup_vx_i32m1(output, reduced_0, filled++, output_vl);

                vint32m1_t reduced_1 = __riscv_vredsum_vs_i32m4_i32m1(result_1, zero, accum_vl);
                output = __riscv_vslideup_vx_i32m1(output, reduced_1, filled++, output_vl);

                vint32m1_t reduced_2 = __riscv_vredsum_vs_i32m4_i32m1(result_2, zero, accum_vl);
                output = __riscv_vslideup_vx_i32m1(output, reduced_2, filled++, output_vl);

                vint32m1_t reduced_3 = __riscv_vredsum_vs_i32m4_i32m1(result_3, zero, accum_vl);
                output = __riscv_vslideup_vx_i32m1(output, reduced_3, filled++, output_vl);

                //pn_profiler_end_event(&profiler, stage);

                rhs += rhs_cols * 4;
            }

            // Append the last one to three rows to the same output register.
            while (filled < output_vl)
            {
                const q7_t *lhs_ptr = lhs;
                const q7_t *rhs_ptr_0 = rhs + rhs_cols * 0;
                vint32m4_t result_0 = __riscv_vmv_v_x_i32m4(0, accum_vl);

                //uint32_t stage = pn_profiler_begin_event(&profiler, "macc");

                int32_t cols_remaining = rhs_cols;
                while (cols_remaining > 0)
                {

                    const size_t vl = __riscv_vsetvl_e32m4(cols_remaining);

                    vint32m4_t lhs_value = __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(lhs_ptr, vl), vl);
                    lhs_value = __riscv_vadd_vx_i32m4(lhs_value, lhs_offset, vl);

                    // The final chunk must preserve partial sums in inactive lanes.

                    vint32m4_t rhs_value_0 = __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(rhs_ptr_0, vl), vl);
                    rhs_value_0 = __riscv_vadd_vx_i32m4(rhs_value_0, rhs_offset, vl);
                    result_0 = __riscv_vmacc_vv_i32m4_tu(result_0, lhs_value, rhs_value_0, vl);
                    rhs_ptr_0 += vl;
                    lhs_ptr += vl;
                    cols_remaining -= vl;
                }

                //pn_profiler_end_event(&profiler, stage);
                //stage = pn_profiler_begin_event(&profiler, "reduce");

                const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);

                vint32m1_t reduced_0 = __riscv_vredsum_vs_i32m4_i32m1(result_0, zero, accum_vl);
                output = __riscv_vslideup_vx_i32m1(output, reduced_0, filled++, output_vl);

                //pn_profiler_end_event(&profiler, stage);

                rhs += rhs_cols * 1;
            }

            // Only process outputs once the register is full or all rows are done.

            //uint32_t stage = pn_profiler_begin_event(&profiler, "requantize");
            if (bias)
            {
                output = __riscv_vadd_vv_i32m1(output, __riscv_vle32_v_i32m1(bias, output_vl), output_vl);
                bias += output_vl;
            }

            // The vector helper handles the right shift; apply positive shifts first.
            if (dst_shift > 0)
            {
                output = __riscv_vsll_vx_i32m1(output, dst_shift, output_vl);
            }

            vint32m1_t mult_vec = __riscv_vmv_v_x_i32m1(dst_multiplier, output_vl);
            vint32m1_t shift_vec = __riscv_vmv_v_x_i32m1(MIN(dst_shift, 0), output_vl);
            output = muriscv_nn_requantize_vint32m1(output, mult_vec, shift_vec, output_vl);
            output = __riscv_vadd_vx_i32m1(output, dst_offset, output_vl);
            output = __riscv_vmax_vx_i32m1(output, activation_min, output_vl);
            output = __riscv_vmin_vx_i32m1(output, activation_max, output_vl);

            vint16mf2_t output_16 = __riscv_vncvt_x_x_w_i16mf2(output, output_vl);
            vint8mf4_t output_8 = __riscv_vncvt_x_x_w_i8mf4(output_16, output_vl);
            __riscv_vse8_v_i8mf4(dst, output_8, output_vl);

            //pn_profiler_end_event(&profiler, stage);

            dst += output_vl;
            rows_remaining -= output_vl;
        }
    }
    else
    { // TODO: remove this after #81 is fixed
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

#elif defined(USE_PEXT) /* defined(USE_VEXT) */

    // const uint32_t lhs_offset_s8x4 = (__rv_packu(lhs_offset, lhs_offset)) | ((__rv_packu(lhs_offset, lhs_offset)) <<
    // 8);
    const uint32_t lhs_offset_s16x2 = (__rv_packu(lhs_offset, lhs_offset));
    // const uint32_t lhs_offset_s8x4 = (lhs_offset & 0x000000FF) | (lhs_offset << 8) & 0x0000FF00 | (lhs_offset << 16)
    // & 0x00FF0000 | (lhs_offset << 24) & 0xFF000000;
    const int32_t row_loop_cnt = rhs_rows >> 2;

    for (int32_t i = 0; i < row_loop_cnt; i++)
    {
        int32_t acc_0 = 0;
        int32_t acc_1 = 0;
        int32_t acc_2 = 0;
        int32_t acc_3 = 0;

        if (bias)
        {
            acc_0 = *bias++;
            acc_1 = *bias++;
            acc_2 = *bias++;
            acc_3 = *bias++;
        }

        const int32_t col_loop_cnt = rhs_cols >> 2;

        const int8_t *lhs_ptr = lhs;

        const int8_t *rhs_ptr_0 = rhs;
        const int8_t *rhs_ptr_1 = rhs + rhs_cols;
        const int8_t *rhs_ptr_2 = rhs_ptr_1 + rhs_cols;
        const int8_t *rhs_ptr_3 = rhs_ptr_2 + rhs_cols;

        rhs += 4 * rhs_cols;

        // TODO(fabianpedd): Why is the compiler soo dumb?!
        // This 'for (int j = 0; j < col_loop_cnt; j++)'
        // adds a couple of instructions compared to the zero
        // comparison currently used.

        for (int j = col_loop_cnt; j != 0; j--)
        {
            q31_t rhs_val_0, rhs_val_1, rhs_val_2, rhs_val_3;

            // int32_t lhs_packed = *(int32_t *)lhs_ptr;
            // lhs_packed = __rv_add8(lhs_offset_s8x4, lhs_packed);

            // q31_t lhs_val_0 = (*lhs_ptr++);
            // lhs_val_0 += lhs_offset;
            // q31_t lhs_val_1 = (*lhs_ptr++);
            // lhs_val_1 += lhs_offset;
            // q31_t lhs_val_2 = (*lhs_ptr++);
            // lhs_val_2 += lhs_offset;
            // q31_t lhs_val_3 = (*lhs_ptr++);
            // lhs_val_3 += lhs_offset;

            int32_t lhs_packed = *(int32_t *)lhs_ptr;
            int32_t lhs_32 = __rv_sunpkd832(lhs_packed);
            int32_t lhs_10 = __rv_sunpkd810(lhs_packed);
            lhs_32 = __rv_add16(lhs_32, lhs_offset_s16x2);
            lhs_10 = __rv_add16(lhs_10, lhs_offset_s16x2);

            // dont think i can use smaqa here, unsure how this worked before...  Possibly undefined behavior in
            // simulators Valid input range for LHS is signed with values < -128, cannot fit in 8 bit int.  TODO:
            // Rewrite this using the 16 bit SIMD instructions.
            /* Accumulate first rhs row */
            // int32_t rhs_packed = *(int32_t *)rhs_ptr_0;
            // acc_0 = __rv_smaqa_su(acc_0, rhs_packed, lhs_packed);

            // rhs_val_0 = *rhs_ptr_0++;
            // rhs_val_1 = *rhs_ptr_0++;
            // rhs_val_2 = *rhs_ptr_0++;
            // rhs_val_3 = *rhs_ptr_0++;
            // acc_0 = acc_0 + lhs_val_0*rhs_val_0 + lhs_val_1*rhs_val_1 + lhs_val_2*rhs_val_2 + lhs_val_3*rhs_val_3;

            int32_t rhs_packed = *(int32_t *)rhs_ptr_0;
            int32_t rhs_32 = __rv_sunpkd832(rhs_packed);
            int32_t rhs_10 = __rv_sunpkd810(rhs_packed);

            acc_0 = __rv_kmada(acc_0, rhs_32, lhs_32);
            acc_0 = __rv_kmada(acc_0, rhs_10, lhs_10);

            /* Accumulate second rhs row */
            // rhs_packed = *(int32_t *)rhs_ptr_1;
            // acc_1 = __rv_smaqa_su(acc_1, rhs_packed, lhs_packed);
            // rhs_val_0 = *rhs_ptr_1++;
            // rhs_val_1 = *rhs_ptr_1++;
            // rhs_val_2 = *rhs_ptr_1++;
            // rhs_val_3 = *rhs_ptr_1++;
            // acc_1 = acc_1+ lhs_val_0*rhs_val_0 + lhs_val_1*rhs_val_1 + lhs_val_2*rhs_val_2 + lhs_val_3*rhs_val_3;

            rhs_packed = *(int32_t *)rhs_ptr_1;
            rhs_32 = __rv_sunpkd832(rhs_packed);
            rhs_10 = __rv_sunpkd810(rhs_packed);

            acc_1 = __rv_kmada(acc_1, rhs_32, lhs_32);
            acc_1 = __rv_kmada(acc_1, rhs_10, lhs_10);

            /* Accumulate third rhs row */

            // rhs_packed = *(int32_t *)rhs_ptr_2;
            // acc_2 = __rv_smaqa_su(acc_2, rhs_packed, lhs_packed);
            // rhs_val_0 = *rhs_ptr_2++;
            // rhs_val_1 = *rhs_ptr_2++;
            // rhs_val_2 = *rhs_ptr_2++;
            // rhs_val_3 = *rhs_ptr_2++;
            // acc_2 = acc_2 + lhs_val_0*rhs_val_0 + lhs_val_1*rhs_val_1 + lhs_val_2*rhs_val_2 + lhs_val_3*rhs_val_3;

            rhs_packed = *(int32_t *)rhs_ptr_2;
            rhs_32 = __rv_sunpkd832(rhs_packed);
            rhs_10 = __rv_sunpkd810(rhs_packed);

            acc_2 = __rv_kmada(acc_2, rhs_32, lhs_32);
            acc_2 = __rv_kmada(acc_2, rhs_10, lhs_10);

            /* Accumulate fourth rhs row */
            // rhs_packed = *(int32_t *)rhs_ptr_3;
            // acc_3 = __rv_smaqa_su(acc_3, rhs_packed, lhs_packed);
            // rhs_val_0 = *rhs_ptr_3++;
            // rhs_val_1 = *rhs_ptr_3++;
            // rhs_val_2 = *rhs_ptr_3++;
            // rhs_val_3 = *rhs_ptr_3++;
            // acc_3 = acc_3 + lhs_val_0*rhs_val_0 + lhs_val_1*rhs_val_1 + lhs_val_2*rhs_val_2 + lhs_val_3*rhs_val_3;

            rhs_packed = *(int32_t *)rhs_ptr_3;
            rhs_32 = __rv_sunpkd832(rhs_packed);
            rhs_10 = __rv_sunpkd810(rhs_packed);

            acc_3 = __rv_kmada(acc_3, rhs_32, lhs_32);
            acc_3 = __rv_kmada(acc_3, rhs_10, lhs_10);

            rhs_ptr_0 += 4;
            rhs_ptr_1 += 4;
            rhs_ptr_2 += 4;
            rhs_ptr_3 += 4;

            lhs_ptr += 4;
        }

        for (int k = 0; k < rhs_cols % 4; k++)
        {
            q31_t lhs_value = *lhs_ptr + lhs_offset;

            q31_t rhs_value = *rhs_ptr_0;
            acc_0 += lhs_value * rhs_value;

            rhs_value = *rhs_ptr_1;
            acc_1 += lhs_value * rhs_value;

            rhs_value = *rhs_ptr_2;
            acc_2 += lhs_value * rhs_value;

            rhs_value = *rhs_ptr_3;
            acc_3 += lhs_value * rhs_value;

            ++rhs_ptr_0;
            ++rhs_ptr_1;
            ++rhs_ptr_2;
            ++rhs_ptr_3;

            ++lhs_ptr;
        }

        // Quantize down
        acc_0 = muriscv_nn_requantize(acc_0, dst_multiplier, dst_shift);
        acc_1 = muriscv_nn_requantize(acc_1, dst_multiplier, dst_shift);
        acc_2 = muriscv_nn_requantize(acc_2, dst_multiplier, dst_shift);
        acc_3 = muriscv_nn_requantize(acc_3, dst_multiplier, dst_shift);

        // Add offset
        acc_0 += dst_offset;
        acc_1 += dst_offset;
        acc_2 += dst_offset;
        acc_3 += dst_offset;

        // Clamp result
        acc_0 = MAX(acc_0, activation_min);
        acc_0 = MIN(acc_0, activation_max);
        acc_1 = MAX(acc_1, activation_min);
        acc_1 = MIN(acc_1, activation_max);
        acc_2 = MAX(acc_2, activation_min);
        acc_2 = MIN(acc_2, activation_max);
        acc_3 = MAX(acc_3, activation_min);
        acc_3 = MIN(acc_3, activation_max);

        // Store back
        *dst = (int8_t)acc_0;
        *(dst + address_offset) = (int8_t)acc_1;
        *(dst + 2 * address_offset) = (int8_t)acc_2;
        *(dst + 3 * address_offset) = (int8_t)acc_3;

        dst += 4 * address_offset;
    }
    for (int i = 0; i < rhs_rows % 4; i++)
    {
        int32_t acc_0 = 0;
        if (bias)
        {
            acc_0 = *bias++;
        }

        const int32_t col_loop_cnt = rhs_cols / 4;

        const int8_t *lhs_ptr = lhs;
        const int8_t *rhs_ptr_0 = rhs;

        // TODO(fabianpedd): Why is the compiler soo dumb?!
        // This 'for (int j = 0; j < col_loop_cnt; j++)'
        // adds a couple of instructions compared to the zero
        // comparison currently used.
        for (int j = col_loop_cnt; j != 0; j--)
        {
            q31_t rhs_val_0, rhs_val_1, rhs_val_2, rhs_val_3;

            // int32_t lhs_packed = *(int32_t *)lhs_ptr;
            // lhs_packed = __rv_add8(lhs_offset_s8x4, lhs_packed);

            q31_t lhs_val_0 = (*lhs_ptr++);
            lhs_val_0 += lhs_offset;
            q31_t lhs_val_1 = (*lhs_ptr++);
            lhs_val_1 += lhs_offset;
            q31_t lhs_val_2 = (*lhs_ptr++);
            lhs_val_2 += lhs_offset;
            q31_t lhs_val_3 = (*lhs_ptr++);
            lhs_val_3 += lhs_offset;

            // dont think i can use smaqa here, unsure how this worked before...  Possibly undefined behavior in
            // simulators
            /* Accumulate first rhs row */
            // int32_t rhs_packed = *(int32_t *)rhs_ptr_0;
            // acc_0 = __rv_smaqa_su(acc_0, rhs_packed, lhs_packed);

            rhs_val_0 = *rhs_ptr_0++;
            rhs_val_1 = *rhs_ptr_0++;
            rhs_val_2 = *rhs_ptr_0++;
            rhs_val_3 = *rhs_ptr_0++;
            acc_0 =
                acc_0 + lhs_val_0 * rhs_val_0 + lhs_val_1 * rhs_val_1 + lhs_val_2 * rhs_val_2 + lhs_val_3 * rhs_val_3;

            // rhs_ptr_0 += 4;
            // lhs_ptr += 4;
        }

        for (int k = 0; k < rhs_cols % 4; k++)
        {
            q31_t lhs_value = *lhs_ptr + lhs_offset;

            q31_t rhs_value = *rhs_ptr_0;
            acc_0 += lhs_value * rhs_value;

            ++rhs_ptr_0;
            ++lhs_ptr;
        }
        // Quantize down
        acc_0 = muriscv_nn_requantize(acc_0, dst_multiplier, dst_shift);

        // Add offset
        acc_0 += dst_offset;

        // Clamp result
        acc_0 = MAX(acc_0, activation_min);
        acc_0 = MIN(acc_0, activation_max);

        // Store back
        *dst = (int8_t)acc_0;

        // Bump pointer
        dst += address_offset;
        rhs += rhs_cols;
    }

#else /* defined(USE_PEXT) */

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
