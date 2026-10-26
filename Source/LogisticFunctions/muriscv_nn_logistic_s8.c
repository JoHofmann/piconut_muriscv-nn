/*
 * Portions derived from TensorFlow/gemmlowp, Copyright 2015, 2019 The TensorFlow Authors.
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

#include "muriscv_nn_functions.h"
#include "muriscv_nn_tables.h"

#include <stdio.h>

#if defined(USE_VEXT)
#include <riscv_vector.h>
#endif

/** @ingroup groupNN
 * @addtogroup Logistic
 * @{ */
muriscv_nn_status muriscv_nn_logistic_s8(const int8_t *input,
                                         int8_t *output,
                                         const int32_t size,
                                         const int32_t input_zero_point,
                                         const int32_t input_range_radius,
                                         const int32_t input_multiplier,
                                         const int32_t left_shift)
{
    /* Parameters come from TFLM LogisticPrepare. Output scale is 1/256,
     * zero point -128. Scaling is converted once per call to LUT coordinates.
     * LUT interpolation is an approximation, not bit-exact to gemmlowp. */
    if (size < 0 || input_zero_point < -128 || input_zero_point > 127 || input_range_radius < 0 ||
        input_multiplier < 0 || left_shift < -31 || left_shift > 30 || (size > 0 && (input == 0 || output == 0)))
    {
        return MURISCV_NN_ARG_ERROR;
    }

    // printf("sigmoid s8: zero_point: %d, input_range_radius %d\n", input_zero_point, input_range_radius);

    // TFLM: real input = centered * input_multiplier * 2^(left_shift - 58).
    // LUT coordinate (14 fractional bits) = real input * 24 * 2^14
    //                                     = centered * input_multiplier * 3 * 2^(left_shift - 41).
    // Round the Q31 multiplier down to 15 fractional bits once per call.
    const uint32_t lut_multiplier = (((uint32_t)input_multiplier + (1u << 15)) >> 16) * 3u;
    const int32_t lut_shift = 25 - left_shift;
    const uint32_t rounding = (lut_shift > 0 && lut_shift < 32) ? 1u << (lut_shift - 1) : 0;
    // |centered| <= 255, multiplier <= 98304: product <= 25067520.
    // The largest left shift is 5, so even that result fits in uint32.

#if defined(USE_VEXT_)
    size_t avl = (size_t)size;
    while (avl > 0)
    {
        const size_t vl = __riscv_vsetvl_e32m4(avl);
        vint32m4_t centered = __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(input, vl), vl);
        centered = __riscv_vsub_vx_i32m4(centered, input_zero_point, vl);
        vuint32m4_t magnitude =
            __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax_vv_i32m4(centered, __riscv_vneg_v_i32m4(centered, vl), vl));
        vuint32m4_t coordinate = __riscv_vmul_vx_u32m4(magnitude, lut_multiplier, vl);
        if (lut_shift >= 32)
        {
            // RVV masks shift amounts to five bits; handle large shifts explicitly.
            coordinate = __riscv_vmv_v_x_u32m4(0, vl);
        }
        else if (lut_shift > 0)
        {
            coordinate = __riscv_vadd_vx_u32m4(coordinate, rounding, vl);
            coordinate = __riscv_vsrl_vx_u32m4(coordinate, lut_shift, vl);
        }
        else
        {
            coordinate = __riscv_vsll_vx_u32m4(coordinate, -lut_shift, vl);
        }

        vuint32m4_t index = __riscv_vsrl_vx_u32m4(coordinate, 14, vl);
        vbool8_t compute = __riscv_vmsltu_vx_u32m4_b8(index, 255, vl);
        vuint16m2_t offsets = __riscv_vnsrl_wx_u16m2(__riscv_vsll_vx_u32m4(index, 1, vl), 0, vl);
        // Mask both gathers so saturated lanes cannot read past the table.
        vuint16m2_t ua16 = __riscv_vloxei16_v_u16m2_m(compute, sigmoid_table_uint16, offsets, vl);
        vuint16m2_t ub16 = __riscv_vloxei16_v_u16m2_m(
            compute, sigmoid_table_uint16, __riscv_vadd_vx_u16m2_m(compute, offsets, 2, vl), vl);
        vuint32m4_t ua = __riscv_vzext_vf2_u32m4_m(compute, ua16, vl);
        vuint32m4_t ub = __riscv_vzext_vf2_u32m4_m(compute, ub16, vl);
        vuint32m4_t fraction = __riscv_vand_vx_u32m4_m(compute, coordinate, 0x3fff, vl);
        vuint32m4_t result = __riscv_vsub_vv_u32m4_m(compute, ub, ua, vl);
        result = __riscv_vmul_vv_u32m4_m(compute, result, fraction, vl);
        result = __riscv_vadd_vv_u32m4_m(compute, result, __riscv_vsll_vx_u32m4_m(compute, ua, 14, vl), vl);
        result = __riscv_vmerge_vxm_u32m4(result, 65535u << 14, __riscv_vmnot_m_b8(compute, vl), vl);

        vbool8_t negative = __riscv_vmslt_vx_i32m4_b8(centered, 0, vl);
        result = __riscv_vrsub_vx_u32m4_mu(negative, result, result, 1u << 30, vl);
        result = __riscv_vadd_vx_u32m4(result, 1u << 21, vl);
        result = __riscv_vsrl_vx_u32m4(result, 22, vl);
        vint32m4_t output_data = __riscv_vsub_vx_i32m4(__riscv_vreinterpret_v_u32m4_i32m4(result), 128, vl);
        output_data = __riscv_vmin_vx_i32m4(output_data, INT8_MAX, vl);
        output_data = __riscv_vmax_vx_i32m4(output_data, INT8_MIN, vl);

        // Apply the same cutoff precedence as the scalar path, including radius 0.
        vbool8_t high = __riscv_vmsge_vx_i32m4_b8(centered, input_range_radius, vl);
        vbool8_t low = __riscv_vmsle_vx_i32m4_b8(centered, -input_range_radius, vl);
        output_data = __riscv_vmerge_vxm_i32m4(output_data, INT8_MAX, high, vl);
        output_data = __riscv_vmerge_vxm_i32m4(output_data, INT8_MIN, low, vl);
        vint16m2_t output16 = __riscv_vnsra_wx_i16m2(output_data, 0, vl);
        __riscv_vse8_v_i8m1(output, __riscv_vnsra_wx_i8m1(output16, 0, vl), vl);
        input += vl;
        output += vl;
        avl -= vl;
    }
#else
    for (int32_t i = 0; i < size; ++i)
    {
        const int32_t centered = (int32_t)input[i] - input_zero_point;
        if (centered <= -input_range_radius)
        {
            output[i] = INT8_MIN;
        }
        else if (centered >= input_range_radius)
        {
            output[i] = INT8_MAX;
        }
        else
        {
            const uint32_t magnitude = centered < 0 ? -centered : centered;
            const uint32_t product = magnitude * lut_multiplier;
            uint32_t coordinate;
            if (lut_shift >= 32)
            {
                coordinate = 0;
            }
            else if (lut_shift > 0)
            {
                coordinate = (product + rounding) >> lut_shift;
            }
            else
            {
                coordinate = product << -lut_shift;
            }
            const uint32_t index = coordinate >> 14;
            uint32_t interpolated;
            if (index >= 255)
            {
                interpolated = 65535u << 14;
            }
            else
            {
                const uint32_t ua = sigmoid_table_uint16[index];
                const uint32_t ub = sigmoid_table_uint16[index + 1];
                const uint32_t fraction = coordinate & 0x3fffu;
                interpolated = (ua << 14) + fraction * (ub - ua);
            }
            if (centered < 0)
            {
                interpolated = (1u << 30) - interpolated;
            }
            // Q0.30 to int8: output scale 1/256, zero point -128.
            int32_t result = (int32_t)((interpolated + (1u << 21)) >> 22) - 128;
            if (result > INT8_MAX)
                result = INT8_MAX;
            if (result < INT8_MIN)
                result = INT8_MIN;
            output[i] = (int8_t)result;
        }
    }
#endif // USE_VEXT
    return MURISCV_NN_SUCCESS;
}
/** @} end of Logistic group */
