/*
 * SPDX-FileCopyrightText: Copyright 2010-2020, 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Modifications copyright (C) 2023 Chair of Electronic Design Automation, TUM
 * Modifications copyright (C) 2026 Johannes Hofmann, THA <johannes.hofmann1@tha.de>
 */

/* ----------------------------------------------------------------------
 * Project:      MURISCV NN Library
 * Title:        muriscv_nn_activations_q15.c
 * Description:  Q15 neural network activation function using direct table look-up
 *
 * $Date:        19 February 2024
 * $Revision:    V.1.0.0
 *
 * Target Processor:  RISCV cores
 *
 * -------------------------------------------------------------------- */

#include "muriscv_nn_tables.h"
#include "muriscv_nn_functions.h"

#if defined(USE_VEXT)
#include <riscv_vector.h>
#endif

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Acti
 * @{
 */

/*
 * @brief Neural network activation function using direct table look-up
 *
 * @note  Refer header file for details.
 *
 */

muriscv_nn_status muriscv_nn_activation_s16(const int16_t *input,
                                            int16_t *output,
                                            const int32_t size,
                                            const int32_t left_shift,
                                            const muriscv_nn_activation_type type)
{
    uint32_t abs_input_shift, max_saturation;
    switch (type)
    {
    case MURISCV_SIGMOID:
        abs_input_shift = 9;
        max_saturation = 0x7FFF << 10;
        break;
    case MURISCV_TANH:
    default:
        abs_input_shift = 8;
        max_saturation = 0xFFFF << 8;
        break;
    }

    // int32_t input_multiplier = ((int32_t)3) << left_shift;
    const int32_t input_multiplier = (left_shift < 0) ? 3 : 3 << left_shift;
    const int32_t abs_left_shift = (left_shift < 0) ? -left_shift : 0;
    const int32_t rounding = (abs_left_shift > 0) ? 1 << (abs_left_shift - 1) : 0;
    // Use the LUT for sigmoid and take into account, that
    // tanh(x) = 2*sigmoid(2*x) - 1

#if defined(USE_VEXT)

    size_t vl, avl = size;

    while (avl > 0)
    {
        vl = __riscv_vsetvl_e32m4(avl);

        vint32m4_t input_data = __riscv_vsext_vf2_i32m4(__riscv_vle16_v_i16m2(input, vl), vl);
        input_data = __riscv_vmul_vx_i32m4(input_data, input_multiplier, vl);
        input_data = __riscv_vadd_vx_i32m4(input_data, rounding, vl);
        input_data = __riscv_vsra_vx_i32m4(input_data, abs_left_shift, vl);

        vuint32m4_t abs_input_data = __riscv_vreinterpret_v_i32m4_u32m4(
            __riscv_vmax_vv_i32m4(input_data, __riscv_vneg_v_i32m4(input_data, vl), vl));

        vuint32m4_t uh = __riscv_vsrl_vx_u32m4(abs_input_data, abs_input_shift, vl);
        vbool8_t compute = __riscv_vmsltu_vx_u32m4_b8(uh, 255, vl);

        vuint16m2_t offsets = __riscv_vnsrl_wx_u16m2(__riscv_vsll_vx_u32m4(uh, 1, vl), 0, vl);

        vuint16m2_t ua16 = __riscv_vloxei16_v_u16m2_m(compute, sigmoid_table_uint16, offsets, vl);
        vuint16m2_t ub16 = __riscv_vloxei16_v_u16m2_m(
            compute, sigmoid_table_uint16, __riscv_vadd_vx_u16m2_m(compute, offsets, 2, vl), vl);
        vuint32m4_t ua = __riscv_vzext_vf2_u32m4_m(compute, ua16, vl);
        vuint32m4_t ub = __riscv_vzext_vf2_u32m4_m(compute, ub16, vl);

        uint32_t fraction_mask = type == MURISCV_SIGMOID ? 0x1ff : 0x0ff;

        vuint32m4_t ut = __riscv_vand_vx_u32m4_m(compute, abs_input_data, fraction_mask, vl);

        vuint32m4_t result = __riscv_vsub_vv_u32m4_m(compute, ub, ua, vl);
        result = __riscv_vmul_vv_u32m4_m(compute, result, ut, vl);
        result =
            __riscv_vadd_vv_u32m4_m(compute, result, __riscv_vsll_vx_u32m4_m(compute, ua, abs_input_shift, vl), vl);
        result = __riscv_vmerge_vxm_u32m4(result, max_saturation, __riscv_vmnot_m_b8(compute, vl), vl);

        vbool8_t negative = __riscv_vmslt_vx_i32m4_b8(input_data, 0, vl);
        if (type == MURISCV_SIGMOID)
        {
            result = __riscv_vrsub_vx_u32m4_mu(negative, result, result, (1u << 25) - 1, vl);
            result = __riscv_vadd_vx_u32m4(result, 1u << 9, vl);
            result = __riscv_vsrl_vx_u32m4(result, 10, vl);
        }
        else
        {
            result = __riscv_vsub_vx_u32m4(result, 1u << 23, vl);
            result = __riscv_vrsub_vx_u32m4_mu(negative, result, result, UINT32_MAX, vl);
            result = __riscv_vadd_vx_u32m4(result, 1u << 7, vl);
            result = __riscv_vsrl_vx_u32m4(result, 8, vl);
        }

        __riscv_vse16_v_i16m2(output, __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vnsrl_wx_u16m2(result, 0, vl)), vl);

        input += vl;
        output += vl;
        avl -= vl;
    }

#else // USE_VEXT

    for (int i = 0; i < size; ++i, input++, output++)
    {
        const int32_t input_data = ((*input) * input_multiplier + rounding) >> abs_left_shift;

        uint32_t abs_input_data = input_data > 0 ? input_data : -input_data;

        uint32_t uh = abs_input_data >> abs_input_shift;

        uint32_t result;

        if (uh >= 255)
        {
            result = max_saturation;
        }
        else
        {
            uint32_t ua = sigmoid_table_uint16[uh];
            uint32_t ub = sigmoid_table_uint16[uh + 1];
            uint32_t ut;
            if (type == MURISCV_SIGMOID)
            {
                ut = abs_input_data & 0x1ff;
            }
            else
            {
                ut = abs_input_data & 0x0ff;
            }
            result = (ua << abs_input_shift) + ut * (ub - ua);
        }
        if (type == MURISCV_SIGMOID)
        {
            result = (input_data >= 0) ? (result + (1 << 9)) : ((1 << 25) - result + (1 << 9) - 1);
            result >>= 10;
        }
        else
        {
            result = (input_data >= 0) ? (result - (1 << 23)) + (1 << 7) : ((-result + (1 << 23)) + (1 << 7) - 1);
            result >>= 8;
        }
        *output = (int16_t)result;
    }

#endif // USE_VEXT

    return MURISCV_NN_SUCCESS;
}

/**
 * @} end of Acti group
 */
