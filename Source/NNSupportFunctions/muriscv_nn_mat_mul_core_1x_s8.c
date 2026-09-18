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
 * Modifications copyright (C) 2021-2022 Chair of Electronic Design Automation, TUM
 */

#if defined(USE_VEXT)
#include <riscv_vector.h>
#elif defined(USE_PEXT)
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
 * s8 matrix multiplication to process 1 row
 *
 * Refer header file for details.
 *
 */

muriscv_nn_status muriscv_nn_mat_mul_core_1x_s8(int32_t row_elements,
                                                const int8_t *row_base,
                                                const int8_t *col_base,
                                                int32_t *const sum_col,
                                                int32_t *const output)
{
    int32_t acc_n0 = 0;
    int32_t sum_tmp = 0;
#if defined(USE_VEXT)
    const int8_t *row_ptr = row_base;
    const int8_t *col_ptr = col_base;
    int32_t loop_cnt = row_elements;

    size_t vl = __riscv_vsetvl_e32m4(loop_cnt);
    vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
    vint32m4_t sum = __riscv_vmv_v_x_i32m4(0, vl);

    while (loop_cnt > 0)
    {
        vl = __riscv_vsetvl_e32m4(loop_cnt);

        vint32m4_t col_val = __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(col_ptr, vl), vl);
        vint32m4_t row_val = __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(row_ptr, vl), vl);

        acc = __riscv_vmacc_vv_i32m4(acc, col_val, row_val, vl);
        sum = __riscv_vadd_vv_i32m4(sum, col_val, vl);

        loop_cnt -= vl;
        row_ptr += vl;
        col_ptr += vl;
    }

    vl = __riscv_vsetvl_e32m4(row_elements);

    vint32m1_t reduct_acc = __riscv_vmv_v_x_i32m1(0, vl);
    reduct_acc = __riscv_vredsum_vs_i32m4_i32m1(acc, reduct_acc, vl);
    acc_n0 = __riscv_vmv_x_s_i32m1_i32(reduct_acc);

    vint32m1_t reduct_sum = __riscv_vmv_v_x_i32m1(0, vl);
    reduct_sum = __riscv_vredsum_vs_i32m4_i32m1(sum, reduct_sum, vl);
    sum_tmp = __riscv_vmv_x_s_i32m1_i32(reduct_sum);

#else
    for (int i = 0; i < row_elements; i++)
    {
        sum_tmp += col_base[i];
        acc_n0 += row_base[i] * col_base[i];
    }
#endif

    *sum_col = sum_tmp;
    *output = acc_n0;
    return MURISCV_NN_SUCCESS;
}

/**
 * @} end of NNBasicMath group
 */
