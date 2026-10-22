/*
 * Copyright (C) 2022 Arm Limited or its affiliates.
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
 * Modifications copyright (C)      2026 Johannes Hofmann, THA <johannes.hofmann1@tha.de>
 */

#include "muriscv_nn_compatible.h"
#include "muriscv_nn_functions.h"
#include "muriscv_nn_support_functions.h"

#define ACCUM_BITS 12

#if defined(USE_VEXT)

/*
 * Helper functions for vector operations.
 * Note: These helper functions are created using AI (Claude Opus 5.5).
 */

/*
 * Rounding arithmetic right shift, identical to vssra.vx with vxrm = RNU:
 * (x >> n) + bit (n - 1) of x. Valid for 1 <= n <= 31, cannot overflow.
 * For x >= 0 this is identical to DIV_POW2(x, n).
 */
static inline vint32m4_t vrshr_rnu_i32m4(vint32m4_t x, size_t n, size_t vl)
{
    vint32m4_t q = __riscv_vsra_vx_i32m4(x, n, vl);
    vint32m4_t r = __riscv_vand_vx_i32m4(__riscv_vsra_vx_i32m4(x, n - 1, vl), 1, vl);
    return __riscv_vadd_vv_i32m4(q, r, vl);
}

/* Vector version of muriscv_nn_exp_on_negative_values (gemmlowp algorithm). Bit-exact. */
static inline vint32m4_t exp_on_negative_values_i32m4(vint32m4_t val, size_t vl)
{
    const int shift = 24;

    /* val = f - rem, f in [-1/4, 0) */
    vint32m4_t vmq = __riscv_vsub_vx_i32m4(__riscv_vand_vx_i32m4(val, (1 << shift) - 1, vl), 1 << shift, vl);
    vint32m4_t rem = __riscv_vsub_vv_i32m4(vmq, val, vl);

    /* exp(f): Taylor series around -1/8 */
    vint32m4_t x = __riscv_vadd_vx_i32m4(__riscv_vsll_vx_i32m4(vmq, 5, vl), 1 << 28, vl);
    vint32m4_t x2 = __riscv_vsmul_vv_i32m4(x, x, __RISCV_VXRM_RNU, vl);
    vint32m4_t x3 = __riscv_vsmul_vv_i32m4(x2, x, __RISCV_VXRM_RNU, vl);
    vint32m4_t x4 = __riscv_vsmul_vv_i32m4(x2, x2, __RISCV_VXRM_RNU, vl);

    vint32m4_t t = vrshr_rnu_i32m4(x4, 2, vl); /* DIV_POW2(x4, 2), x4 >= 0 */
    t = __riscv_vadd_vv_i32m4(t, x3, vl);
    t = __riscv_vsmul_vx_i32m4(t, 715827883, __RISCV_VXRM_RNU, vl); /* * 1/3 */
    t = __riscv_vadd_vv_i32m4(t, x2, vl);
    t = vrshr_rnu_i32m4(t, 1, vl); /* DIV_POW2(t, 1), t >= 0 */
    t = __riscv_vadd_vv_i32m4(x, t, vl);
    t = __riscv_vsmul_vx_i32m4(t, 1895147668, __RISCV_VXRM_RNU, vl);
    vint32m4_t result = __riscv_vadd_vx_i32m4(t, 1895147668, vl);

    /* exp(-rem): multiply by exp(-2^k / 4) for every set bit (24 + k) of rem */
    static const int32_t c[7] = {1672461947, 1302514674, 790015084, 290630308, 39332535, 720401, 242};
    for (int k = 0; k < 7; ++k)
    {
        vbool8_t m = __riscv_vmsne_vx_i32m4_b8(__riscv_vand_vx_i32m4(rem, 1 << (shift + k), vl), 0, vl);
        vint32m4_t p = __riscv_vsmul_vx_i32m4(result, c[k], __RISCV_VXRM_RNU, vl);
        result = __riscv_vmerge_vvm_i32m4(result, p, m, vl); /* m ? p : result */
    }

    /* val == 0 -> NN_Q31_MAX */
    vbool8_t z = __riscv_vmseq_vx_i32m4_b8(val, 0, vl);
    return __riscv_vmerge_vxm_i32m4(result, NN_Q31_MAX, z, vl);
}

/*
 * EXP_ON_NEG(MUL_SAT(diff * (1 << shift), mult)) for diff in [-255, 0].
 * Lanes with diff < diff_min are set to 0. A 0 gives exactly the scalar result
 * of a skipped lane in both passes: term = 0 in the sum, Q7_MIN / Q15_MIN in the output.
 */
static inline vint32m4_t
softmax_exp_i32m4(vint32m4_t diff, const int32_t shift, const int32_t mult, const int32_t diff_min, size_t vl)
{
    vbool8_t drop = __riscv_vmslt_vx_i32m4_b8(diff, diff_min, vl);
    vint32m4_t a = __riscv_vsll_vx_i32m4(diff, shift, vl);
    a = __riscv_vsmul_vx_i32m4(a, mult, __RISCV_VXRM_RNU, vl);
    vint32m4_t e = exp_on_negative_values_i32m4(a, vl);
    return __riscv_vmerge_vxm_i32m4(e, 0, drop, vl); /* drop ? 0 : e */
}

/* exp for vl input elements, evaluated directly for every element */
static inline vint32m4_t row_exp_i32m4(const int8_t *in,
                                       const int32_t max,
                                       const int32_t shift,
                                       const int32_t mult,
                                       const int32_t diff_min,
                                       size_t vl)
{
    vint32m4_t diff = __riscv_vsext_vf4_i32m4(__riscv_vle8_v_i8m1(in, vl), vl);
    diff = __riscv_vsub_vx_i32m4(diff, max, vl);
    return softmax_exp_i32m4(diff, shift, mult, diff_min, vl);
}

#endif /* USE_VEXT */

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup Softmax
 * @{
 */

/*
 * Softmax function with s8 input and output of s8 or s16.
 *
 * Refer header file for details.
 *
 */
void muriscv_nn_softmax_common_s8(const int8_t *input,
                                  const int32_t num_rows,
                                  const int32_t row_size,
                                  const int32_t mult,
                                  const int32_t shift,
                                  const int32_t diff_min,
                                  const bool int16_output,
                                  void *output)
{

#if defined(USE_VEXT)

    size_t avl, vl;

    for (int32_t row_idx = 0; row_idx < num_rows; ++row_idx)
    {
        /* ---------- max ---------- */
        vint8m1_t vmax = __riscv_vmv_s_x_i8m1(INT8_MIN, 1);
        {
            const int8_t *in = input;
            avl = row_size;
            while (avl > 0)
            {
                vl = __riscv_vsetvl_e8m4(avl);
                vmax = __riscv_vredmax_vs_i8m4_i8m1(__riscv_vle8_v_i8m4(in, vl), vmax, vl);
                in += vl;
                avl -= vl;
            }
        }
        const int32_t max = __riscv_vmv_x_s_i8m1_i8(vmax);

        /* ---------- sum ---------- */
        vint32m1_t vsum = __riscv_vmv_s_x_i32m1(0, 1);
        {
            const int8_t *in = input;
            avl = row_size;
            while (avl > 0)
            {
                vl = __riscv_vsetvl_e32m4(avl);

                vint32m4_t term = row_exp_i32m4(in, max, shift, mult, diff_min, vl);
                term = vrshr_rnu_i32m4(term, ACCUM_BITS, vl); /* DIV_POW2(exp, ACCUM_BITS) */

                vsum = __riscv_vredsum_vs_i32m4_i32m1(term, vsum, vl);
                in += vl;
                avl -= vl;
            }
        }
        const int32_t sum = __riscv_vmv_x_s_i32m1_i32(vsum);

        /* ---------- output ---------- */
        const int8_t *in = input;
        avl = row_size;

        /* sum > 0 if at least one lane was kept (then sum >= 2^19 and headroom <= 12) */
        const int32_t headroom = (sum > 0) ? __builtin_clz((uint32_t)sum) : 32;
        const int32_t bits_over_unit = ACCUM_BITS - headroom + (int16_output ? 15 : 23);

        /* sum == 0:            every lane was dropped (diff < diff_min)
         * bits_over_unit > 31: every kept lane rounds to 0 (vsra only uses 5 shift bits)
         * In both cases every output is the minimum value. */
        const bool all_min = (sum <= 0) || (bits_over_unit > 31);
        const int32_t shifted_scale = all_min ? 0 : ONE_OVER1((sum << headroom) - (1 << 31));

        if (int16_output)
        {
            int16_t *out = (int16_t *)output + row_idx * row_size;

            if (all_min)
            {
                while (avl > 0)
                {
                    vl = __riscv_vsetvl_e16m2(avl);
                    __riscv_vse16_v_i16m2(out, __riscv_vmv_v_x_i16m2(Q15_MIN, vl), vl);
                    out += vl;
                    avl -= vl;
                }
            }
            else
            {
                while (avl > 0)
                {
                    vl = __riscv_vsetvl_e32m4(avl);

                    vint32m4_t res = row_exp_i32m4(in, max, shift, mult, diff_min, vl);
                    res = __riscv_vsmul_vx_i32m4(res, shifted_scale, __RISCV_VXRM_RNU, vl); /* MUL_SAT  */
                    res = vrshr_rnu_i32m4(res, bits_over_unit, vl);                         /* DIV_POW2 */
                    res = __riscv_vadd_vx_i32m4(res, Q15_MIN, vl);

                    /* CLAMP to [Q15_MIN, Q15_MAX]: saturating narrow 32 -> 16 */
                    vint16m2_t o = __riscv_vnclip_wx_i16m2(res, 0, __RISCV_VXRM_RNU, vl);

                    __riscv_vse16_v_i16m2(out, o, vl);
                    in += vl;
                    out += vl;
                    avl -= vl;
                }
            }
        }
        else
        {
            int8_t *out = (int8_t *)output + row_idx * row_size;

            if (all_min)
            {
                while (avl > 0)
                {
                    vl = __riscv_vsetvl_e8m4(avl);
                    __riscv_vse8_v_i8m4(out, __riscv_vmv_v_x_i8m4(Q7_MIN, vl), vl);
                    out += vl;
                    avl -= vl;
                }
            }
            else
            {
                while (avl > 0)
                {
                    vl = __riscv_vsetvl_e32m4(avl);

                    vint32m4_t res = row_exp_i32m4(in, max, shift, mult, diff_min, vl);
                    res = __riscv_vsmul_vx_i32m4(res, shifted_scale, __RISCV_VXRM_RNU, vl); /* MUL_SAT  */
                    res = vrshr_rnu_i32m4(res, bits_over_unit, vl);                         /* DIV_POW2 */
                    res = __riscv_vadd_vx_i32m4(res, Q7_MIN, vl);

                    /* CLAMP to [Q7_MIN, Q7_MAX]: saturating narrow 32 -> 16 -> 8 */
                    vint16m2_t r16 = __riscv_vnclip_wx_i16m2(res, 0, __RISCV_VXRM_RNU, vl);
                    vint8m1_t o = __riscv_vnclip_wx_i8m1(r16, 0, __RISCV_VXRM_RNU, vl);

                    __riscv_vse8_v_i8m1(out, o, vl);
                    in += vl;
                    out += vl;
                    avl -= vl;
                }
            }
        }

        input += row_size;
    }

#else /* USE_VEXT */

    const int32_t mask = (1 << shift);

    int32_t col = 0;
    int32_t row_idx;

    for (row_idx = 0; row_idx < num_rows; ++row_idx)
    {
        // Find the maximum value in order to ensure numerical stability
        int8_t max = *input;

        for (col = 1; col < row_size; ++col)
        {
            max = MAX(max, input[col]);
        }

        int32_t diff = 0;
        int32_t sum = 0;

        for (col = 0; col < row_size; ++col)
        {
            diff = input[col] - max;
            if (diff >= diff_min)
            {
                sum += DIV_POW2(EXP_ON_NEG(MUL_SAT(diff * mask, mult)), ACCUM_BITS);
            }
        }

        const int32_t headroom = __CLZ(sum);
        const int32_t shifted_scale = ONE_OVER1((sum > 0 ? sum << headroom : 0) - (1 << 31));
        int32_t bits_over_unit;

        if (int16_output)
        {
            int16_t *output_s16 = (int16_t *)output + row_idx * row_size;

            bits_over_unit = ACCUM_BITS - headroom + 15;

            for (col = 0; col < row_size; ++col)
            {
                diff = input[col] - max;

                if (diff >= diff_min)
                {
                    const int32_t res =
                        DIV_POW2(MUL_SAT(shifted_scale, EXP_ON_NEG(MUL_SAT(diff * mask, mult))), bits_over_unit) +
                        Q15_MIN;
                    output_s16[col] = (int16_t)CLAMP(res, (int32_t)Q15_MAX, (int32_t)Q15_MIN);
                }
                else
                {
                    output_s16[col] = Q15_MIN;
                }
            }
        }
        else
        {
            int8_t *output_s8 = (int8_t *)output + row_idx * row_size;

            bits_over_unit = ACCUM_BITS - headroom + 23;

            for (col = 0; col < row_size; ++col)
            {
                diff = input[col] - max;
                if (diff >= diff_min)
                {
                    const int32_t res =
                        DIV_POW2(MUL_SAT(shifted_scale, EXP_ON_NEG(MUL_SAT(diff * mask, mult))), bits_over_unit) +
                        Q7_MIN;
                    output_s8[col] = (int8_t)CLAMP(res, (int32_t)Q7_MAX, (int32_t)Q7_MIN);
                }
                else
                {
                    output_s8[col] = Q7_MIN;
                }
            }
        }

        input += row_size;
    }

#endif /* USE_VEXT */
}

/**
 * @} end of NNBasicMath group
 */
