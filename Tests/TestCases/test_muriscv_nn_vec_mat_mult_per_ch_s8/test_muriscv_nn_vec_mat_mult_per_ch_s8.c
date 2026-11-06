/* SPDX-License-Identifier: Apache-2.0 */

#include <muriscv_nn_functions.h>
#include <muriscv_nn_support_functions.h>
#include <unity.h>

enum { MAX_ROWS = 517, MAX_COLS = 17, MAX_STRIDE = 3 };

void setUp(void) {}
void tearDown(void) {}

static void check_case(const int32_t rows,
                       const int32_t cols,
                       const int32_t lhs_offset,
                       const int32_t rhs_offset,
                       const int32_t stride,
                       const int use_bias,
                       const int32_t activation_min,
                       const int32_t activation_max)
{
    const int32_t shifts[] = {-31, -17, -9, -2, -1, 0, 1, 3};
    int8_t lhs[MAX_COLS];
    int8_t rhs[MAX_ROWS * MAX_COLS];
    int8_t rhs_input_major[MAX_ROWS * MAX_COLS];
    int32_t bias[MAX_ROWS];
    int32_t multipliers[MAX_ROWS];
    int32_t channel_shifts[MAX_ROWS];
    int8_t expected[MAX_ROWS * MAX_STRIDE + 2];
    int8_t actual[sizeof(expected)];
    const int32_t dst_offset = -7;

    memset(actual, 0x5a, sizeof(actual));
    memset(expected, 0x5a, sizeof(expected));
    for (int32_t col = 0; col < cols; col++)
    {
        lhs[col] = (int8_t)((col * 73 + 19) % 256 - 128);
    }
    for (int32_t row = 0; row < rows; row++)
    {
        // Small signed biases also exercise rounding without saturation
        // in the zero-column cases, including negative halfway values.
        bias[row] = row % 31 - 15;
        multipliers[row] = 0x40000000 + (row % 3) * 0x10000000;
        channel_shifts[row] = shifts[row % (sizeof(shifts) / sizeof(shifts[0]))];
        int32_t acc = use_bias ? bias[row] : 0;
        for (int32_t col = 0; col < cols; col++)
        {
            rhs[row * cols + col] = (int8_t)((row * 43 + col * 61) % 256 - 128);
            rhs_input_major[col * rows + row] = rhs[row * cols + col];
            acc += (lhs[col] + lhs_offset) * (rhs[row * cols + col] + rhs_offset);
        }
        acc = muriscv_nn_requantize(acc, multipliers[row], channel_shifts[row]);
        acc += dst_offset;
        expected[1 + row * stride] = (int8_t)MIN(MAX(acc, activation_min), activation_max);
    }

    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
                      muriscv_nn_vec_mat_mult_t_per_ch_s8(lhs,
                                                        rhs,
                                                        NULL,
                                                        use_bias ? bias : NULL,
                                                        actual + 1,
                                                        lhs_offset,
                                                        dst_offset,
                                                        multipliers,
                                                        channel_shifts,
                                                        cols,
                                                        rows,
                                                        activation_min,
                                                        activation_max,
                                                        stride,
                                                        rhs_offset));
    // Compare the guards and unused output slots as well as the results.
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, actual, sizeof(expected));

    memset(actual, 0x5a, sizeof(actual));
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
                      muriscv_nn_vec_mat_mult_per_ch_s8(lhs,
                                                      rhs_input_major,
                                                      NULL,
                                                      use_bias ? bias : NULL,
                                                      actual + 1,
                                                      lhs_offset,
                                                      dst_offset,
                                                      multipliers,
                                                      channel_shifts,
                                                      cols,
                                                      rows,
                                                      activation_min,
                                                      activation_max,
                                                      stride,
                                                      rhs_offset));
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, actual, sizeof(expected));
}

void per_channel_blocks_and_rounding(void)
{
    // Cross the m8 block boundaries at VLEN 128 and 1024, and process
    // multiple full blocks plus a tail at both vector lengths.
    const int32_t rows[] = {0, 1, 2, 3, 5, 31, 32, 33, 255, 256, 257, MAX_ROWS};
    const int32_t cols[] = {0, 1, 7, MAX_COLS};
    for (size_t r = 0; r < sizeof(rows) / sizeof(rows[0]); r++)
    {
        for (size_t c = 0; c < sizeof(cols) / sizeof(cols[0]); c++)
        {
            for (int use_bias = 0; use_bias <= 1; use_bias++)
            {
                check_case(rows[r], cols[c], 0, 0, 1, use_bias, -128, 127);
                check_case(rows[r], cols[c], 128, -127, 1, use_bias, -97, 83);
                check_case(rows[r], cols[c], -127, 128, 1, use_bias, -97, 83);
            }
        }
    }
}

void per_channel_spaced_outputs(void)
{
    for (int32_t rows = 0; rows <= 8; rows++)
    {
        for (int use_bias = 0; use_bias <= 1; use_bias++)
        {
            check_case(rows, MAX_COLS, 13, 0, MAX_STRIDE, use_bias, -128, 127);
            check_case(rows, MAX_COLS, -9, 7, MAX_STRIDE, use_bias, -97, 83);
        }
    }
}

void per_channel_fully_connected_batches(void)
{
    enum { BATCHES = 3, INPUTS = 7, OUTPUTS = 259 };
    const int32_t shifts[] = {-8, -3, 0, 1};
    int8_t input[BATCHES * INPUTS];
    int8_t weights[OUTPUTS * INPUTS];
    int32_t bias[OUTPUTS];
    int32_t multipliers[OUTPUTS];
    int32_t channel_shifts[OUTPUTS];
    int8_t expected[BATCHES * OUTPUTS + 2];
    int8_t actual[sizeof(expected)];
    const muriscv_nn_context ctx = {0};
    const muriscv_nn_dims input_dims = {.n = BATCHES, .h = 1, .w = 1, .c = INPUTS};
    const muriscv_nn_dims filter_dims = {.n = INPUTS, .c = OUTPUTS};
    const muriscv_nn_dims bias_dims = {.c = OUTPUTS};
    const muriscv_nn_dims output_dims = {.n = BATCHES, .c = OUTPUTS};
    const muriscv_nn_per_channel_quant_params quant = {.multiplier = multipliers, .shift = channel_shifts};
    muriscv_nn_fc_params params = {
        .input_offset = -3, .output_offset = 5, .activation = {.min = -97, .max = 83}};

    for (int32_t i = 0; i < BATCHES * INPUTS; i++)
    {
        input[i] = (int8_t)((i * 13 + 7) % 16 - 8);
    }
    for (int32_t row = 0; row < OUTPUTS; row++)
    {
        bias[row] = row % 31 - 15;
        multipliers[row] = 0x40000000 + (row % 3) * 0x10000000;
        channel_shifts[row] = shifts[row % 4];
        for (int32_t col = 0; col < INPUTS; col++)
        {
            const int8_t weight = (int8_t)((row * 3 + col * 7) % 11 - 5);
#if defined(FC_WEIGHTS_TRANSPOSED)
            weights[row * INPUTS + col] = weight;
#else
            weights[col * OUTPUTS + row] = weight;
#endif
        }
    }

    for (int use_bias = 0; use_bias <= 1; use_bias++)
    {
        for (int32_t rhs_offset = 0; rhs_offset <= 2; rhs_offset += 2)
        {
            params.filter_offset = rhs_offset;
            memset(expected, 0x5a, sizeof(expected));
            memset(actual, 0x5a, sizeof(actual));
            for (int32_t batch = 0; batch < BATCHES; batch++)
            {
                for (int32_t row = 0; row < OUTPUTS; row++)
                {
                    int32_t acc = use_bias ? bias[row] : 0;
                    for (int32_t col = 0; col < INPUTS; col++)
                    {
                        const int32_t weight = (row * 3 + col * 7) % 11 - 5;
                        acc += (input[batch * INPUTS + col] + params.input_offset) * (weight + rhs_offset);
                    }
                    acc = muriscv_nn_requantize(acc, multipliers[row], channel_shifts[row]);
                    acc += params.output_offset;
                    expected[1 + batch * OUTPUTS + row] =
                        (int8_t)MIN(MAX(acc, params.activation.min), params.activation.max);
                }
            }
            TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
                              muriscv_nn_fully_connected_per_channel_s8(&ctx,
                                                                       &params,
                                                                       &quant,
                                                                       &input_dims,
                                                                       input,
                                                                       &filter_dims,
                                                                       weights,
                                                                       &bias_dims,
                                                                       use_bias ? bias : NULL,
                                                                       &output_dims,
                                                                       actual + 1));
            TEST_ASSERT_EQUAL_INT8_ARRAY(expected, actual, sizeof(expected));
        }
    }
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(per_channel_blocks_and_rounding);
    RUN_TEST(per_channel_spaced_outputs);
    RUN_TEST(per_channel_fully_connected_batches);
    int failures = UNITY_END();
#if defined(__riscv) || defined(__riscv__)
    if (failures != 0)
    {
        __asm__ volatile("unimp");
    }
#endif
    return failures;
}
