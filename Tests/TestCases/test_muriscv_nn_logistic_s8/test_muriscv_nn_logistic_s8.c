/*
 * Copyright (C) 2021-2022 Chair of Electronic Design Automation, TUM.
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

#include <muriscv_nn_functions.h>
#include <unity.h>
#include <stdio.h>
#include <string.h>
#include "../../TestData/test_muriscv_nn_logistic_s8/test_data.h"

void setUp(void) {}
void tearDown(void) {}

// Print before entering each kernel call so a simulator hang is localized.
static void entering(const char *name)
{
    printf("Entering logistic_s8: %s\n", name);
    fflush(stdout);
}

static void check_reference(int32_t zero_point, const int8_t *reference, int size)
{
    int8_t output[LOGISTIC_SIZE + 2];
    memset(output, 42, sizeof(output));
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
        muriscv_nn_logistic_s8(logistic_input, output + 1, size, zero_point, 60, 1073741824, 25));
    for (int i = 0; i < size; ++i)
        TEST_ASSERT_INT_WITHIN(1, reference[i], output[i + 1]);
    TEST_ASSERT_EQUAL_INT8(42, output[0]);
    for (int i = size + 1; i < LOGISTIC_SIZE + 2; ++i)
        TEST_ASSERT_EQUAL_INT8(42, output[i]);
}

void test_reference_and_tails(void)
{
    const int sizes[] = {1, 17, 255, 256, LOGISTIC_SIZE};
    for (unsigned i = 0; i < sizeof(sizes)/sizeof(sizes[0]); ++i)
    {
        printf("size=%d\n", sizes[i]);
        entering("reference and tails");
        check_reference(0, output_ref_zero, sizes[i]);
    }
}

void test_nonzero_zero_point(void)
{
    entering("nonzero zero point");
    check_reference(-37, output_ref_offset, LOGISTIC_SIZE);
}

void test_in_place(void)
{
    int8_t data[LOGISTIC_SIZE], output[LOGISTIC_SIZE];
    memcpy(data, logistic_input, sizeof(data));
    entering("in-place reference");
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
        muriscv_nn_logistic_s8(data, output, LOGISTIC_SIZE, -37, 60, 1073741824, 25));
    entering("in-place");
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
        muriscv_nn_logistic_s8(data, data, LOGISTIC_SIZE, -37, 60, 1073741824, 25));
    TEST_ASSERT_EQUAL_INT8_ARRAY(output, data, LOGISTIC_SIZE);
}

void test_repeated_table_address(void)
{
    int8_t input[LOGISTIC_SIZE] = {0}, output[LOGISTIC_SIZE];
    entering("all lanes gather the same address");
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
        muriscv_nn_logistic_s8(input, output, LOGISTIC_SIZE, 0, 256, 1073741824, 25));
    for (int i = 0; i < LOGISTIC_SIZE; ++i) TEST_ASSERT_EQUAL_INT8(0, output[i]);
}

void test_all_gathers_masked(void)
{
    int8_t input[LOGISTIC_SIZE], output[LOGISTIC_SIZE];
    memset(input, 127, sizeof(input));
    entering("all gathers masked off, left-shift scaling");
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
        muriscv_nn_logistic_s8(input, output, LOGISTIC_SIZE, 0, 256, 1073741824, 30));
    for (int i = 0; i < LOGISTIC_SIZE; ++i) TEST_ASSERT_EQUAL_INT8(127, output[i]);
}

void test_mixed_gather_masks(void)
{
    const int8_t pattern[] = {-128, -4, 0, 4, 127};
    const int8_t expected[] = {-128, -123, 0, 123, 127}; // scale 1/8 * 8 = 1
    int8_t input[LOGISTIC_SIZE], output[LOGISTIC_SIZE];
    for (int i = 0; i < LOGISTIC_SIZE; ++i) input[i] = pattern[i % 5];
    entering("mixed valid and out-of-range table indices");
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
        muriscv_nn_logistic_s8(input, output, LOGISTIC_SIZE, 0, 256, 1073741824, 28));
    for (int i = 0; i < LOGISTIC_SIZE; ++i)
        TEST_ASSERT_INT_WITHIN(1, expected[i % 5], output[i]);
}

void test_shift_and_radius_limits(void)
{
    int8_t output[LOGISTIC_SIZE];
    entering("large right shift");
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
        muriscv_nn_logistic_s8(logistic_input, output, LOGISTIC_SIZE, 0, 256, INT32_MAX, -31));
    for (int i = 0; i < LOGISTIC_SIZE; ++i) TEST_ASSERT_EQUAL_INT8(0, output[i]);
    entering("radius zero");
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
        muriscv_nn_logistic_s8(logistic_input, output, LOGISTIC_SIZE, 0, 0, 1073741824, 25));
    for (int i = 0; i < LOGISTIC_SIZE; ++i)
        TEST_ASSERT_EQUAL_INT8(logistic_input[i] <= 0 ? -128 : 127, output[i]);
}

void test_empty_and_invalid(void)
{
    int8_t output = 42;
    entering("empty and invalid arguments");
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS, muriscv_nn_logistic_s8(NULL, &output, 0, 0, 1, 1, 0));
    TEST_ASSERT_EQUAL(MURISCV_NN_ARG_ERROR, muriscv_nn_logistic_s8(NULL, &output, 1, 0, 1, 1, 0));
    TEST_ASSERT_EQUAL(MURISCV_NN_ARG_ERROR, muriscv_nn_logistic_s8(logistic_input, &output, 1, 0, 1, 1, 31));
    TEST_ASSERT_EQUAL_INT8(42, output);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_reference_and_tails);
    RUN_TEST(test_nonzero_zero_point);
    RUN_TEST(test_in_place);
    RUN_TEST(test_repeated_table_address);
    RUN_TEST(test_all_gathers_masked);
    RUN_TEST(test_mixed_gather_masks);
    RUN_TEST(test_shift_and_radius_limits);
    RUN_TEST(test_empty_and_invalid);
    int failures = UNITY_END();
#if defined(__riscv) || defined(__riscv__)
    if (failures != 0) __asm__ volatile("unimp");
#endif
    return failures;
}
