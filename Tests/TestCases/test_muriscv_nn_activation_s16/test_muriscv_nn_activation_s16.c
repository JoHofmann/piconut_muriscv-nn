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

#include "../../TestData/test_muriscv_nn_activation_s16/test_data.h"
#include "../../Utils/validate.h"

#include <string.h>

void setUp(void)
{ /* set stuff up here */
}

void tearDown(void)
{ /* clean stuff up here */
}

static void check_sigmoid(const int32_t left_shift, const int16_t *output_ref)
{
    int16_t output[SIGMOID_SIZE] = {0};
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
                      muriscv_nn_activation_s16(sigmoid_input, output, SIGMOID_SIZE, left_shift, MURISCV_SIGMOID));
    TEST_ASSERT_TRUE(validate_s16(output, output_ref, SIGMOID_SIZE));
}

void test_sigmoid_shift_minus_two(void)
{
    check_sigmoid(-2, sigmoid_output_ref_shift_minus_two);
}

void test_sigmoid_shift_minus_one(void)
{
    check_sigmoid(-1, sigmoid_output_ref_shift_minus_one);
}

void test_sigmoid_shift_zero(void)
{
    check_sigmoid(0, sigmoid_output_ref_shift_zero);
}

void test_sigmoid_shift_one(void)
{
    check_sigmoid(1, sigmoid_output_ref_shift_one);
}

void test_sigmoid_shift_two(void)
{
    check_sigmoid(2, sigmoid_output_ref_shift_two);
}

void test_sigmoid_in_place(void)
{
    int16_t data[SIGMOID_SIZE];
    memcpy(data, sigmoid_input, sizeof(data));
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
                      muriscv_nn_activation_s16(data, data, SIGMOID_SIZE, 0, MURISCV_SIGMOID));
    TEST_ASSERT_TRUE(validate_s16(data, sigmoid_output_ref_shift_zero, SIGMOID_SIZE));
}

void test_sigmoid_empty(void)
{
    int16_t output = 1234;
    TEST_ASSERT_EQUAL(MURISCV_NN_SUCCESS,
                      muriscv_nn_activation_s16(sigmoid_input, &output, 0, 0, MURISCV_SIGMOID));
    TEST_ASSERT_EQUAL_INT16(1234, output);
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_sigmoid_shift_minus_two);
    RUN_TEST(test_sigmoid_shift_minus_one);
    RUN_TEST(test_sigmoid_shift_zero);
    RUN_TEST(test_sigmoid_shift_one);
    RUN_TEST(test_sigmoid_shift_two);
    RUN_TEST(test_sigmoid_in_place);
    RUN_TEST(test_sigmoid_empty);

#if defined(__riscv) || defined(__riscv__)
    /* If an error occurred make sure the simulator fails so CTest can detect that. */
    int failures = UNITY_END();
    if (failures != 0)
    {
        __asm__ volatile("unimp");
    }
    return failures;
#else
    return UNITY_END();
#endif
}
