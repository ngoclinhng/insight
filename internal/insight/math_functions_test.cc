// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/internal/math_functions.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {
namespace internal {

using ::testing::FloatEq;
using ::testing::DoubleEq;
using ::testing::ElementsAreArray;

TEST(axpy, ForFloatDataType) {
  const int N = 3;
  float X[] = {1.0f, 2.0f, 3.0f};
  float Y[] = {-1.0f, 0.0f, 6.0f};
  float alpha = 2.0f;

  axpy(N, alpha, X, Y);

  // We expect that Y = alpha * X + Y
  float expected_results[] = {1.0f, 4.0f, 12.0f};
  EXPECT_THAT(Y, ElementsAreArray(expected_results));
}

TEST(axpy, ForDoubleDataType) {
  const int N = 3;
  double X[] = {-2.0, 1.0, 10.0};
  double Y[] = {0.0, 0.0, 0.0};
  double alpha = 4.0;

  axpy(N, alpha, X, Y);

  // We expect that Y = alpha * X + Y.
  double expected_results[] = {-8.0, 4.0, 40.0};
  EXPECT_THAT(Y, ElementsAreArray(expected_results));
}

}  // namespace internal
}  // namespace insight
