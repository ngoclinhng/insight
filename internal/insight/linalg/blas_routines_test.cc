// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include "insight/linalg/detail/blas_routines.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {
namespace linalg_detail {

using ::testing::FloatEq;
using ::testing::DoubleEq;
using ::testing::ElementsAreArray;
using ::testing::ElementsAre;

TEST(blas_axpy, ForFloatDataType) {
  const int N = 3;
  float X[] = {1.0f, 2.0f, 3.0f};
  float Y[] = {-1.0f, 0.0f, 6.0f};
  float alpha = 2.0f;

  blas_axpy(N, alpha, X, Y);

  // We expect that Y = alpha * X + Y
  float expected_results[] = {1.0f, 4.0f, 12.0f};
  EXPECT_THAT(Y, ElementsAreArray(expected_results));
}

TEST(blas_axpy, ForDoubleDataType) {
  const int N = 3;
  double X[] = {-2.0, 1.0, 10.0};
  double Y[] = {0.0, 0.0, 0.0};
  double alpha = 4.0;

  blas_axpy(N, alpha, X, Y);

  // We expect that Y = alpha * X + Y.
  double expected_results[] = {-8.0, 4.0, 40.0};
  EXPECT_THAT(Y, ElementsAreArray(expected_results));
}

TEST(blas_axpby, ForFloatDataType) {
  const int N = 3;
  float alpha = 1.0f;
  float X[] = {1.0f, 3.0f, 5.0f};
  float beta = -1.0f;
  float Y[] = {-1.0f, 0.0f, 2.0f};

  blas_axpby(N, alpha, X, beta, Y);

  // We expect that Y = alpha * X + beta * Y.
  float expected_result[] = {2.0f, 3.0f, 3.0f};

  EXPECT_THAT(Y, ElementsAreArray(expected_result));
}

TEST(blas_axpby, ForDoubleDataType) {
  const int N = 3;
  double alpha = 2.0;
  double X[] = {1.0, 2.0, 3.0};
  double beta = -1.0;
  double Y[] = {0.0, 0.0, 0.0};

  blas_axpby(N, alpha, X, beta, Y);

  double expected_result[] = {2.0, 4.0, 6.0};
  EXPECT_THAT(Y, ElementsAreArray(expected_result));
}

TEST(blas_add, ForFloatDataType) {
  const int N = 3;
  float X[] = {1.0f, 2.0f, 3.0f};
  float Y[] = {-2.0f, 0.5f, 10.0f};
  float Z[3];

  blas_add(N, X, Y, Z);

  float expected_result[] = {-1.0f, 2.5f, 13.0f};
  EXPECT_THAT(Z, ElementsAreArray(expected_result));
}

TEST(blas_add, ForDoubleDataType) {
  const int N = 3;
  double X[] = {2.0, 4.0, 6.0};
  double Y[] = {1.0, 2.0, 3.0};
  double Z[] = {-10.0, 2.0, 0.0};

  blas_add(N, X, Y, Z);

  double expected_result[] = {3.0, 6.0, 9.0};
  EXPECT_THAT(Z, ElementsAreArray(expected_result));
}

TEST(blas_sub, ForFloatDataType) {
  const int N = 5;
  float X[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float Y[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  float Z[N];

  blas_sub(N, X, Y, Z);

  float expected_result[N] = {-9.0f, -18.0f, -27.0f, -36.0f, -45.0f};
  EXPECT_THAT(Z, ElementsAreArray(expected_result));
}

TEST(blas_sub, ForDoubleDataType) {
  const int N = 5;
  double X[] = {1.0, 2.0, 3.0, 4.0, 5.0};
  double Y[] = {10.0, 20.0, 30.0, 40.0, 50.0};
  double Z[N];

  blas_sub(N, X, Y, Z);

  double expected_result[N] = {-9.0, -18.0, -27.0, -36.0, -45.0};
  EXPECT_THAT(Z, ElementsAreArray(expected_result));
}

TEST(blas_mul, ForFloatDataType) {
  const int N = 5;
  float X[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  float Y[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float Z[N];

  blas_mul(N, X, Y, Z);

  double expected_result[] = {10.0f, 40.0f, 90.0f, 160.0f, 250.0f};
  EXPECT_THAT(Z, ElementsAreArray(expected_result));
}

TEST(blas_mul, ForDoubleDataType) {
  const int N = 3;
  double X[] = {1.0, 2.0, 3.0};
  double Y[] = {2.0, 3.0, 4.0};
  double Z[N];

  blas_mul(N, X, Y, Z);

  double expected_result[] = {2.0, 6.0, 12.0};
  EXPECT_THAT(Z, ElementsAreArray(expected_result));
}

TEST(blas_div, ForFloatDataType) {
  const int N = 5;
  float X[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
  float Y[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float Z[N];

  blas_div(N, X, Y, Z);

  for (int i = 0; i < N; ++i) {
    EXPECT_THAT(Z[i], FloatEq(10));
  }
  // TODO(Linh): Somehow gtest complains about this. May it be the bug in
  // gtest?
  // float expected_result[N] = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
  // EXPECT_THAT(Z, ElementsAreArray(expected_result));
}

TEST(blas_div, ForDoubleDataType) {
  const int N = 3;
  double X[] = {6.0, 4.0, 22.0};
  double Y[] = {3.0, 2.0, 2.0};
  double Z[N];

  blas_div(N, X, Y, Z);

  double expected_result[] = {2.0, 2.0, 11.0};
  EXPECT_THAT(Z, ElementsAreArray(expected_result));
}

TEST(math_functions, blas_sqrt) {
  int n = 4;
  double x[] = {100, 10000, 64, 144};
  double y[4];

  blas_sqrt(n, x, y);

  EXPECT_THAT(y, ElementsAre(10, 100, 8, 12));
}

}  // namespace linalg_detail
}  // namespace insight
