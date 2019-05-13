// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include <vector>

#include "insight/matrix.h"

#include "glog/logging.h"
#include "gmock/gmock.h"

namespace insight {
using ::testing::DoubleEq;
using ::testing::FloatEq;
using ::testing::ElementsAre;

TEST(matrix, default_constructor) {
  matrix<double> m;

  ASSERT_EQ(m.num_rows(), 0);
  ASSERT_EQ(m.num_cols(), 0);
  ASSERT_EQ(m.shape().first, 0);
  ASSERT_EQ(m.shape().second, 0);
}

TEST(matrix, constructor_from_specified_dimensions) {
  const int kNumRows = 3;
  const int kNumCols = 2;
  const int kNumElem = kNumRows * kNumCols;

  matrix<double> m(kNumRows, kNumCols);

  ASSERT_EQ(m.num_rows(), kNumRows);
  ASSERT_EQ(m.num_cols(), kNumCols);
  ASSERT_EQ(m.shape().first, kNumRows);
  ASSERT_EQ(m.shape().second, kNumCols);

  double contents[kNumElem];

  for (int i = 0; i < kNumElem; ++i) {
    contents[i] = m[i];
  }

  EXPECT_THAT(contents, ElementsAre(0, 0, 0, 0, 0, 0));

  for (int i = 0; i < kNumElem; ++i) {
    m[i] = i;
  }

  for (int i = 0; i < kNumElem; ++i) {
    contents[i] = m[i];
  }

  EXPECT_THAT(contents, ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(matrix, constructor_from_initializer_list) {
  // Creates an empty matrix.
  matrix<double> m = {};

  ASSERT_EQ(m.shape().first, 0);
  ASSERT_EQ(m.shape().second, 0);

  // Assigns it to a list. memory allocated.

  m = {1, 2, 3, 4, 5, 6};

  ASSERT_EQ(m.shape().first, 1);
  ASSERT_EQ(m.shape().second, 6);

  double contents[6];
  for (int i = 0; i < 6; ++i) {
    contents[i] = m[i];
  }

  EXPECT_THAT(contents, ElementsAre(1, 2, 3, 4, 5, 6));

  // Assigns it to a smaller list. No memory allocated.

  m = {1, 2};

  ASSERT_EQ(m.shape().first, 1);
  ASSERT_EQ(m.shape().second, 2);

  double new_contents[2];
  new_contents[0] = m[0];
  new_contents[1] = m[1];
  EXPECT_THAT(new_contents, ElementsAre(1, 2));
}

TEST(matrix, constructor_from_nested_initializer_list) {
  matrix<double> m = {{1, 2, 3}, {4, 5, 6}};

  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 3);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 3);

  std::vector<double> contents;

  for (int i = 0; i < m.shape().first; ++i) {
    for (int j = 0; j < m.shape().second; ++j) {
      contents.push_back(m(i, j));
    }
  }

  EXPECT_THAT(contents, ElementsAre(1, 2, 3, 4, 5, 6));

  m = {{1, 2}, {3, 4}};

  ASSERT_EQ(m.num_rows(), 2);
  ASSERT_EQ(m.num_cols(), 2);
  ASSERT_EQ(m.shape().first, 2);
  ASSERT_EQ(m.shape().second, 2);

  contents.clear();

  for (int i = 0; i < m.shape().first; ++i) {
    for (int j = 0; j < m.shape().second; ++j) {
      contents.push_back(m(i, j));
    }
  }

  EXPECT_THAT(contents, ElementsAre(1, 2, 3, 4));
}

}  // namespace insight
