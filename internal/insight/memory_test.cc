// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include <vector>

#include "insight/memory.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {

using ::testing::ElementsAre;

TEST(memory, insight_allocator) {
  std::vector<int, insight_allocator<int>> vec;
  vec.reserve(3);
  vec.push_back(1);
  vec.push_back(2);
  vec.push_back(3);

  EXPECT_THAT(vec, ElementsAre(1, 2, 3));
}

}  // namespace insight
