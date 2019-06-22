// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#include <vector>
#include <algorithm>

#include "insight/internal/jump_iterator.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace insight {
namespace internal {

TEST(jump_iterator, increment_by_1) {
  std::vector<int> x = {5, 7, 4, 2, 8, 6, 1, 9, 0, 3};

  // jump by 1 -> normal, contiguous iterator.
  auto b1 = make_jump_iterator(x.begin(), 1, 0, x.size());
  auto b1_end = make_jump_iterator(x.end(), 1, x.size(), 0);

  EXPECT_EQ(*b1, 5);
  ++b1;
  EXPECT_EQ(*b1, 7);
  ++b1;
  EXPECT_EQ(*b1, 4);
  ++b1;
  EXPECT_EQ(*b1, 2);
  ++b1;
  EXPECT_EQ(*b1, 8);
  ++b1;
  EXPECT_EQ(*b1, 6);
  ++b1;
  EXPECT_EQ(*b1, 1);
  ++b1;
  EXPECT_EQ(*b1, 9);
  ++b1;
  EXPECT_EQ(*b1, 0);
  ++b1;
  EXPECT_EQ(*b1, 3);
  ++b1;
  EXPECT_TRUE(b1 == b1_end);
  ++b1;
  ++b1;
  ++b1;
  EXPECT_TRUE(b1 == b1_end);

  // jump by 2: {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}
  auto b2 = make_jump_iterator(x.begin(), 2, 0, x.size());
  auto b2_end = make_jump_iterator(x.end(), 2, x.size(), 0);

  EXPECT_EQ(*b2, 5);
  ++b2;
  EXPECT_EQ(*b2, 4);
  ++b2;
  EXPECT_EQ(*b2, 8);
  ++b2;
  EXPECT_EQ(*b2, 1);
  ++b2;
  EXPECT_EQ(*b2, 0);
  ++b2;
  EXPECT_TRUE(b2 == b2_end);
  ++b2;
  ++b2;
  EXPECT_TRUE(b2 == b2_end);

  // jump by 3: {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}
  auto b3 = make_jump_iterator(x.begin(), 3, 0, x.size());
  auto b3_end = make_jump_iterator(x.end(), 3, x.size(), 0);

  EXPECT_EQ(*b3, 5);
  ++b3;
  EXPECT_EQ(*b3, 2);
  ++b3;
  EXPECT_EQ(*b3, 1);
  ++b3;
  EXPECT_EQ(*b3, 3);
  ++b3;
  EXPECT_TRUE(b3 == b3_end);
  ++b3;
  ++b3;
  EXPECT_TRUE(b3 == b3_end);

  // jump by 4: {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}
  auto b4 = make_jump_iterator(x.begin(), 4, 0, x.size());
  auto b4_end = make_jump_iterator(x.end(), 4, x.size(), 0);

  EXPECT_EQ(*b4, 5);
  ++b4;
  EXPECT_EQ(*b4, 8);
  ++b4;
  EXPECT_EQ(*b4, 0);
  ++b4;
  EXPECT_TRUE(b4 == b4_end);
  ++b4;
  ++b4;
  EXPECT_TRUE(b4 == b4_end);
}

TEST(jump_iterator, decrement_by_one) {
  std::vector<int> x = {5, 7, 4, 2, 8, 6, 1, 9, 0, 3};

  // jump by 1 -> normal, contiguous iterator.
  auto b1 = make_jump_iterator(x.begin(), 1, 0, x.size());
  auto b1_end = make_jump_iterator(x.end(), 1, x.size(), 0);

  --b1_end;
  EXPECT_EQ(*b1_end, 3);
  --b1_end;
  EXPECT_EQ(*b1_end, 0);
  --b1_end;
  EXPECT_EQ(*b1_end, 9);
  --b1_end;
  EXPECT_EQ(*b1_end, 1);
  --b1_end;
  EXPECT_EQ(*b1_end, 6);
  --b1_end;
  EXPECT_EQ(*b1_end, 8);
  --b1_end;
  EXPECT_EQ(*b1_end, 2);
  --b1_end;
  EXPECT_EQ(*b1_end, 4);
  --b1_end;
  EXPECT_EQ(*b1_end, 7);
  --b1_end;
  EXPECT_EQ(*b1_end, 5);
  --b1_end;
  EXPECT_TRUE(b1_end == b1);
  --b1_end;
  --b1_end;
  --b1_end;
  EXPECT_TRUE(b1_end == b1);

  // jump by 2: {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}
  auto b2 = make_jump_iterator(x.begin(), 2, 0, x.size());
  auto b2_end = make_jump_iterator(x.end(), 2, x.size(), 0);

  --b2_end;
  EXPECT_EQ(*b2_end, 0);
  --b2_end;
  EXPECT_EQ(*b2_end, 1);
  --b2_end;
  EXPECT_EQ(*b2_end, 8);
  --b2_end;
  EXPECT_EQ(*b2_end, 4);
  --b2_end;
  EXPECT_EQ(*b2_end, 5);
  --b2_end;
  EXPECT_TRUE(b2_end == b2);
  --b2_end;
  --b2_end;
  EXPECT_TRUE(b2_end == b2);

  // jump by 3: {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}
  auto b3 = make_jump_iterator(x.begin(), 3, 0, x.size());
  auto b3_end = make_jump_iterator(x.end(), 3, x.size(), 0);

  --b3_end;
  EXPECT_EQ(*b3_end, 3);
  --b3_end;
  EXPECT_EQ(*b3_end, 1);
  --b3_end;
  EXPECT_EQ(*b3_end, 2);
  --b3_end;
  EXPECT_EQ(*b3_end, 5);
  --b3_end;
  EXPECT_TRUE(b3_end == b3);
  --b3_end;
  --b3_end;
  EXPECT_TRUE(b3_end == b3);

  // jump by 4: {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}
  auto b4 = make_jump_iterator(x.begin(), 4, 0, x.size());
  auto b4_end = make_jump_iterator(x.end(), 4, x.size(), 0);

  --b4_end;
  EXPECT_EQ(*b4_end, 0);
  --b4_end;
  EXPECT_EQ(*b4_end, 8);
  --b4_end;
  EXPECT_EQ(*b4_end, 5);
  --b4;
  EXPECT_TRUE(b4_end == b4);
  --b4_end;
  --b4_end;
  EXPECT_TRUE(b4_end == b4);
}

TEST(jump_iterator, increment_by_n) {
  std::vector<int> x = {5, 7, 4, 2, 8, 6, 1, 9, 0, 3};

  // jump by 1 -> normal, contiguous iterator.
  auto b1 = make_jump_iterator(x.begin(), 1, 0, x.size());
  auto b1_end = make_jump_iterator(x.end(), 1, x.size(), 0);
  EXPECT_EQ(std::distance(b1, b1_end), x.size());

  b1 += 3;
  EXPECT_EQ(*b1, 2);
  b1 += 5;
  EXPECT_EQ(*b1, 0);
  b1 += 100;
  EXPECT_TRUE(b1 == b1_end);

  // jump by 2: {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}
  //             |     |     |     |     |
  auto b2 = make_jump_iterator(x.begin(), 2, 0, x.size());
  auto b2_end = make_jump_iterator(x.end(), 2, x.size(), 0);
  EXPECT_EQ(std::distance(b2, b2_end), 5);

  b2 += 2;
  EXPECT_EQ(*b2, 8);
  b2 += 2;
  EXPECT_EQ(*b2, 0);
  b2 += 1;
  EXPECT_TRUE(b2 == b2_end);
  b2 += 100;
  EXPECT_TRUE(b2 == b2_end);

  // jump by 3: {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}
  //             |        |        |        |
  auto b3 = make_jump_iterator(x.begin(), 3, 0, x.size());
  auto b3_end = make_jump_iterator(x.end(), 3, x.size(), 0);
  EXPECT_EQ(std::distance(b3, b3_end), 4);

  b3 += 1;
  EXPECT_EQ(*b3, 2);
  b3 += 2;
  EXPECT_EQ(*b3, 3);
  b3 += 1;
  EXPECT_TRUE(b3 == b3_end);
  b3 += 10;
  EXPECT_TRUE(b3 == b3_end);

  // jump by 4: {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}
  //             |           |           |
  auto b4 = make_jump_iterator(x.begin(), 4, 0, x.size());
  auto b4_end = make_jump_iterator(x.end(), 4, x.size(), 0);
  EXPECT_EQ(std::distance(b4, b4_end), 3);

  b4 += 2;
  EXPECT_EQ(*b4, 0);
  b4 += 1;
  EXPECT_TRUE(b4 == b4_end);
  b4 += 10;
  EXPECT_TRUE(b4 == b4_end);
}

TEST(jump_iterator, decrement_by_n) {
  std::vector<int> x = {5, 7, 4, 2, 8, 6, 1, 9, 0, 3};

  // jump by 1 -> normal, contiguous iterator.
  auto b1 = make_jump_iterator(x.begin(), 1, 0, x.size());
  auto b1_end = make_jump_iterator(x.end(), 1, x.size(), 0);

  b1_end -= 3;
  EXPECT_EQ(*b1_end, 9);
  b1_end -= 5;
  EXPECT_EQ(*b1_end, 4);
  b1_end -= 2;
  EXPECT_EQ(*b1, 5);
  b1_end -= 1;
  EXPECT_TRUE(b1_end == b1);
  b1_end -= 10;
  EXPECT_TRUE(b1_end == b1);

  // jump by 2: {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}
  //             |     |     |     |     |
  auto b2 = make_jump_iterator(x.begin(), 2, 0, x.size());
  auto b2_end = make_jump_iterator(x.end(), 2, x.size(), 0);

  b2_end -= 1;
  EXPECT_EQ(*b2_end, 0);
  b2_end -= 3;
  EXPECT_EQ(*b2_end, 4);
  b2_end -= 1;
  EXPECT_EQ(*b2_end, 5);
  b2_end -= 1;
  EXPECT_TRUE(b2_end == b2);
  b2_end -= 10;
  EXPECT_TRUE(b2_end == b2);

  // jump by 3: {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}
  //             |        |        |        |
  auto b3 = make_jump_iterator(x.begin(), 3, 0, x.size());
  auto b3_end = make_jump_iterator(x.end(), 3, x.size(), 0);

  b3_end -= 2;
  EXPECT_EQ(*b3_end, 1);
  b3_end -= 2;
  EXPECT_EQ(*b3_end, 5);
  b3_end -= 2;
  EXPECT_TRUE(b3_end == b3);
  b3_end -= 10;
  EXPECT_TRUE(b3_end == b3);

  // jump by 4: {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}
  //             |           |           |
  auto b4 = make_jump_iterator(x.begin(), 4, 0, x.size());
  auto b4_end = make_jump_iterator(x.end(), 4, x.size(), 0);

  b4_end -= 1;
  EXPECT_EQ(*b4_end, 0);
  b4_end -= 1;
  EXPECT_EQ(*b4_end, 8);
  b4_end -= 1;
  EXPECT_EQ(*b4_end, 5);
  b4_end -= 1;
  EXPECT_TRUE(b4_end == b4);
  b4_end -= 10;
  EXPECT_TRUE(b4_end == b4);
}
}  // namespace internal
}  // namespace insight
