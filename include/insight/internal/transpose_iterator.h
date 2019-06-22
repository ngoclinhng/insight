// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_INTERNAL_TRANSPOSE_ITERATOR_H_
#define INCLUDE_INSIGHT_INTERNAL_TRANSPOSE_ITERATOR_H_

#include <iterator>

namespace insight {
namespace internal {

//  c
//  |
//  1  2  3  4  <- i
//  5  6  7  8
//  9  10  11  12 end
//
// forward: 1 5 9 2 6 10 3 7 11 4 8 12
// backward: 12 8 4 ....

// increment():
//  if distance(i, end) > step_length: move forward.
//    move i by step_length.
//  else if distance(i, end) > 1: circular back.
//    c += 1
//    i = c.
//  else:
//    i += 1. --> reach the end.
//
// increment(n):
//  step = (n * step_length_) % number_of_elements
//  if step == 0
//    c += 1
//    i = c.
//  else
//    advance i by step.
template<typename Iter>
class circular_jump_iterator {
 private:
  using iter_traits = std::iterator_traits<Iter>;

 public:
  using iterator_type = Iter;
  using iterator_category = typename iter_traits::iterator_category;
  using value_type = typename iter_traits::value_type;
  using difference_type = typename iter_traits::difference_type;
  using pointer = typename iter_traits::pointer;
  using reference = typename iter_traits::reference;
 private:
  iterator_type it_;
  iterator_type col_;
  difference_type step_length_;
};

}  // namespace internal
}  // namespace insight
#endif  // INCLUDE_INSIGHT_INTERNAL_TRANSPOSE_ITERATOR_H_
