// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_INTERNAL_JUMP_ITERATOR_H_
#define INCLUDE_INSIGHT_INTERNAL_JUMP_ITERATOR_H_

#include <iterator>

namespace insight {
namespace internal {

template<typename Iter>
class jump_iterator {
 private:
  using iter_traits = std::iterator_traits<Iter>;

 public:
  using iterator_type = Iter;
  using iterator_category = typename iter_traits::iterator_category;
  using value_type = typename iter_traits::value_type;
  using difference_type = typename iter_traits::difference_type;
  using pointer = typename iter_traits::pointer;
  using reference = typename iter_traits::reference;

  jump_iterator() : it_(),
                    step_length_(),
                    distance_from_begin_(),
                    distance_to_end_() {}

  jump_iterator(const Iter& it,
                difference_type step_length,
                difference_type distance_from_begin,
                difference_type distance_to_end)
      : it_(it),
        step_length_(step_length),
        distance_from_begin_(distance_from_begin),
        distance_to_end_(distance_to_end) {
  }

  template<typename U>
  jump_iterator(const jump_iterator<U>& src)
      : it_(src.base()),
        step_length_(src.step_length()),
        distance_from_begin_(src.distance_from_begin()),
        distance_to_end_(src.distance_to_end()) {}

  Iter base() const { return it_; }
  difference_type step_length() const { return step_length_; }
  difference_type distance_to_end() const { return distance_to_end_; }
  difference_type distance_from_begin() const { return distance_from_begin_; }

  reference operator*() const { return *it_; }
  pointer  operator->() const { return it_.operator->(); }

  jump_iterator& operator++() {
    increment_();
    return *this;
  }

  jump_iterator  operator++(int) {
    jump_iterator tmp(*this);
    increment_();
    return tmp;
  }

  jump_iterator& operator--() {
    decrement_();
    return *this;
  }

  jump_iterator  operator--(int) {
    jump_iterator tmp(*this);
    decrement_();
    return tmp;
  }

  jump_iterator  operator+ (difference_type n) const {
    jump_iterator tmp(*this);
    tmp += n;
    return tmp;
  }

  jump_iterator& operator+=(difference_type n) {
    increment_(n);
    return *this;
  }

  jump_iterator  operator- (difference_type n) const {
    jump_iterator tmp(*this);
    tmp -= n;
    return tmp;
  }

  jump_iterator& operator-=(difference_type n) {
    decrement_(n);
    return *this;
  }

  reference operator[](difference_type n) const {
    return static_cast<reference>((it_[n * step_length_]));
  }

 private:
  iterator_type it_;
  difference_type step_length_;
  difference_type distance_from_begin_;
  difference_type distance_to_end_;

  void increment_();
  void increment_(difference_type n);
  void decrement_();
  void decrement_(difference_type n);
};

template<typename Iter>
void
jump_iterator<Iter>::increment_() {
  if (distance_to_end_ < step_length_) {
    std::advance(it_, distance_to_end_);
    distance_from_begin_ += distance_to_end_;
    distance_to_end_ = 0;
  } else {
    std::advance(it_, step_length_);
    distance_to_end_ -= step_length_;
    distance_from_begin_ += step_length_;
  }
}

template<typename Iter>
void
jump_iterator<Iter>::increment_(difference_type n) {
  difference_type step = n * step_length_;
  if (step > distance_to_end_) {
    std::advance(it_, distance_to_end_);
    distance_from_begin_ += distance_to_end_;
    distance_to_end_ = 0;
  } else {
    std::advance(it_, step);
    distance_to_end_ -= step;
    distance_from_begin_ += step;
  }
}

template<typename Iter>
void
jump_iterator<Iter>::decrement_() {
  difference_type s = distance_from_begin_ % step_length_;
  difference_type step = (s == 0) ? step_length_ : s;
  if (distance_from_begin_ < step) {
    std::advance(it_, -distance_from_begin_);
    distance_to_end_ += distance_from_begin_;
    distance_from_begin_ = 0;
  } else {
    std::advance(it_, -step);
    distance_from_begin_ -= step;
    distance_to_end_ += step;
  }
}

template<typename Iter>
void
jump_iterator<Iter>::decrement_(difference_type n) {
  difference_type s = distance_from_begin_ % step_length_;
  difference_type step = (s == 0) ? n * step_length_ :
                         (n-1) * step_length_ + s;
  if (distance_from_begin_ < step) {
    std::advance(it_, -distance_from_begin_);
    distance_to_end_ += distance_from_begin_;
    distance_from_begin_ = 0;
  } else {
    std::advance(it_, -step);
    distance_from_begin_ -= step;
    distance_to_end_ += step;
  }
}

template<typename Iter1, typename Iter2>
inline
bool
operator==(const jump_iterator<Iter1>& x,
           const jump_iterator<Iter2>& y) {
  return x.base() == y.base();
}

template<typename Iter1, typename Iter2>
inline
bool
operator!=(const jump_iterator<Iter1>& x,
           const jump_iterator<Iter2>& y) {
  return x.base() != y.base();
}

template<typename Iter1, typename Iter2>
inline
bool
operator<(const jump_iterator<Iter1>& x,
          const jump_iterator<Iter2>& y) {
  return x.base() < y.base();
}

template<typename Iter1, typename Iter2>
inline
bool
operator<=(const jump_iterator<Iter1>& x,
           const jump_iterator<Iter2>& y) {
  return x.base() <= y.base();
}

template<typename Iter1, typename Iter2>
inline
bool
operator>(const jump_iterator<Iter1>& x,
          const jump_iterator<Iter2>& y) {
  return x.base() > y.base();
}

template<typename Iter1, typename Iter2>
inline
bool
operator>=(const jump_iterator<Iter1>& x,
           const jump_iterator<Iter2>& y) {
  return x.base() >= y.base();
}

template<typename Iter1, typename Iter2>
inline
auto
operator-(const jump_iterator<Iter1>& x,
          const jump_iterator<Iter2>& y)
    -> decltype((x.base() - y.base())) {
  return (x.base() - y.base() + x.step_length() - 1) / x.step_length();
}

template<typename Iter>
inline
jump_iterator<Iter>
operator+(typename jump_iterator<Iter>::difference_type n,
          const jump_iterator<Iter>& x) {
  jump_iterator<Iter> tmp(x);
  tmp += n;
  return tmp;
}

template<typename Iter>
inline
jump_iterator<Iter>
make_jump_iterator(const Iter& it,
                   typename
                   jump_iterator<Iter>::difference_type step_length,
                   typename
                   jump_iterator<Iter>::difference_type distance_from_begin,
                   typename
                   jump_iterator<Iter>::difference_type distance_to_end) {
  return jump_iterator<Iter>(it, step_length, distance_from_begin,
                             distance_to_end);
}
}  // namespace internal
}  // namespace insight
#endif  // INCLUDE_INSIGHT_INTERNAL_JUMP_ITERATOR_H_
