// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_INTERNAL_TYPE_TRAITS_H_
#define INCLUDE_INSIGHT_INTERNAL_TYPE_TRAITS_H_

#include <utility>
#include <iterator>
#include <type_traits>

namespace insight {
namespace internal {

namespace type_traits_iterator_details {
// Does type T has member type named iterator_category?
// TODO(Linh): use the C++11 decltype & declval
template <typename T>
struct has_iterator_category {
 private:
  struct two {char c1; char c2;};

  template <typename U> static two test(...);

  template <typename U>
  static char test(typename U::iterator_category* = 0);

 public:
  static const bool value = sizeof(test<T>(0)) == 1;
};

// Is T an iterator type and is it convertible to U?

template <typename T, typename U,
          bool = has_iterator_category<std::iterator_traits<T> >::value>
struct has_iterator_category_convertible_to
    : public std::integral_constant<
  bool,
  std::is_convertible<typename std::iterator_traits<T>::iterator_category,
                      U>::value>
{};

template <class T, class U>
struct has_iterator_category_convertible_to<T, U, false>
    : public std::false_type {};

}  // namespace type_traits_iterator_details

// Is T an input iterator?
template<typename T>
struct is_input_iterator
    : public type_traits_iterator_details::has_iterator_category_convertible_to<T, std::input_iterator_tag> {};  // NOLINT

// Is T a forward iterator?
template<typename T>
struct is_forward_iterator
    : public type_traits_iterator_details::has_iterator_category_convertible_to<T, std::forward_iterator_tag> {};  // NOLINT

// is_nothrow_swappable. This feature is available with C++ version >= 14 but
// since we's stuck with C++11 we have to implement this feature ourself.
//
// This implementation is borrowed from [1].
//
// [1] - https://github.com/boostorg/type_traits/blob/develop/include/boost/
// type_traits/is_nothrow_swappable.hpp

namespace type_traits_swappable_detail {
using std::swap;

template<typename T, typename U,
         bool B = noexcept(swap(std::declval<T>(), std::declval<U>()))>
std::integral_constant<bool, B> is_nothrow_swappable_with_impl(int);

template<typename T, typename U>
std::false_type  is_nothrow_swappable_with_impl(...);

template<typename T, typename U>
struct is_nothrow_swappable_with_helper {
  using type = decltype(is_nothrow_swappable_with_impl<T, U>(0));
};

template<typename T,
         bool B = noexcept(swap(std::declval<T&>(), std::declval<T&>()))>
std::integral_constant<bool, B> is_nothrow_swappable_impl(int);

template<typename T>
std::false_type is_nothrow_swappable_impl(...);

template<typename T>
struct is_nothrow_swappable_helper {
  using type = decltype(is_nothrow_swappable_impl<T>(0));
};

}  // namespace type_traits_swappable_detail

template<typename T, typename U>
struct is_nothrow_swappable_with
    : public type_traits_swappable_detail::is_nothrow_swappable_with_helper<T, U>::type { };  // NOLINT

template<typename T>
struct is_nothrow_swappable
    : public type_traits_swappable_detail::is_nothrow_swappable_helper<T>::type { };  // NOLINT

}  // namespace internal
}  // namespace insight
#endif  // INCLUDE_INSIGHT_INTERNAL_TYPE_TRAITS_H_
