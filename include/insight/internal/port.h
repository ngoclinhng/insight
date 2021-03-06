// Copyright (C) 2019
//
// Author: mail2ngoclinh@gmail.com (Ngoc Linh)

#ifndef INCLUDE_INSIGHT_INTERNAL_PORT_H_
#define INCLUDE_INSIGHT_INTERNAL_PORT_H_

#include "insight/internal/config.h"

#if defined(_MSC_VER) && defined(SAPIEN_BUILDING_SHARED_LIBRARY)
# define INSIGHT_EXPORT __declspec(dllexport)
#elif defined(_MSC_VER) && defined(SAPIEN_USING_SHARED_LIBRARY)
# define INSIGHT_EXPORT __declspec(dllimport)
#else
# define INSIGHT_EXPORT
#endif

#define INSIGHT_NOEXCEPT noexcept
#define INSIGHT_NOEXCEPT_IF(x) noexcept(x)
#define NO_RETURN [[noreturn]]

#endif  // INCLUDE_INSIGHT_INTERNAL_PORT_H_
