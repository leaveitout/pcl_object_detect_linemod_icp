//
// Created by sean on 30/06/16.
//

#ifndef PCL_OBJECT_DETECT_LINEMOD_COMMON_HPP
#define PCL_OBJECT_DETECT_LINEMOD_COMMON_HPP

#include <pcl/point_cloud.h>

#include <boost/range/irange.hpp>

namespace pcltools {

// User defined literals
// @formatter:off
constexpr size_t operator "" _sz (unsigned long long size) { return size_t{size}; }
constexpr double operator "" _deg (long double deg) { return deg * M_PI / 180.0; }
constexpr double operator "" _deg (unsigned long long deg) { return deg * M_PI / 180.0; }
constexpr double operator "" _cm (long double cm) { return cm / 100.0; }
constexpr double operator "" _cm (unsigned long long cm) { return cm / 100.0; }
constexpr double operator "" _mm (long double mm) { return mm / 1000.0; }
constexpr double operator "" _mm (unsigned long long mm) { return mm / 1000.0; }
// @formatter:on

template <typename T>
constexpr auto izrange (T upper)
-> decltype (boost::irange (static_cast <T> (0), upper)) {
  return boost::irange (static_cast <T> (0), upper);
}

}


#endif //PCL_OBJECT_DETECT_LINEMOD_COMMON_HPP
