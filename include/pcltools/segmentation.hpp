//
// Created by sean on 30/06/16.
//

#ifndef PCL_OBJECT_DETECT_LINEMOD_SEGMENTATION_HPP
#define PCL_OBJECT_DETECT_LINEMOD_SEGMENTATION_HPP

#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/extract_indices.h>


namespace pcltools {

namespace seg {

/**
 * Extract a point cloud for the indices passed.
 *
 * @param cloud
 * The cloud to extract indices from.
 * @param indices
 * The indices to use for the extraction.
 * @param keep_organised
 * If passed cloud is organised, requests result cloud to be organised.
 * @param negative
 * The complement of the indices set will be used for extraction.
 * @return
 * The point cloud for the indices of the passed cloud.
 */
template <typename PointType>
auto extractIndices (typename pcl::PointCloud <PointType>::ConstPtr const & cloud,
                     pcl::PointIndicesPtr indices,
                     bool keep_organised = false,
                     bool negative = false)
-> typename pcl::PointCloud <PointType>::Ptr {
  auto extract = pcl::ExtractIndices <PointType>{};
  extract.setInputCloud (cloud);
  extract.setIndices (indices);
  if (cloud->isOrganized ())
    extract.setKeepOrganized (keep_organised);
  extract.setNegative (negative);
  auto extracted_cloud = boost::make_shared <pcl::PointCloud <PointType>> ();
  extract.filter (*extracted_cloud);
  return extracted_cloud;
}
}
}
#endif //PCL_OBJECT_DETECT_LINEMOD_SEGMENTATION_HPP
