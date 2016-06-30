//
// Created by sean on 30/06/16.
//

#ifndef PCL_OBJECT_DETECT_LINEMOD_SEGMENTATION_HPP
#define PCL_OBJECT_DETECT_LINEMOD_SEGMENTATION_HPP

#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

#include "pcltools/common.hpp"


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

//extern const unsigned DEFAULT_MIN_CLUSTER_SIZE;
//extern const unsigned DEFAULT_MAX_CLUSTER_SIZE;
//extern const decltype(0_cm) DEFAULT_CLUSTER_TOLERANCE;
//extern const unsigned DEFAULT_MAX_NUM_CLUSTERS;

constexpr auto DEFAULT_MIN_CLUSTER_SIZE = 100U;
constexpr auto DEFAULT_MAX_CLUSTER_SIZE = 2500U;
constexpr auto DEFAULT_CLUSTER_TOLERANCE = 2_cm;
constexpr auto DEFAULT_MAX_NUM_CLUSTERS = 20U;

/**
 * Get the different indices for the different clusters that are present in a point cloud.
 */
template <typename PointType>
auto getClustersIndices (typename pcl::PointCloud <PointType>::ConstPtr const & input_cloud,
                         unsigned min_cluster_size = DEFAULT_MIN_CLUSTER_SIZE,
                         unsigned max_cluster_size = DEFAULT_MAX_CLUSTER_SIZE,
                         double cluster_tolerance = DEFAULT_CLUSTER_TOLERANCE,
                         unsigned max_num_clusters = DEFAULT_MAX_NUM_CLUSTERS)
-> std::vector <pcl::PointIndices> {
  auto tree = boost::make_shared <pcl::search::KdTree <PointType>> ();
  tree->setInputCloud (input_cloud);

  auto clusters = std::vector <pcl::PointIndices> {};
  auto clusterer = pcl::EuclideanClusterExtraction <PointType> {};

  clusterer.setClusterTolerance (cluster_tolerance);
  clusterer.setMinClusterSize (min_cluster_size);
  clusterer.setMaxClusterSize (max_cluster_size);
  clusterer.setSearchMethod (tree);
  clusterer.setInputCloud (input_cloud);
  clusterer.extract (clusters);

  auto larger_cluster_lambda = [] (pcl::PointIndices a, pcl::PointIndices b) {
    return a.indices.size () >= b.indices.size ();
  };
  std::sort (clusters.begin (), clusters.end (), larger_cluster_lambda);

  if (clusters.size () > max_num_clusters)
    clusters.erase (clusters.begin () + max_num_clusters, clusters.end ());

  return clusters;
}

}
}
#endif //PCL_OBJECT_DETECT_LINEMOD_SEGMENTATION_HPP
