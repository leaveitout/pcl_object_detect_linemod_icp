//
// Created by sean on 30/06/16.
//

#ifndef PCL_OBJECT_DETECT_LINEMOD_SEGMENTATION_HPP
#define PCL_OBJECT_DETECT_LINEMOD_SEGMENTATION_HPP

#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>

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
                     pcl::PointIndices indices,
                     bool keep_organised = false,
                     bool negative = false)
-> typename pcl::PointCloud <PointType>::Ptr {
  auto extract = pcl::ExtractIndices <PointType>{};
  extract.setInputCloud (cloud);
  auto indices_ptr = boost::make_shared <pcl::PointIndices> (indices);
  extract.setIndices (indices_ptr);
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
constexpr auto DEFAULT_MAX_CLUSTER_SIZE = std::numeric_limits <unsigned int>::max ();
constexpr auto DEFAULT_CLUSTER_TOLERANCE = 2_cm;
constexpr auto DEFAULT_MAX_NUM_CLUSTERS = std::numeric_limits <unsigned int>::max ();


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


constexpr auto PLANE_THRESHOLD_DEFAULT = 2_cm;
constexpr auto MAX_SAC_ITERATIONS_DEFAULT = 1000;
constexpr auto SAC_SIGMA_DEFAULT = 2.0;
constexpr auto REFINE_ITERATIONS_DEFAULT = 50;


template <typename PointType>
auto getPlaneIndices (typename pcl::PointCloud <PointType>::ConstPtr const & cloud,
                      double plane_thold = PLANE_THRESHOLD_DEFAULT,
                      int max_iterations = MAX_SAC_ITERATIONS_DEFAULT,
                      double refine_sigma = SAC_SIGMA_DEFAULT,
                      int refine_iterations = REFINE_ITERATIONS_DEFAULT)
-> pcl::PointIndicesPtr {
  auto model = boost::make_shared <pcl::SampleConsensusModelPlane <PointType>> (cloud);
  auto sac = pcl::RandomSampleConsensus <PointType>{model, plane_thold};
  sac.setMaxIterations (max_iterations);
  auto inliers_indices_ptr = boost::make_shared <pcl::PointIndices> ();
  auto result = sac.computeModel ();
  sac.getInliers (inliers_indices_ptr->indices);

  if (!result || inliers_indices_ptr->indices.empty ()) {
    pcl::console::print_error ("No planar model found, relax thresholds and continue.");
    return nullptr;
  }

  sac.refineModel (refine_sigma, refine_iterations);

  sac.getSampleConsensusModel ();
  sac.getInliers (inliers_indices_ptr->indices);

  return inliers_indices_ptr;
}


template <typename PointType>
auto getTable (typename pcl::PointCloud <PointType>::ConstPtr const & cloud,
               double plane_thold = PLANE_THRESHOLD_DEFAULT,
               int max_iterations = MAX_SAC_ITERATIONS_DEFAULT,
               double refine_sigma = SAC_SIGMA_DEFAULT,
               int refine_iterations = REFINE_ITERATIONS_DEFAULT,
               bool keep_organised = true)
-> typename pcl::PointCloud <PointType>::Ptr {
  auto point_indices = getPlaneIndices <PointType> (cloud);

  if (point_indices->indices.size () == 0) {
    pcl::console::print_highlight ("No plane to be found in input pcd.\n");
    return nullptr;
  }

  auto extracted_plane = extractIndices <PointType> (cloud, point_indices, keep_organised);

  auto largest_cluster_indices = getClustersIndices <PointType> (extracted_plane).at (0);

  return extractIndices <PointType> (extracted_plane, largest_cluster_indices, keep_organised);
}


/**
 * This function removes the table from cloud and returns the table as a cloud
 */
template <typename PointType>
auto removeTable (typename pcl::PointCloud <PointType>::Ptr & cloud,
                  double plane_thold = PLANE_THRESHOLD_DEFAULT,
                  int max_iterations = MAX_SAC_ITERATIONS_DEFAULT,
                  double refine_sigma = SAC_SIGMA_DEFAULT,
                  int refine_iterations = REFINE_ITERATIONS_DEFAULT)
-> pcl::PointIndicesPtr {
  auto point_indices = getPlaneIndices <PointType> (cloud);

  if (point_indices->indices.size () == 0) {
    pcl::console::print_highlight ("No plane to be found in input cloud.\n");
    return nullptr;
  }

  auto extracted_plane = extractIndices <PointType> (cloud, point_indices);

  auto largest_cluster_indices = getClustersIndices <PointType> (extracted_plane).at (0);

  cloud = extractIndices <PointType> (cloud, point_indices, true);

  return boost::make_shared <pcl::PointIndices> (largest_cluster_indices);
}


template <typename PointType>
auto getTableConvexHull (typename pcl::PointCloud <PointType>::ConstPtr const & table_cloud)
-> typename pcl::PointCloud <PointType>::Ptr {
  auto hull = pcl::ConvexHull <PointType>{};
  auto convex_hull_cloud = boost::make_shared <pcl::PointCloud <PointType>> ();

  hull.setInputCloud (table_cloud);
  hull.reconstruct (*convex_hull_cloud);

  if (hull.getDimension () != 2 || convex_hull_cloud->size () == 0) {

    pcl::console::print_error ("The input cloud does not represent a planar surface for hull.\n");
    return nullptr;

  } else {
    return convex_hull_cloud;
  }
}


constexpr auto MIN_HEIGHT_FROM_HULL_DEFAULT = 0.5_cm;
constexpr auto MAX_HEIGHT_FROM_HULL_DEFAULT = 70_cm;


template <typename PointType>
auto getPointsAboveConvexHull (typename pcl::PointCloud <PointType>::Ptr cloud,
                               typename pcl::PointCloud <PointType>::Ptr convex_hull_cloud,
                               double min_height = MIN_HEIGHT_FROM_HULL_DEFAULT,
                               double max_height = MAX_HEIGHT_FROM_HULL_DEFAULT,
                               bool keep_organised = true)
-> typename pcl::PointCloud <PointType>::Ptr {
  if (convex_hull_cloud) {
    if (convex_hull_cloud->size () != 0) {
      auto prism = pcl::ExtractPolygonalPrismData <PointType> {};
      prism.setInputCloud (cloud);
      prism.setInputPlanarHull (convex_hull_cloud);
      prism.setHeightLimits (min_height, max_height);
      auto indices_ptr = boost::make_shared <pcl::PointIndices> ();
      prism.segment (*indices_ptr);
      return extractIndices <PointType> (cloud, indices_ptr, keep_organised);
    }
    else {
      pcl::console::print_error ("The input cloud does not represent a planar surface for hull.\n");
      return nullptr;
    }
  }
}
}
}
#endif //PCL_OBJECT_DETECT_LINEMOD_SEGMENTATION_HPP
