#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/point_cloud_image_extractors.h>
#include <pcl/recognition/linemod.h>
#include <pcl/recognition/color_gradient_modality.h>
#include <pcl/recognition/surface_normal_modality.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>

#include <boost/range/irange.hpp>
#include <boost/range/counting_range.hpp>
#include <boost/smart_ptr/make_shared.hpp>


#include "pcltools/segmentation.hpp"


// Typedefs
using PointType = pcl::PointXYZRGBA;
using Cloud = pcl::PointCloud <PointType>;
using PointNormalType = pcl::PointXYZRGBNormal;


namespace fs = boost::filesystem;


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


// Constants
constexpr auto MIN_VALID_ARGS = 5U;
constexpr auto MAX_VALID_ARGS = 5U;
constexpr auto NUM_PCD_FILES_EXPECTED = 3U;
constexpr auto NUM_LMT_FILES_EXPECTED = 1U;
//constexpr auto NUM_PCD_DIRS_EXPECTED = 2U;
//constexpr auto INPUT_DIR_ARG_POS = 1U;
//constexpr auto OUTPUT_DIR_ARG_POS = 2U;
constexpr auto DEFAULT_MIN_CLUSTER_SIZE = 100U;
constexpr auto DEFAULT_MAX_CLUSTER_SIZE = 25000U;
constexpr auto DEFAULT_TOLERANCE = 2_cm;
constexpr auto DEFAULT_MAX_NUM_CLUSTERS = 20U;


template <typename T>
constexpr auto izrange (T upper)
-> decltype (boost::irange (static_cast<T> (0), upper)) {
  return boost::irange (static_cast <T> (0), upper);
}


auto printHelp (int argc, char ** argv)
-> void {
  using pcl::console::print_error;
  using pcl::console::print_info;

  print_error ("Syntax is: %s <merged-pcd-path> <center-view-pcd-path> <object-pcd-path>"
                   "<object-lmt-path> <options> | -h | --help\n", argv[0]);
  print_info ("%s -h | --help : shows this help\n", argv[0]);
  print_info ("<merger-pcd-path> : This cloud should be the merged cloud from multiple cameras. "
                  "The dominant plane should have been removed using the pcl_remove_table_tool"
                  "- https://github.com/leaveitout/pcl_remove_table.git\n");
  print_info ("<center-view-pcd-path> : This cloud should be the centre cameras view "
                  "of the combined cloud.\n");
  print_info ("<object-pcd-path> : This cloud should represent the object to be detected.\n");
  print_info ("<object-lmt-path> : This is the linemod template file that should be generated "
                  "using the linemod training tool - "
                  "https://github.com/leaveitout/pcl_train_linemod.git\n");

  // TODO: Update this help for any options
  //  print_info ("-min X : use a minimum of X points per cluster (default: 100)\n");
  //  print_info ("-max X : use a maximum of X points per cluster (default: 25000)\n");
  //  print_info ("-tol X : the spatial distance (in meters) between clusters (default: 0.002.\n");
}


auto expandTilde (std::string path_string) -> fs::path {
  if (path_string.at (0) == '~')
    path_string.replace (0, 1, getenv ("HOME"));
  return fs::path{path_string};
}


auto checkValidFile (fs::path const & filepath)
-> bool {
  // Check that the file is valid
  return fs::exists (filepath) && fs::is_regular_file (filepath);
}


auto addNormals (pcl::PointCloud <PointType>::ConstPtr const & cloud)
-> pcl::PointCloud <PointNormalType>::Ptr {
  auto normal_cloud = boost::make_shared <pcl::PointCloud <pcl::Normal>> ();
  auto cloud_xyz = boost::make_shared <pcl::PointCloud <pcl::PointXYZ>> ();
  pcl::copyPointCloud (*cloud, *cloud_xyz);

  auto search_tree = boost::make_shared <pcl::search::KdTree <PointType>> ();
  search_tree->setInputCloud (cloud);

  auto normal_estimator = pcl::NormalEstimation <PointType, pcl::Normal> {};
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setSearchMethod (search_tree);
  normal_estimator.setKSearch (30);
  normal_estimator.compute (*normal_cloud);

  auto result_cloud = boost::make_shared <pcl::PointCloud <PointNormalType>> ();
  pcl::concatenateFields (*cloud_xyz, *normal_cloud, *result_cloud);
  return result_cloud;
}


auto detectTemplates (Cloud::ConstPtr const & cloud, pcl::LINEMOD & linemod)
-> std::vector <pcl::LINEMODDetection> {
  auto color_grad_mod = pcl::ColorGradientModality <PointType> {};
  color_grad_mod.setInputCloud (cloud);
  color_grad_mod.processInputData ();

  auto surface_grad_mod = pcl::SurfaceNormalModality <PointType> {};
  surface_grad_mod.setInputCloud (cloud);
  surface_grad_mod.processInputData ();

  auto modalities = std::vector <pcl::QuantizableModality *> (2);
  modalities[0] = &color_grad_mod;
  modalities[1] = &surface_grad_mod;

  auto detections = std::vector <pcl::LINEMODDetection> {};
  linemod.matchTemplates (modalities, detections);

  return detections;
}


auto outputTemplateMatch (pcl::LINEMODDetection const & detection) -> void {
  std::cout << "x (" << detection.x << ") " <<
      "y (" << detection.y << ") " <<
      "id (" << detection.template_id << ") " <<
      "scale (" << detection.scale << ") " <<
      "score (" << detection.score << ")." << std::endl;
}


auto outputTemplateMatches (std::vector <pcl::LINEMODDetection> detections) -> void {
  auto index = 0UL;
  for (auto const & d : detections) {
    std::cout << "Detection " << index << " ";
    outputTemplateMatch (d);
    ++index;
  }
}


auto getClosestMatchedTemplate (std::vector <pcl::LINEMODDetection> detections)
-> pcl::LINEMODDetection {
  auto score_compare = [] (pcl::LINEMODDetection const & a, pcl::LINEMODDetection const & b) {
    return a.score < b.score;
  };
  return *(std::max_element (detections.begin (), detections.end (), score_compare));
}


/**
 * Get center of mass of the region in the cloud bounded by the passed region
 */
auto getCentroidOrganisedRect (pcl::PointCloud <PointType>::ConstPtr const & cloud,
                               unsigned int x1,
                               unsigned int y1,
                               unsigned int x2,
                               unsigned int y2)
-> Eigen::Vector4d {

  if (!cloud->isOrganized ())
    return Eigen::Vector4d {};

  auto indices = std::vector <int> {};

  for (auto row = y1; row < y2; ++row)
    for (auto col = x1; col < x2; ++col) {
      indices.emplace_back (row * cloud->width + col);
    }

  auto centroid = Eigen::Vector4d {};
  auto num_points_used = pcl::compute3DCentroid (*cloud, indices, centroid);

  if (num_points_used == 0)
    return Eigen::Vector4d {};
  else
    return centroid;
}


/**
 * Get the different indices for the different clusters that are present in a point cloud.
 */
auto getClustersIndices (pcl::PointCloud <PointType>::ConstPtr const & input_cloud,
                         unsigned min_cluster_size = DEFAULT_MIN_CLUSTER_SIZE,
                         unsigned max_cluster_size = DEFAULT_MAX_CLUSTER_SIZE,
                         double cluster_tolerance = DEFAULT_TOLERANCE,
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


auto pairAlign (pcl::PointCloud <PointType>::Ptr const & source_cloud,
                pcl::PointCloud <PointType>::Ptr const & target_cloud,
                bool downsample = false)
-> Eigen::Matrix4f {
  //    -> std::pair<pcl::PointCloud<PointType>::Ptr, Eigen::Matrix4f> {
  auto output = boost::make_shared <pcl::PointCloud <PointType>> ();
  auto final_transform = Eigen::Matrix4f {};

  // Downsample for consistency and speed, enable this for large datasets
  auto source = boost::make_shared <Cloud> ();
  auto target = boost::make_shared <Cloud> ();
  auto grid = pcl::VoxelGrid <PointType>{};
  if (downsample) {
    grid.setLeafSize (0.01, 0.01, 0.01);
    grid.setInputCloud (source_cloud);
    grid.filter (*source);

    grid.setInputCloud (target_cloud);
    grid.filter (*target);
  }
  else {
    source = source_cloud;
    target = target_cloud;
  }

  std::cout << source->size () << std::endl;
  std::cout << target->size () << std::endl;


  // Compute surface normals and curvature
  auto points_with_normals_src = addNormals (source);
  auto points_with_normals_tgt = addNormals (target);

  // Align
  auto reg = pcl::IterativeClosestPointWithNormals <PointNormalType, PointNormalType> {};
  reg.setTransformationEpsilon (1e-6);
  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
  // Note: adjust this based on the size of your datasets
  //  reg.setMaxCorrespondenceDistance (0.10);
  reg.setMaxCorrespondenceDistance (0.01);

  reg.setInputSource (points_with_normals_src);
  reg.setInputTarget (points_with_normals_tgt);

  // Run the same optimization in a loop and visualize the results
  auto prev = Eigen::Matrix4f {};
  auto Ti = Eigen::Matrix4f {Eigen::Matrix4f::Identity ()};
  //  auto prev = Eigen::Matrix4f::Identity ();
  //  auto targetToSource = Eigen::Matrix4f::Identity ();
  auto reg_result = points_with_normals_src;
  reg.setMaximumIterations (2);
  for (int i = 0; i < 40; ++i) {
    PCL_INFO ("Iteration Nr. %d.\n", i);

    // save cloud for visualization purpose
    points_with_normals_src = reg_result;

    // Estimate
    reg.setInputSource (points_with_normals_src);
    reg.align (*reg_result);

    //accumulate transformation between each Iteration
    Ti = static_cast<Eigen::Matrix4f>(reg.getFinalTransformation ()) * Ti;

    //if the difference between this transformation and the previous one
    //is smaller than the threshold, refine the process by reducing
    //the maximal correspondence distance
    auto incremental_diff = fabs ((reg.getLastIncrementalTransformation () - prev).sum ());
    if (incremental_diff < reg.getTransformationEpsilon ()) {
      reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () * 0.95);
    }

    prev = reg.getLastIncrementalTransformation ();
  }

  //add the source to the transformed target
  *output += *source_cloud;

  final_transform = Ti;

  return final_transform;
  //  return std::make_pair (output, final_transform);
}


auto getCentroidTransform (pcl::PointCloud <PointType>::ConstPtr const & cloud,
                           Eigen::Vector4d const & point)
-> Eigen::Matrix4f {
  auto object_centroid = Eigen::Vector4d {};
  auto result = pcl::compute3DCentroid (*cloud, object_centroid);
  auto diff = Eigen::Vector4d {point - object_centroid};

  if (result == 0) {
    std::cout << "Unable to calculate centroid transform." << std::endl;
    return Eigen::Matrix4f {Eigen::Matrix4f::Identity ()};
  }

  auto centroid_transform = Eigen::Matrix4f {Eigen::Matrix4f::Identity ()};

  for (auto row = 0U; row < centroid_transform.rows () - 1; ++row) {
    centroid_transform (row, centroid_transform.cols () - 1) = diff (row);
  }

  std::cout << "Centroid transform : \n" << centroid_transform << std::endl;

  return centroid_transform;
}


/**
 * Find the closest cluster in a cloud to a specific point and return the extracted cluster.
 */
auto getClosestCluster (pcl::PointCloud <PointType>::ConstPtr const & cloud,
                        Eigen::Vector4d const & point)
-> pcl::PointCloud <PointType>::Ptr {
  auto clusters = getClustersIndices (cloud,
                                      DEFAULT_MIN_CLUSTER_SIZE,
                                      DEFAULT_MAX_CLUSTER_SIZE,
                                      DEFAULT_TOLERANCE,
                                      5);
  auto cluster_centroids = std::vector <Eigen::Vector4d,
                                        Eigen::aligned_allocator <Eigen::Vector4d>> {};

  // Compute the centroids for each cluster
  for (auto const & cluster : clusters) {
    auto current_cluster_centroid = Eigen::Vector4d {};
    pcl::compute3DCentroid (*cloud, cluster, current_cluster_centroid);
    cluster_centroids.push_back (current_cluster_centroid);
    std::cout << "Centroid :\n" << current_cluster_centroid << std::endl;
  }

  auto min_dist = [&point] (Eigen::Vector4d const & left, Eigen::Vector4d const & right) {
    return (left - point).norm () < (right - point).norm ();
  };
  auto min_iter = std::min_element (cluster_centroids.begin (), cluster_centroids.end (), min_dist);
  auto min_idx = min_iter - cluster_centroids.begin ();

  auto indices_ptr = boost::make_shared <pcl::PointIndices> (clusters.at (min_idx));
  auto object_cluster_cloud = pcltools::seg::extractIndices <PointType> (cloud, indices_ptr);

  return object_cluster_cloud;
}


auto main (int argc, char * argv[])
-> int {
  pcl::console::print_highlight ("Tool to detect pose of object in a point cloud.\n");

  auto help_flag_1 = pcl::console::find_switch (argc, argv, "-h");
  auto help_flag_2 = pcl::console::find_switch (argc, argv, "--help");

  if (help_flag_1 || help_flag_2) {
    printHelp (argc, argv);
    return -1;
  }

  if (argc > MAX_VALID_ARGS || argc < MIN_VALID_ARGS) {
    pcl::console::print_error ("Invalid number of arguments.\n");
    printHelp (argc, argv);
    return -1;
  }

  // Check if we are working with individual files
  auto const lmt_arg_indices = pcl::console::parse_file_extension_argument (argc, argv, ".lmt");
  auto const pcd_arg_indices = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

  if (lmt_arg_indices.size () != NUM_LMT_FILES_EXPECTED ||
      pcd_arg_indices.size () != NUM_PCD_FILES_EXPECTED) {
    pcl::console::print_error ("Invalid number of arguments.\n");
    printHelp (argc, argv);
    return -1;
  }

  auto lmt_file = fs::path {argv[lmt_arg_indices.at (0)]};
  auto merged_pcd_file = fs::path {argv[pcd_arg_indices.at (0)]};
  auto center_pcd_file = fs::path {argv[pcd_arg_indices.at (1)]};
  auto object_pcd_file = fs::path {argv[pcd_arg_indices.at (2)]};

  if (!checkValidFile (lmt_file)) {
    pcl::console::print_error ("A valid lmt file was not specified.\n");
    printHelp (argc, argv);
    return -1;
  }

  auto valid_merged = checkValidFile (merged_pcd_file);
  auto valid_center = checkValidFile (center_pcd_file);
  auto valid_object = checkValidFile (object_pcd_file);
  auto all_valid_pcd_files = valid_merged && valid_center && valid_object;
  if (!all_valid_pcd_files) {
    pcl::console::print_error ("Not all pcd files specified were valid.\n");
    printHelp (argc, argv);
    return -1;
  }

  auto linemod = pcl::LINEMOD {};
  linemod.loadTemplates (lmt_file.c_str ());

  if (linemod.getNumOfTemplates () == 0) {
    pcl::console::print_error ("No valid templates found in lmt file.\n");
    printHelp (argc, argv);
    return -1;
  }

  // Load input pcds
  auto center_cloud = boost::make_shared <Cloud> ();
  if (pcl::io::loadPCDFile <PointType> (center_pcd_file.c_str (), *center_cloud) == -1) {
    pcl::console::print_error ("Failed to load: %s\n", center_pcd_file.c_str ());
    printHelp (argc, argv);
    return -1;
  }

  if (!center_cloud->isOrganized ()) {
    pcl::console::print_error ("Center camera cloud is not organised");
    printHelp (argc, argv);
    return -1;
  }

  auto merged_cloud = boost::make_shared <Cloud> ();
  if (pcl::io::loadPCDFile <PointType> (merged_pcd_file.c_str (), *merged_cloud) == -1) {
    pcl::console::print_error ("Failed to load: %s\n", merged_pcd_file.c_str ());
    printHelp (argc, argv);
    return -1;
  }

  auto object_cloud = boost::make_shared <Cloud> ();
  if (pcl::io::loadPCDFile <PointType> (object_pcd_file.c_str (), *object_cloud) == -1) {
    pcl::console::print_error ("Failed to load: %s\n", object_pcd_file.c_str ());
    printHelp (argc, argv);
    return -1;
  }

  auto object_cloud_copy = boost::make_shared <Cloud> (*object_cloud);

  auto matched_templates = detectTemplates (center_cloud, linemod);

  outputTemplateMatches (matched_templates);

  auto best_match_template = getClosestMatchedTemplate (matched_templates);

  outputTemplateMatch (best_match_template);

  auto const & multi_mod_template = linemod.getTemplate (best_match_template.template_id);

  auto x1 = best_match_template.x;
  auto y1 = best_match_template.y;
  auto x2 = x1 + multi_mod_template.region.width;
  auto y2 = y1 + multi_mod_template.region.height;

  // TODO: Remove table from the center_cloud to remove table points from centroid calculation

  auto linemod_region_centroid = getCentroidOrganisedRect (center_cloud, x1, y1, x2, y2);

  std::cout << "Centroid of linemod detection region: \n" << linemod_region_centroid << std::endl;
  auto image_extractor = pcl::io::PointCloudImageExtractorFromRGBField <pcl::PointXYZRGBA> {};
  auto image = boost::make_shared <pcl::PCLImage> ();
  auto extracted = image_extractor.extract (*center_cloud, *image);

  // Paint region red
  if (extracted) {
    for (int y = y1; y < y2; ++y)
      for (int x = x1; x < x2; ++x) {
        // Paint red
        auto offset = 3 * (y * image->width + x);
        image->data.at (offset) = 255;
      }
  }

  auto image_data = reinterpret_cast <unsigned char *> (image->data.data ());

  auto image_viewer = std::make_shared <pcl::visualization::ImageViewer> ("Image Viewer");
  image_viewer->addRGBImage (image_data, image->width, image->height);
  image_viewer->spin ();

  auto cloud_viz = std::make_shared <pcl::visualization::PCLVisualizer> ("Cloud Viewer");
  cloud_viz->initCameraParameters ();

  auto camera = pcl::visualization::Camera {};
  cloud_viz->getCameraParameters (camera);
  camera.view[1] *= -1;
  cloud_viz->setCameraParameters (camera);

  // Rotate the object about the x axis so that it is upright w.r.t. camera
  auto rotate_about_x = Eigen::Matrix4f {};
  rotate_about_x <<
      1.0, 0.0, 0.0, 0.0,
      0.0, -1., 0.0, 0.0,
      0.0, 0.0, -1., 0.0,
      0.0, 0.0, 0.0, 1.0;
  pcl::transformPointCloud (*object_cloud, *object_cloud, rotate_about_x);

  auto centroid_transform = getCentroidTransform (object_cloud, linemod_region_centroid);
  pcl::transformPointCloud (*object_cloud, *object_cloud, centroid_transform);

  auto object_cluster_cloud = getClosestCluster (merged_cloud, linemod_region_centroid);

  auto icp_transform = pairAlign (object_cloud, object_cluster_cloud);
  pcl::transformPointCloud (*object_cloud, *object_cloud, icp_transform);
  auto red_color_handler = pcl::visualization::PointCloudColorHandlerCustom <PointType>
      {object_cloud, 255, 0, 0};
  cloud_viz->addPointCloud <PointType> (merged_cloud);

  std::cout << "Rotate Transform : \n" << rotate_about_x << std::endl;
  std::cout << "Centroid Transform : \n" << centroid_transform << std::endl;
  std::cout << "ICP Transform : \n" << icp_transform << std::endl;

  // Validate that the transform is correct
  auto final_transform = Eigen::Matrix4f {icp_transform * centroid_transform * rotate_about_x};

  pcl::transformPointCloud (*object_cloud_copy, *object_cloud_copy, final_transform);
  cloud_viz->addPointCloud <PointType> (object_cloud_copy, red_color_handler, "object");
  std::cout << "Final transform = \n" << final_transform << std::endl;
  // Output this to a file
  cloud_viz->spin ();

  return (0);
}