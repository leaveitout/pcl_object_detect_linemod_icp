//
// Created by sean on 30/06/16.
//

#include "pcltools/fileio.hpp"


auto pcltools::fileio::getPcdFilesInPath (fs::path const & pcd_dir)
-> std::deque <fs::path> {
  auto result_set = std::deque < fs::path > {};
  for (auto const & entry : boost::make_iterator_range (fs::directory_iterator{pcd_dir})) {
    if (fs::is_regular_file (entry.status ())) {
      if (entry.path ().extension () == ".pcd") {
        result_set.emplace_back (entry);
      }
    }
  }
  return result_set;
}


auto pcltools::fileio::checkValidFile (fs::path const & filepath)
-> bool {
  return fs::exists (filepath) && fs::is_regular_file (filepath);
}


auto pcltools::fileio::checkValidDir (fs::path const & dirpath) -> bool {
  return fs::exists (dirpath) && fs::is_directory (dirpath);
}


auto expandTilde (std::string path_string) -> fs::path {
  if (path_string.at (0) == '~')
    path_string.replace (0, 1, getenv ("HOME"));
  return fs::path{path_string};
}

