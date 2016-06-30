//
// Created by sean on 30/06/16.
//

#ifndef PCL_OBJECT_DETECT_LINEMOD_FILEIO_HPP
#define PCL_OBJECT_DETECT_LINEMOD_FILEIO_HPP

#include <deque>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

namespace fs = boost::filesystem;

namespace pcltools {
namespace fileio {

auto getPcdFilesInPath (fs::path const & pcd_dir) -> std::deque <fs::path>;

auto checkValidFile (fs::path const & filepath) -> bool;

auto checkValidDir (fs::path const & dirpath) -> bool;

}
}

#endif //PCL_OBJECT_DETECT_LINEMOD_FILEIO_HPP

