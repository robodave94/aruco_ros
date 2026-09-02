/**
 * @file aruco_cv_utils.hpp
 * @brief Header-only utilities for ArUco marker detection using OpenCV's cv::aruco API.
 *
 * All new aruco_ros_cv nodes include this header directly — no shared library needed.
 */

#ifndef ARUCO_ROS_CV__ARUCO_CV_UTILS_HPP_
#define ARUCO_ROS_CV__ARUCO_CV_UTILS_HPP_

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace aruco_ros_cv
{

// ============================================================================
// Detection result structures
// ============================================================================

/** Single detected marker with all associated data. */
struct DetectedMarker
{
  int id;
  std::string dictionary_name;
  double marker_size;
  std::vector<cv::Point2f> corners;  // 4 corners in image coords
  cv::Vec3d rvec;                    // rotation vector (Rodrigues)
  cv::Vec3d tvec;                    // translation vector
};

/** Aggregated detection result for an entire frame. */
struct DetectionResult
{
  std::vector<DetectedMarker> markers;
};

// ============================================================================
// Dictionary name <-> OpenCV enum mapping
// ============================================================================

/** Maps a string dictionary name to the OpenCV cv::aruco predefined dictionary ID.
 *  Throws std::invalid_argument for unknown names. */
inline int dictionaryFromString(const std::string & name)
{
  static const std::map<std::string, int> dict_map = {
    {"DICT_4X4_50",    cv::aruco::DICT_4X4_50},
    {"DICT_4X4_100",   cv::aruco::DICT_4X4_100},
    {"DICT_4X4_250",   cv::aruco::DICT_4X4_250},
    {"DICT_4X4_1000",  cv::aruco::DICT_4X4_1000},
    {"DICT_5X5_50",    cv::aruco::DICT_5X5_50},
    {"DICT_5X5_100",   cv::aruco::DICT_5X5_100},
    {"DICT_5X5_250",   cv::aruco::DICT_5X5_250},
    {"DICT_5X5_1000",  cv::aruco::DICT_5X5_1000},
    {"DICT_6X6_50",    cv::aruco::DICT_6X6_50},
    {"DICT_6X6_100",   cv::aruco::DICT_6X6_100},
    {"DICT_6X6_250",   cv::aruco::DICT_6X6_250},
    {"DICT_6X6_1000",  cv::aruco::DICT_6X6_1000},
    {"DICT_7X7_50",    cv::aruco::DICT_7X7_50},
    {"DICT_7X7_100",   cv::aruco::DICT_7X7_100},
    {"DICT_7X7_250",   cv::aruco::DICT_7X7_250},
    {"DICT_7X7_1000",  cv::aruco::DICT_7X7_1000},
    {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
    {"DICT_APRILTAG_16h5",  cv::aruco::DICT_APRILTAG_16h5},
    {"DICT_APRILTAG_25h9",  cv::aruco::DICT_APRILTAG_25h9},
    {"DICT_APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10},
    {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11},
  };
  auto it = dict_map.find(name);
  if (it == dict_map.end()) {
    throw std::invalid_argument("Unknown ArUco dictionary: " + name);
  }
  return it->second;
}

// ============================================================================
// Camera intrinsics helpers
// ============================================================================

/** Build a 3x3 camera matrix from explicit parameters. */
inline cv::Mat buildCameraMatrix(double fx, double fy, double cx, double cy)
{
  cv::Mat cam = cv::Mat::eye(3, 3, CV_64F);
  cam.at<double>(0, 0) = fx;
  cam.at<double>(1, 1) = fy;
  cam.at<double>(0, 2) = cx;
  cam.at<double>(1, 2) = cy;
  return cam;
}

/** Build a 5x1 distortion coefficient vector. */
inline cv::Mat buildDistCoeffs(double k1, double k2, double p1, double p2, double k3)
{
  cv::Mat dist(5, 1, CV_64F);
  dist.at<double>(0) = k1;
  dist.at<double>(1) = k2;
  dist.at<double>(2) = p1;
  dist.at<double>(3) = p2;
  dist.at<double>(4) = k3;
  return dist;
}

/** Load camera intrinsics from an OpenCV FileStorage YAML file.
 *  Expected keys: "cameraMatrix" (3x3) and "distCoeffs" (5x1).
 *  @return true on success, false if file cannot be opened or keys are missing. */
inline bool loadCameraIntrinsicsFromYAML(
  const std::string & path,
  cv::Mat & camera_matrix,
  cv::Mat & dist_coeffs)
{
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }
  fs["cameraMatrix"] >> camera_matrix;
  fs["distCoeffs"] >> dist_coeffs;
  // Fall back to the snake_case keys used by saveCameraIntrinsicsToYAML and some external tools.
  if (camera_matrix.empty()) {
    fs["camera_matrix"] >> camera_matrix;
  }
  if (dist_coeffs.empty()) {
    fs["dist_coeffs"] >> dist_coeffs;
  }
  fs.release();
  if (camera_matrix.empty() || dist_coeffs.empty()) {
    return false;
  }
  camera_matrix.convertTo(camera_matrix, CV_64F);
  dist_coeffs.convertTo(dist_coeffs, CV_64F);
  return true;
}

/** Save camera intrinsics to an OpenCV FileStorage YAML file. */
inline bool saveCameraIntrinsicsToYAML(
  const std::string & path,
  const cv::Mat & camera_matrix,
  const cv::Mat & dist_coeffs)
{
  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) {
    return false;
  }
  fs << "camera_matrix" << camera_matrix;
  fs << "dist_coeffs" << dist_coeffs;
  fs.release();
  return true;
}

// ============================================================================
// Multi-dictionary detection
// ============================================================================

/** Detect ArUco markers across multiple dictionaries, each with its own marker size.
 *  Runs cv::aruco::detectMarkers + cv::aruco::estimatePoseSingleMarkers per dictionary.
 *
 *  @param image       BGR or grayscale input image.
 *  @param dictionaries  Vector of dictionary name strings.
 *  @param marker_sizes  Vector of physical marker sizes in meters (parallel to dictionaries).
 *  @param camera_matrix 3x3 camera intrinsic matrix.
 *  @param dist_coeffs   Distortion coefficients.
 *  @return DetectionResult with all detected markers. */
inline DetectionResult detectMultiDictMarkers(
  const cv::Mat & image,
  const std::vector<std::string> & dictionaries,
  const std::vector<double> & marker_sizes,
  const cv::Mat & camera_matrix,
  const cv::Mat & dist_coeffs)
{
  if (dictionaries.size() != marker_sizes.size()) {
    throw std::invalid_argument(
            "dictionaries and marker_sizes must have the same length");
  }

  DetectionResult result;
  cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();

  for (size_t d = 0; d < dictionaries.size(); ++d) {
    int dict_id = dictionaryFromString(dictionaries[d]);
    cv::Ptr<cv::aruco::Dictionary> dictionary =
      cv::aruco::getPredefinedDictionary(dict_id);

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<std::vector<cv::Point2f>> rejected;

    cv::aruco::detectMarkers(image, dictionary, corners, ids, params, rejected);

    if (ids.empty()) {
      continue;
    }

    // Estimate 3D pose for each detected marker
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(
      corners, static_cast<float>(marker_sizes[d]),
      camera_matrix, dist_coeffs, rvecs, tvecs);

    for (size_t i = 0; i < ids.size(); ++i) {
      DetectedMarker m;
      m.id = ids[i];
      m.dictionary_name = dictionaries[d];
      m.marker_size = marker_sizes[d];
      m.corners = corners[i];
      m.rvec = rvecs[i];
      m.tvec = tvecs[i];
      result.markers.push_back(m);
    }
  }

  return result;
}

// ============================================================================
// Zoomed (crop + upscale) detection for small markers
// ============================================================================

/** Virtual crop + upscale applied before detection to help see small markers. */
struct ZoomConfig
{
  double center_x{-1.0};        // crop center px; <0 => camera principal point
  double center_y{-1.0};        // crop center py; <0 => camera principal point
  double width{500.0};          // crop width px
  double height{500.0};         // crop height px
  double upscale{1.5};          // upscale factor applied to the crop
  bool rescale_distortion{false};  // undistort the crop before detection when true
};

/** Detect markers on a virtually cropped + upscaled region.
 *  Poses are returned in the true camera optical frame (valid with the ORIGINAL
 *  camera matrix) and corners are remapped back to ORIGINAL image coordinates.
 *
 *  Cropping shifts the principal point and upscaling multiplies focal length +
 *  principal point, so estimatePoseSingleMarkers on the adjusted intrinsics yields
 *  the same physical pose as detecting on the full frame.
 *
 *  @param debug_image  If non-null, receives the upscaled crop with detections drawn
 *                      (in zoomed coordinates) for diagnostics. */
inline DetectionResult detectMultiDictMarkersZoomed(
  const cv::Mat & image,
  const std::vector<std::string> & dictionaries,
  const std::vector<double> & marker_sizes,
  const cv::Mat & camera_matrix,
  const cv::Mat & dist_coeffs,
  const ZoomConfig & zoom,
  cv::Mat * debug_image = nullptr)
{
  const double fx = camera_matrix.at<double>(0, 0);
  const double fy = camera_matrix.at<double>(1, 1);
  const double cx = camera_matrix.at<double>(0, 2);
  const double cy = camera_matrix.at<double>(1, 2);

  const double center_x = (zoom.center_x < 0.0) ? cx : zoom.center_x;
  const double center_y = (zoom.center_y < 0.0) ? cy : zoom.center_y;
  const double scale = (zoom.upscale > 0.0) ? zoom.upscale : 1.0;

  // Crop ROI clamped to image bounds
  int roi_x = static_cast<int>(std::lround(center_x - zoom.width / 2.0));
  int roi_y = static_cast<int>(std::lround(center_y - zoom.height / 2.0));
  int roi_w = static_cast<int>(std::lround(zoom.width));
  int roi_h = static_cast<int>(std::lround(zoom.height));

  roi_x = std::max(0, std::min(roi_x, image.cols - 1));
  roi_y = std::max(0, std::min(roi_y, image.rows - 1));
  roi_w = std::max(1, std::min(roi_w, image.cols - roi_x));
  roi_h = std::max(1, std::min(roi_h, image.rows - roi_y));

  cv::Rect roi(roi_x, roi_y, roi_w, roi_h);
  cv::Mat crop = image(roi).clone();

  // Intrinsics for the crop (principal point shifted by the ROI origin)
  cv::Mat adj_cam = camera_matrix.clone();
  adj_cam.at<double>(0, 2) = cx - roi_x;
  adj_cam.at<double>(1, 2) = cy - roi_y;
  cv::Mat adj_dist = dist_coeffs.clone();

  if (zoom.rescale_distortion && !dist_coeffs.empty()) {
    cv::Mat undistorted;
    cv::undistort(crop, undistorted, adj_cam, adj_dist);
    crop = undistorted;
    adj_dist = cv::Mat::zeros(dist_coeffs.rows, dist_coeffs.cols, dist_coeffs.type());
  }

  // Upscale the crop and its intrinsics
  cv::Mat scaled;
  cv::resize(crop, scaled, cv::Size(), scale, scale, cv::INTER_CUBIC);
  adj_cam.at<double>(0, 0) = fx * scale;
  adj_cam.at<double>(1, 1) = fy * scale;
  adj_cam.at<double>(0, 2) = adj_cam.at<double>(0, 2) * scale;
  adj_cam.at<double>(1, 2) = adj_cam.at<double>(1, 2) * scale;

  DetectionResult result = detectMultiDictMarkers(
    scaled, dictionaries, marker_sizes, adj_cam, adj_dist);

  // Build the zoomed debug visualization (in upscaled-crop coordinates) before remap
  if (debug_image != nullptr) {
    cv::Mat vis = scaled.clone();
    for (const auto & m : result.markers) {
      for (int j = 0; j < 4; ++j) {
        cv::line(vis, m.corners[j], m.corners[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
      }
      float axis_len = static_cast<float>(m.marker_size) * 0.75f;
      cv::drawFrameAxes(vis, adj_cam, adj_dist, m.rvec, m.tvec, axis_len);
      std::string label = std::to_string(m.id) + ":" + m.dictionary_name;
      cv::Point label_pos(
        static_cast<int>(m.corners[0].x),
        std::max(14, static_cast<int>(m.corners[0].y) - 8));
      cv::putText(vis, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(0, 255, 0), 1);
    }
    *debug_image = vis;
  }

  // Remap corners from the upscaled crop back into original image coordinates
  for (auto & m : result.markers) {
    for (auto & c : m.corners) {
      c.x = static_cast<float>(c.x / scale + roi_x);
      c.y = static_cast<float>(c.y / scale + roi_y);
    }
  }

  return result;
}

// ============================================================================
// Visualization
// ============================================================================

/** Per-dictionary color palette so different dictionaries look distinct. */
inline cv::Scalar getDictColor(size_t dict_index)
{
  static const cv::Scalar colors[] = {
    {0, 255, 0},     // green
    {255, 0, 0},     // blue (BGR)
    {0, 0, 255},     // red
    {255, 255, 0},   // cyan
    {255, 0, 255},   // magenta
    {0, 255, 255},   // yellow
    {128, 255, 0},
    {255, 128, 0},
  };
  return colors[dict_index % 8];
}

/** Resolve dict name -> index for coloring. */
inline size_t dictNameToColorIndex(
  const std::string & name,
  const std::vector<std::string> & dict_list)
{
  for (size_t i = 0; i < dict_list.size(); ++i) {
    if (dict_list[i] == name) return i;
  }
  return 0;
}

/** Draw 2D bounding boxes around detected markers on a copy of the image.
 *  Each marker gets a colored quadrilateral + text label "ID:dict". */
inline cv::Mat draw2DVisualization(
  const cv::Mat & image,
  const DetectionResult & result,
  const std::vector<std::string> & dict_list)
{
  cv::Mat vis = image.clone();

  for (const auto & m : result.markers) {
    cv::Scalar color = getDictColor(dictNameToColorIndex(m.dictionary_name, dict_list));

    // Draw quadrilateral
    for (int j = 0; j < 4; ++j) {
      cv::line(vis, m.corners[j], m.corners[(j + 1) % 4], color, 2);
    }

    // Label
    std::string label = std::to_string(m.id) + ":" + m.dictionary_name;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    cv::Point label_pos(
      static_cast<int>(m.corners[0].x),
      static_cast<int>(m.corners[0].y) - 8);
    // clamp
    label_pos.x = std::max(0, label_pos.x);
    label_pos.y = std::max(text_size.height, label_pos.y);

    cv::putText(vis, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
  }

  return vis;
}

/** Draw 3D axis on each detected marker (uses cv::drawFrameAxes). */
inline cv::Mat draw3DVisualization(
  const cv::Mat & image,
  const DetectionResult & result,
  const cv::Mat & camera_matrix,
  const cv::Mat & dist_coeffs,
  const std::vector<std::string> & dict_list)
{
  cv::Mat vis = image.clone();

  for (const auto & m : result.markers) {
    float axis_len = static_cast<float>(m.marker_size) * 0.75f;
    cv::drawFrameAxes(vis, camera_matrix, dist_coeffs, m.rvec, m.tvec, axis_len);

    // Also draw bounding box
    cv::Scalar color = getDictColor(dictNameToColorIndex(m.dictionary_name, dict_list));
    for (int j = 0; j < 4; ++j) {
      cv::line(vis, m.corners[j], m.corners[(j + 1) % 4], color, 2);
    }

    // Label with 3D translation info
    std::ostringstream oss;
    oss << "ID" << m.id << " [" << std::fixed;
    oss.precision(3);
    oss << m.tvec[0] << ", " << m.tvec[1] << ", " << m.tvec[2] << "]";

    cv::Point label_pos(
      static_cast<int>(m.corners[0].x),
      static_cast<int>(m.corners[0].y) - 8);
    label_pos.x = std::max(0, label_pos.x);
    label_pos.y = std::max(14, label_pos.y);

    cv::putText(vis, oss.str(), label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
  }

  return vis;
}

// ============================================================================
// ROS message conversion helpers
// ============================================================================

/** Convert rvec + tvec to a geometry_msgs-compatible representation.
 *  Returns: position (x,y,z) and quaternion (x,y,z,w). */
inline void rvecTvecToPositionQuat(
  const cv::Vec3d & rvec, const cv::Vec3d & tvec,
  double & px, double & py, double & pz,
  double & qx, double & qy, double & qz, double & qw)
{
  px = tvec[0];
  py = tvec[1];
  pz = tvec[2];

  // Convert Rodrigues to rotation matrix, then to quaternion
  cv::Mat rot_mat;
  cv::Rodrigues(rvec, rot_mat);

  // Rotation matrix to quaternion (Shepperd's method)
  double m00 = rot_mat.at<double>(0, 0);
  double m01 = rot_mat.at<double>(0, 1);
  double m02 = rot_mat.at<double>(0, 2);
  double m10 = rot_mat.at<double>(1, 0);
  double m11 = rot_mat.at<double>(1, 1);
  double m12 = rot_mat.at<double>(1, 2);
  double m20 = rot_mat.at<double>(2, 0);
  double m21 = rot_mat.at<double>(2, 1);
  double m22 = rot_mat.at<double>(2, 2);

  double trace = m00 + m11 + m22;
  if (trace > 0.0) {
    double s = 0.5 / std::sqrt(trace + 1.0);
    qw = 0.25 / s;
    qx = (m21 - m12) * s;
    qy = (m02 - m20) * s;
    qz = (m10 - m01) * s;
  } else if (m00 > m11 && m00 > m22) {
    double s = 2.0 * std::sqrt(1.0 + m00 - m11 - m22);
    qw = (m21 - m12) / s;
    qx = 0.25 * s;
    qy = (m01 + m10) / s;
    qz = (m02 + m20) / s;
  } else if (m11 > m22) {
    double s = 2.0 * std::sqrt(1.0 + m11 - m00 - m22);
    qw = (m02 - m20) / s;
    qx = (m01 + m10) / s;
    qy = 0.25 * s;
    qz = (m12 + m21) / s;
  } else {
    double s = 2.0 * std::sqrt(1.0 + m22 - m00 - m11);
    qw = (m10 - m01) / s;
    qx = (m02 + m20) / s;
    qy = (m12 + m21) / s;
    qz = 0.25 * s;
  }
}

// ============================================================================
// TF frame naming
// ============================================================================

/** Format a marker size in meters as a compact millimeter token (e.g. 0.004 -> "4mm"). */
inline std::string markerSizeToMillimeterToken(double marker_size)
{
  double mm = marker_size * 1000.0;
  double rounded = std::round(mm);
  std::ostringstream oss;
  if (std::abs(mm - rounded) < 1e-6) {
    oss << static_cast<long long>(rounded) << "mm";
  } else {
    oss << std::fixed << std::setprecision(1) << mm << "mm";
  }
  return oss.str();
}

/** Build a self-describing TF frame name, e.g. "marker_DICT_4X4_50_4mm_7". */
inline std::string makeMarkerFrameName(
  const std::string & dictionary_name, double marker_size, int id)
{
  return "marker_" + dictionary_name + "_" +
         markerSizeToMillimeterToken(marker_size) + "_" + std::to_string(id);
}

/** Build unique TF frame names for a detection result, appending _A/_B/... when the
 *  same dictionary+id appears multiple times in a single frame. */
inline std::vector<std::string> buildUniqueMarkerFrameNames(const DetectionResult & result)
{
  std::map<std::string, int> totals;
  std::vector<std::string> bases;
  bases.reserve(result.markers.size());
  for (const auto & m : result.markers) {
    std::string base = makeMarkerFrameName(m.dictionary_name, m.marker_size, m.id);
    bases.push_back(base);
    totals[base]++;
  }

  std::map<std::string, int> seen;
  std::vector<std::string> names;
  names.reserve(result.markers.size());
  for (const auto & base : bases) {
    if (totals[base] > 1) {
      int idx = seen[base]++;
      char suffix = static_cast<char>('A' + (idx % 26));
      names.push_back(base + "_" + std::string(1, suffix));
    } else {
      names.push_back(base);
    }
  }
  return names;
}

}  // namespace aruco_ros_cv

#endif  // ARUCO_ROS_CV__ARUCO_CV_UTILS_HPP_
