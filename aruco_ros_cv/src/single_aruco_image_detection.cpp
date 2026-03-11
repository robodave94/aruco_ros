/**
 * @file single_aruco_image_detection.cpp
 * @brief ROS2 node for single-image ArUco marker detection.
 *
 * Reads an image from file, detects markers, prints results, saves 2D/3D visualizations.
 * Parameters:
 *   image_path           - Path to input image
 *   camera_params_file   - Path to OpenCV YAML camera intrinsics
 *   dictionaries         - List of dictionary names
 *   marker_sizes         - List of marker sizes (meters), parallel to dictionaries
 *   output_dir           - Directory to save visualizations (defaults to same as image)
 */

#include <rclcpp/rclcpp.hpp>
#include <filesystem>

#include "aruco_ros_cv/aruco_cv_utils.hpp"

class SingleArucoImageDetection : public rclcpp::Node
{
public:
  SingleArucoImageDetection()
  : Node("single_aruco_image_detection")
  {
    this->declare_parameter<std::string>("image_path", "");
    this->declare_parameter<std::string>("camera_params_file", "");
    this->declare_parameter<std::vector<std::string>>("dictionaries", {"DICT_6X6_250"});
    this->declare_parameter<std::vector<double>>("marker_sizes", {0.05});
    this->declare_parameter<std::string>("output_dir", "");
  }

  void run()
  {
    std::string image_path = this->get_parameter("image_path").as_string();
    std::string camera_params_file = this->get_parameter("camera_params_file").as_string();
    std::vector<std::string> dictionaries =
      this->get_parameter("dictionaries").as_string_array();
    std::vector<double> marker_sizes = this->get_parameter("marker_sizes").as_double_array();
    std::string output_dir = this->get_parameter("output_dir").as_string();

    // Validate
    if (image_path.empty()) {
      RCLCPP_ERROR(this->get_logger(), "image_path parameter is required");
      return;
    }
    if (camera_params_file.empty()) {
      RCLCPP_ERROR(this->get_logger(), "camera_params_file parameter is required");
      return;
    }
    if (dictionaries.size() != marker_sizes.size()) {
      RCLCPP_ERROR(this->get_logger(), "dictionaries and marker_sizes must have equal length");
      return;
    }

    // Load image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load image: %s", image_path.c_str());
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Loaded image: %s (%dx%d)",
      image_path.c_str(), image.cols, image.rows);

    // Load camera intrinsics
    cv::Mat cam_mtx, dist_coeffs;
    if (!aruco_ros_cv::loadCameraIntrinsicsFromYAML(camera_params_file, cam_mtx, dist_coeffs)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load camera params: %s",
        camera_params_file.c_str());
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Loaded camera intrinsics from: %s",
      camera_params_file.c_str());

    // Detect
    aruco_ros_cv::DetectionResult result;
    try {
      result = aruco_ros_cv::detectMultiDictMarkers(
        image, dictionaries, marker_sizes, cam_mtx, dist_coeffs);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Detection error: %s", e.what());
      return;
    }

    RCLCPP_INFO(this->get_logger(), "=== Detection Results ===");
    RCLCPP_INFO(this->get_logger(), "Total markers detected: %zu", result.markers.size());

    for (const auto & m : result.markers) {
      double px, py, pz, qx, qy, qz, qw;
      aruco_ros_cv::rvecTvecToPositionQuat(m.rvec, m.tvec, px, py, pz, qx, qy, qz, qw);

      RCLCPP_INFO(this->get_logger(), "--- Marker ID: %d ---", m.id);
      RCLCPP_INFO(this->get_logger(), "  Dictionary: %s", m.dictionary_name.c_str());
      RCLCPP_INFO(this->get_logger(), "  Marker size: %.4f m", m.marker_size);
      RCLCPP_INFO(this->get_logger(), "  2D Corners:");
      for (size_t i = 0; i < m.corners.size(); ++i) {
        RCLCPP_INFO(this->get_logger(), "    [%zu] (%.1f, %.1f)", i,
          m.corners[i].x, m.corners[i].y);
      }
      RCLCPP_INFO(this->get_logger(), "  3D Position: (%.4f, %.4f, %.4f)", px, py, pz);
      RCLCPP_INFO(this->get_logger(), "  Quaternion:  (%.4f, %.4f, %.4f, %.4f)", qx, qy, qz, qw);
    }

    // Generate visualizations
    cv::Mat vis_2d = aruco_ros_cv::draw2DVisualization(image, result, dictionaries);
    cv::Mat vis_3d = aruco_ros_cv::draw3DVisualization(
      image, result, cam_mtx, dist_coeffs, dictionaries);

    // Determine output paths
    namespace fs = std::filesystem;
    fs::path img_path(image_path);
    fs::path out_dir = output_dir.empty() ? img_path.parent_path() : fs::path(output_dir);
    if (!fs::exists(out_dir)) {
      fs::create_directories(out_dir);
    }
    std::string stem = img_path.stem().string();
    std::string ext = img_path.extension().string();
    if (ext.empty()) ext = ".jpg";

    fs::path out_2d = out_dir / (stem + "_2dvis" + ext);
    fs::path out_3d = out_dir / (stem + "_3dvis" + ext);

    if (cv::imwrite(out_2d.string(), vis_2d)) {
      RCLCPP_INFO(this->get_logger(), "Saved 2D visualization: %s", out_2d.c_str());
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to save 2D visualization: %s", out_2d.c_str());
    }

    if (cv::imwrite(out_3d.string(), vis_3d)) {
      RCLCPP_INFO(this->get_logger(), "Saved 3D visualization: %s", out_3d.c_str());
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to save 3D visualization: %s", out_3d.c_str());
    }

    RCLCPP_INFO(this->get_logger(), "=== Detection complete ===");
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SingleArucoImageDetection>();
  node->run();
  rclcpp::spin_some(node);
  rclcpp::shutdown();
  return 0;
}
