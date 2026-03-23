/**
 * @file camera_calibration.cpp
 * @brief ROS2 node for camera calibration using a checkerboard pattern.
 *
 * Subscribes to an RGB image topic, collects frames with detected checkerboard corners,
 * runs cv::calibrateCamera, and saves the result as an OpenCV FileStorage YAML.
 *
 * Parameters:
 *   image_topic        - Input RGB image topic
 *   checkerboard_rows  - Number of inner corners per row
 *   checkerboard_cols  - Number of inner corners per column
 *   square_size        - Size of each square in meters
 *   output_file        - Path to save the calibration YAML
 *   num_frames         - Number of frames to collect for calibration
 */

#include <rclcpp/rclcpp.hpp>

#if __has_include("cv_bridge/cv_bridge.hpp")
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <sensor_msgs/image_encodings.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "aruco_ros_cv/aruco_cv_utils.hpp"

#include <mutex>
#include <vector>

class CameraCalibrationNode : public rclcpp::Node
{
public:
  CameraCalibrationNode()
  : Node("camera_calibration"), frames_collected_(0), calibration_done_(false)
  {
    this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    this->declare_parameter<int>("checkerboard_rows", 6);
    this->declare_parameter<int>("checkerboard_cols", 9);
    this->declare_parameter<double>("square_size", 0.025);
    this->declare_parameter<std::string>("output_file", "camera_params.yml");
    this->declare_parameter<int>("num_frames", 20);

    image_topic_ = this->get_parameter("image_topic").as_string();
    board_rows_ = this->get_parameter("checkerboard_rows").as_int();
    board_cols_ = this->get_parameter("checkerboard_cols").as_int();
    square_size_ = this->get_parameter("square_size").as_double();
    output_file_ = this->get_parameter("output_file").as_string();
    num_frames_ = this->get_parameter("num_frames").as_int();

    board_size_ = cv::Size(board_cols_, board_rows_);

    // Generate 3D object points for the checkerboard
    for (int i = 0; i < board_rows_; ++i) {
      for (int j = 0; j < board_cols_; ++j) {
        obj_point_template_.push_back(
          cv::Point3f(
            static_cast<float>(j * square_size_),
            static_cast<float>(i * square_size_),
            0.0f));
      }
    }

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&CameraCalibrationNode::image_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Camera calibration node started");
    RCLCPP_INFO(this->get_logger(), "  Image topic:        %s", image_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  Checkerboard:       %dx%d", board_cols_, board_rows_);
    RCLCPP_INFO(this->get_logger(), "  Square size:        %.4f m", square_size_);
    RCLCPP_INFO(this->get_logger(), "  Frames to collect:  %d", num_frames_);
    RCLCPP_INFO(this->get_logger(), "  Output file:        %s", output_file_.c_str());
    RCLCPP_INFO(this->get_logger(), "Waiting for images with checkerboard pattern...");
  }

private:
  std::string image_topic_;
  int board_rows_, board_cols_;
  double square_size_;
  std::string output_file_;
  int num_frames_;
  cv::Size board_size_;
  cv::Size image_size_;

  std::vector<cv::Point3f> obj_point_template_;
  std::vector<std::vector<cv::Point3f>> obj_points_;
  std::vector<std::vector<cv::Point2f>> img_points_;

  int frames_collected_;
  bool calibration_done_;
  std::mutex mutex_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    RCLCPP_INFO(this->get_logger(), "Processing incoming image frame...");

    if (calibration_done_) return;

    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat image = cv_ptr->image;
    image_size_ = image.size();

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(
      gray, board_size_, corners,
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

    if (found) {
      // Refine corner positions
      cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

      obj_points_.push_back(obj_point_template_);
      img_points_.push_back(corners);
      frames_collected_++;

      RCLCPP_INFO(this->get_logger(), "Checkerboard found! Frame %d/%d collected",
        frames_collected_, num_frames_);

      if (frames_collected_ >= num_frames_) {
        run_calibration();
      }
    }
  }

  void run_calibration()
  {
    RCLCPP_INFO(this->get_logger(), "Running calibration with %d frames...", frames_collected_);

    cv::Mat camera_matrix, dist_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;

    double rms = cv::calibrateCamera(
      obj_points_, img_points_, image_size_,
      camera_matrix, dist_coeffs, rvecs, tvecs);

    RCLCPP_INFO(this->get_logger(), "=== Calibration Complete ===");
    RCLCPP_INFO(this->get_logger(), "RMS reprojection error: %.6f", rms);
    RCLCPP_INFO(this->get_logger(), "Camera Matrix:");
    RCLCPP_INFO(this->get_logger(), "  fx=%.6f  fy=%.6f",
      camera_matrix.at<double>(0, 0), camera_matrix.at<double>(1, 1));
    RCLCPP_INFO(this->get_logger(), "  cx=%.6f  cy=%.6f",
      camera_matrix.at<double>(0, 2), camera_matrix.at<double>(1, 2));
    RCLCPP_INFO(this->get_logger(), "Distortion Coefficients:");

    std::ostringstream dist_str;
    for (int i = 0; i < dist_coeffs.cols * dist_coeffs.rows; ++i) {
      if (i > 0) dist_str << ", ";
      dist_str << dist_coeffs.at<double>(i);
    }
    RCLCPP_INFO(this->get_logger(), "  [%s]", dist_str.str().c_str());

    // Save to YAML
    if (aruco_ros_cv::saveCameraIntrinsicsToYAML(output_file_, camera_matrix, dist_coeffs)) {
      RCLCPP_INFO(this->get_logger(), "Calibration saved to: %s", output_file_.c_str());
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to save calibration to: %s", output_file_.c_str());
    }

    calibration_done_ = true;
    RCLCPP_INFO(this->get_logger(), "Calibration node finished. You may shut down.");

    // Node shutdown
    rclcpp::shutdown();
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CameraCalibrationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
