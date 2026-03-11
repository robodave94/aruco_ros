/**
 * @file aruco_rt_capture.cpp
 * @brief ROS2 node for real-time ArUco marker detection from an RGB image topic.
 *
 * Subscribes to an RGB image topic, detects ArUco markers in each frame,
 * and publishes:
 *   - <image_topic>/arucofeed        : Visualized image (2D boxes + 3D axes)
 *   - <image_topic>/arucofeed/markers : aruco_msgs/MarkerArray with 2D+3D poses
 *
 * Parameters:
 *   image_topic     - Input RGB image topic name
 *   dictionaries    - List of dictionary names
 *   marker_sizes    - List of marker sizes in meters
 *   fx, fy, cx, cy, k1, k2, p1, p2, k3 - Camera intrinsics
 *   image_width, image_height - Image dimensions (informational)
 */

#include <rclcpp/rclcpp.hpp>

#if __has_include("cv_bridge/cv_bridge.hpp")
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <image_transport/image_transport.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include "aruco_msgs/msg/marker.hpp"
#include "aruco_msgs/msg/marker_array.hpp"
#include "aruco_ros_cv/aruco_cv_utils.hpp"

class ArucoRtCaptureNode : public rclcpp::Node
{
public:
  ArucoRtCaptureNode()
  : Node("aruco_rt_capture")
  {
    // Declare parameters
    this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    this->declare_parameter<std::vector<std::string>>("dictionaries", {"DICT_6X6_250"});
    this->declare_parameter<std::vector<double>>("marker_sizes", {0.05});
    this->declare_parameter<double>("fx", 951.264);
    this->declare_parameter<double>("fy", 944.244);
    this->declare_parameter<double>("cx", 644.983);
    this->declare_parameter<double>("cy", 360.018);
    this->declare_parameter<double>("k1", 0.0);
    this->declare_parameter<double>("k2", 0.0);
    this->declare_parameter<double>("p1", 0.0);
    this->declare_parameter<double>("p2", 0.0);
    this->declare_parameter<double>("k3", 0.0);
    this->declare_parameter<int>("image_width", 1280);
    this->declare_parameter<int>("image_height", 720);

    // Get parameters
    image_topic_ = this->get_parameter("image_topic").as_string();
    dictionaries_ = this->get_parameter("dictionaries").as_string_array();
    marker_sizes_ = this->get_parameter("marker_sizes").as_double_array();

    double fx = this->get_parameter("fx").as_double();
    double fy = this->get_parameter("fy").as_double();
    double cx = this->get_parameter("cx").as_double();
    double cy = this->get_parameter("cy").as_double();
    double k1 = this->get_parameter("k1").as_double();
    double k2 = this->get_parameter("k2").as_double();
    double p1 = this->get_parameter("p1").as_double();
    double p2 = this->get_parameter("p2").as_double();
    double k3 = this->get_parameter("k3").as_double();

    cam_mtx_ = aruco_ros_cv::buildCameraMatrix(fx, fy, cx, cy);
    dist_coeffs_ = aruco_ros_cv::buildDistCoeffs(k1, k2, p1, p2, k3);

    if (dictionaries_.size() != marker_sizes_.size()) {
      RCLCPP_ERROR(this->get_logger(), "dictionaries and marker_sizes must have equal length");
      return;
    }

    // Validate dictionaries
    for (const auto & d : dictionaries_) {
      try {
        aruco_ros_cv::dictionaryFromString(d);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(), "%s", e.what());
        return;
      }
    }

    // Create publishers with sanitized topic names
    std::string feed_topic = image_topic_ + "/arucofeed";
    std::string markers_topic = image_topic_ + "/arucofeed/markers";

    feed_pub_ = this->create_publisher<sensor_msgs::msg::Image>(feed_topic, 10);
    markers_pub_ = this->create_publisher<aruco_msgs::msg::MarkerArray>(markers_topic, 10);

    // Subscribe to the image topic
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ArucoRtCaptureNode::image_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "ArUco RT capture started");
    RCLCPP_INFO(this->get_logger(), "  Input topic:   %s", image_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "  Feed topic:    %s", feed_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  Markers topic: %s", markers_topic.c_str());
    for (size_t i = 0; i < dictionaries_.size(); ++i) {
      RCLCPP_INFO(this->get_logger(), "  Dict[%zu]: %s  size=%.4fm",
        i, dictionaries_[i].c_str(), marker_sizes_[i]);
    }
  }

private:
  std::string image_topic_;
  std::vector<std::string> dictionaries_;
  std::vector<double> marker_sizes_;
  cv::Mat cam_mtx_;
  cv::Mat dist_coeffs_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feed_pub_;
  rclcpp::Publisher<aruco_msgs::msg::MarkerArray>::SharedPtr markers_pub_;

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat image = cv_ptr->image;

    // Detect
    aruco_ros_cv::DetectionResult det;
    try {
      det = aruco_ros_cv::detectMultiDictMarkers(
        image, dictionaries_, marker_sizes_, cam_mtx_, dist_coeffs_);
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "Detection error: %s", e.what());
      return;
    }

    auto stamp = msg->header.stamp;

    // Publish visualized feed (combined 2D boxes + 3D axes)
    if (feed_pub_->get_subscription_count() > 0) {
      cv::Mat vis = aruco_ros_cv::draw2DVisualization(image, det, dictionaries_);
      // overlay 3D axes on the same image
      for (const auto & m : det.markers) {
        float axis_len = static_cast<float>(m.marker_size) * 0.75f;
        cv::drawFrameAxes(vis, cam_mtx_, dist_coeffs_, m.rvec, m.tvec, axis_len);
      }

      cv_bridge::CvImage out;
      out.header.stamp = stamp;
      out.header.frame_id = msg->header.frame_id;
      out.encoding = sensor_msgs::image_encodings::BGR8;
      out.image = vis;
      feed_pub_->publish(*out.toImageMsg());
    }

    // Publish marker array
    if (markers_pub_->get_subscription_count() > 0) {
      aruco_msgs::msg::MarkerArray marker_array;
      marker_array.header.stamp = stamp;
      marker_array.header.frame_id = msg->header.frame_id;

      for (const auto & m : det.markers) {
        aruco_msgs::msg::Marker marker_msg;
        marker_msg.header.stamp = stamp;
        marker_msg.header.frame_id = msg->header.frame_id;
        marker_msg.id = static_cast<uint32_t>(m.id);
        marker_msg.confidence = 1.0;
        marker_msg.dictionary = m.dictionary_name;
        marker_msg.marker_size = m.marker_size;

        for (const auto & c : m.corners) {
          geometry_msgs::msg::Point pt;
          pt.x = static_cast<double>(c.x);
          pt.y = static_cast<double>(c.y);
          pt.z = 0.0;
          marker_msg.corners.push_back(pt);
        }

        double px, py, pz, qx, qy, qz, qw;
        aruco_ros_cv::rvecTvecToPositionQuat(m.rvec, m.tvec, px, py, pz, qx, qy, qz, qw);
        marker_msg.pose.pose.position.x = px;
        marker_msg.pose.pose.position.y = py;
        marker_msg.pose.pose.position.z = pz;
        marker_msg.pose.pose.orientation.x = qx;
        marker_msg.pose.pose.orientation.y = qy;
        marker_msg.pose.pose.orientation.z = qz;
        marker_msg.pose.pose.orientation.w = qw;

        marker_array.markers.push_back(marker_msg);
      }

      markers_pub_->publish(marker_array);
    }
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoRtCaptureNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
