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
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/header.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
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
    this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
    this->declare_parameter<std::string>("reference_frame", "");

    // Get parameters
    image_topic_ = this->get_parameter("image_topic").as_string();
    dictionaries_ = this->get_parameter("dictionaries").as_string_array();
    marker_sizes_ = this->get_parameter("marker_sizes").as_double_array();
    camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();
    reference_frame_ = this->get_parameter("reference_frame").as_string();

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

    // Subscribe to camera info (transient_local to receive latched messages)
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::QoS(1).transient_local(),
      std::bind(&ArucoRtCaptureNode::camera_info_callback, this, std::placeholders::_1));

    // TF broadcaster + listener for marker transforms
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

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
  std::string camera_info_topic_;
  std::vector<std::string> dictionaries_;
  std::vector<double> marker_sizes_;
  cv::Mat cam_mtx_;
  cv::Mat dist_coeffs_;
  bool camera_info_received_ = false;

  std::string reference_frame_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feed_pub_;
  rclcpp::Publisher<aruco_msgs::msg::MarkerArray>::SharedPtr markers_pub_;

  void camera_info_callback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
  {
    if (camera_info_received_) return;
    cam_mtx_ = aruco_ros_cv::buildCameraMatrix(msg->k[0], msg->k[4], msg->k[2], msg->k[5]);
    dist_coeffs_ = cv::Mat(msg->d, true);
    camera_info_received_ = true;
    RCLCPP_INFO(this->get_logger(),
      "Camera intrinsics received from %s (fx=%.2f fy=%.2f cx=%.2f cy=%.2f, D size=%zu)",
      camera_info_topic_.c_str(), msg->k[0], msg->k[4], msg->k[2], msg->k[5], msg->d.size());
  }

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    if (!camera_info_received_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "Waiting for camera_info on %s", camera_info_topic_.c_str());
      return;
    }
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

    broadcast_marker_tfs(det, msg->header);
  }

  /** Broadcast one TF per detected marker; transform into reference_frame via tf2. */
  void broadcast_marker_tfs(
    const aruco_ros_cv::DetectionResult & det,
    const std_msgs::msg::Header & header)
  {
    if (det.markers.empty()) return;

    const std::string & camera_frame = header.frame_id;
    std::string parent_frame =
      reference_frame_.empty() ? camera_frame : reference_frame_;
    bool need_transform = (parent_frame != camera_frame);

    auto names = aruco_ros_cv::buildUniqueMarkerFrameNames(det);

    std::vector<geometry_msgs::msg::TransformStamped> transforms;
    transforms.reserve(det.markers.size());

    for (size_t i = 0; i < det.markers.size(); ++i) {
      const auto & m = det.markers[i];
      double px, py, pz, qx, qy, qz, qw;
      aruco_ros_cv::rvecTvecToPositionQuat(m.rvec, m.tvec, px, py, pz, qx, qy, qz, qw);

      geometry_msgs::msg::PoseStamped pose_in;
      pose_in.header = header;
      pose_in.pose.position.x = px;
      pose_in.pose.position.y = py;
      pose_in.pose.position.z = pz;
      pose_in.pose.orientation.x = qx;
      pose_in.pose.orientation.y = qy;
      pose_in.pose.orientation.z = qz;
      pose_in.pose.orientation.w = qw;

      std::string frame_parent = parent_frame;
      geometry_msgs::msg::Pose pose_out = pose_in.pose;

      if (need_transform) {
        try {
          geometry_msgs::msg::PoseStamped transformed =
            tf_buffer_->transform(pose_in, parent_frame, tf2::durationFromSec(0.05));
          pose_out = transformed.pose;
        } catch (const tf2::TransformException & ex) {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "TF %s->%s failed: %s (publishing in camera frame)",
            camera_frame.c_str(), parent_frame.c_str(), ex.what());
          frame_parent = camera_frame;
          pose_out = pose_in.pose;
        }
      }

      geometry_msgs::msg::TransformStamped t;
      t.header.stamp = header.stamp;
      t.header.frame_id = frame_parent;
      t.child_frame_id = names[i];
      t.transform.translation.x = pose_out.position.x;
      t.transform.translation.y = pose_out.position.y;
      t.transform.translation.z = pose_out.position.z;
      t.transform.rotation = pose_out.orientation;
      transforms.push_back(t);
    }

    tf_broadcaster_->sendTransform(transforms);
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
