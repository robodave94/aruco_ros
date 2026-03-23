/**
 * @file aruco_detect_service.cpp
 * @brief ROS2 service node for ArUco marker detection.
 *
 * Service: /aruco_detect (aruco_ros_cv_interfaces/srv/ArucoDetect)
 * Takes an image + dictionaries/sizes + camera intrinsics, returns 2D/3D visualizations
 * and a MarkerArray with detected poses.
 */

#include <rclcpp/rclcpp.hpp>

#if __has_include("cv_bridge/cv_bridge.hpp")
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include "aruco_ros_cv_interfaces/srv/aruco_detect.hpp"
#include "aruco_msgs/msg/marker.hpp"
#include "aruco_msgs/msg/marker_array.hpp"
#include "aruco_ros_cv/aruco_cv_utils.hpp"

class ArucoDetectServiceNode : public rclcpp::Node
{
public:
  ArucoDetectServiceNode()
  : Node("aruco_detect_service")
  {
    service_ = this->create_service<aruco_ros_cv_interfaces::srv::ArucoDetect>(
      "aruco_detect",
      std::bind(&ArucoDetectServiceNode::handle_detect, this,
      std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "ArucoDetect service ready on /aruco_detect");
  }

private:
  rclcpp::Service<aruco_ros_cv_interfaces::srv::ArucoDetect>::SharedPtr service_;

  void handle_detect(
    const std::shared_ptr<aruco_ros_cv_interfaces::srv::ArucoDetect::Request> request,
    std::shared_ptr<aruco_ros_cv_interfaces::srv::ArucoDetect::Response> response)
  {
    RCLCPP_INFO(this->get_logger(), "Received detection request");

    try {
      // Validate inputs
      if (request->dictionaries.size() != request->marker_sizes.size()) {
        RCLCPP_ERROR(this->get_logger(), "dictionaries and marker_sizes must have equal length");
        return;
      }
      if (request->dictionaries.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No dictionaries provided");
        return;
      }

      // Decode input image
      cv_bridge::CvImagePtr cv_ptr;
      try {
        cv_ptr = cv_bridge::toCvCopy(request->image, sensor_msgs::image_encodings::BGR8);
      } catch (const cv_bridge::Exception & e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
      }
      cv::Mat image = cv_ptr->image;

      // Build camera intrinsics from CameraInfo
      const auto & ci = request->camera_info;
      cv::Mat cam_mtx = aruco_ros_cv::buildCameraMatrix(ci.k[0], ci.k[4], ci.k[2], ci.k[5]);
      cv::Mat dist = cv::Mat(ci.d, true);

      // Detect markers across all dictionaries
      aruco_ros_cv::DetectionResult det = aruco_ros_cv::detectMultiDictMarkers(
        image, request->dictionaries, request->marker_sizes, cam_mtx, dist);

      RCLCPP_INFO(this->get_logger(), "Detected %zu markers", det.markers.size());

      // Generate 2D visualization
      cv::Mat vis_2d = aruco_ros_cv::draw2DVisualization(image, det, request->dictionaries);

      // Generate 3D visualization
      cv::Mat vis_3d = aruco_ros_cv::draw3DVisualization(
        image, det, cam_mtx, dist, request->dictionaries);

      // Encode visualization images
      auto stamp = this->now();

      cv_bridge::CvImage vis_2d_msg;
      vis_2d_msg.header.stamp = stamp;
      vis_2d_msg.encoding = sensor_msgs::image_encodings::BGR8;
      vis_2d_msg.image = vis_2d;
      response->visualization_2d = *vis_2d_msg.toImageMsg();

      cv_bridge::CvImage vis_3d_msg;
      vis_3d_msg.header.stamp = stamp;
      vis_3d_msg.encoding = sensor_msgs::image_encodings::BGR8;
      vis_3d_msg.image = vis_3d;
      response->visualization_3d = *vis_3d_msg.toImageMsg();

      // Build MarkerArray response
      response->markers.header.stamp = stamp;
      response->markers.header.frame_id = "camera";

      for (const auto & m : det.markers) {
        aruco_msgs::msg::Marker marker_msg;
        marker_msg.header.stamp = stamp;
        marker_msg.header.frame_id = "camera";
        marker_msg.id = static_cast<uint32_t>(m.id);
        marker_msg.confidence = 1.0;
        marker_msg.dictionary = m.dictionary_name;
        marker_msg.marker_size = m.marker_size;

        // Corners
        for (const auto & c : m.corners) {
          geometry_msgs::msg::Point pt;
          pt.x = static_cast<double>(c.x);
          pt.y = static_cast<double>(c.y);
          pt.z = 0.0;
          marker_msg.corners.push_back(pt);
        }

        // Pose
        double px, py, pz, qx, qy, qz, qw;
        aruco_ros_cv::rvecTvecToPositionQuat(m.rvec, m.tvec, px, py, pz, qx, qy, qz, qw);
        marker_msg.pose.pose.position.x = px;
        marker_msg.pose.pose.position.y = py;
        marker_msg.pose.pose.position.z = pz;
        marker_msg.pose.pose.orientation.x = qx;
        marker_msg.pose.pose.orientation.y = qy;
        marker_msg.pose.pose.orientation.z = qz;
        marker_msg.pose.pose.orientation.w = qw;

        response->markers.markers.push_back(marker_msg);
      }

      RCLCPP_INFO(this->get_logger(), "Response sent with %zu markers",
        response->markers.markers.size());
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "Detection error: %s", e.what());
    }
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoDetectServiceNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
