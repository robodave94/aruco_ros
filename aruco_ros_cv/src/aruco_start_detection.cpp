/**
 * @file aruco_start_detection.cpp
 * @brief One-shot CLI node that sends a 'start_detection' goal to aruco_action_server.
 *
 * Reads camera intrinsics from a YAML file and sends an ArucoRtStart goal built from
 * simple parameters, so users don't have to hand-write a `ros2 action send_goal` call.
 *
 * Parameters:
 *   action_server_name  - Name of the aruco_action_server node (default "aruco_action_server")
 *   image_topic         - Camera image topic to start detection on
 *   dictionaries        - List of ArUco dictionary names
 *   marker_sizes        - List of marker sizes in meters, parallel to dictionaries
 *   camera_params_file  - Path to an OpenCV YAML camera intrinsics file
 *   detection_id        - Optional id for this detection (default: auto from dictionaries)
 */

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <sensor_msgs/msg/camera_info.hpp>

#include "aruco_ros_cv_interfaces/action/aruco_rt_start.hpp"
#include "aruco_ros_cv/aruco_cv_utils.hpp"

class ArucoStartDetectionNode : public rclcpp::Node
{
public:
  using ArucoRtStart = aruco_ros_cv_interfaces::action::ArucoRtStart;

  ArucoStartDetectionNode()
  : Node("aruco_start_detection")
  {
    this->declare_parameter<std::string>("action_server_name", "aruco_action_server");
    this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    this->declare_parameter<std::vector<std::string>>("dictionaries", {"DICT_6X6_250"});
    this->declare_parameter<std::vector<double>>("marker_sizes", {0.05});
    this->declare_parameter<std::string>("camera_params_file", "");
    this->declare_parameter<std::string>("detection_id", "");
    this->declare_parameter<std::string>("reference_frame", "");
    this->declare_parameter<bool>("zoom_enabled", false);
    this->declare_parameter<double>("zoom_center_x", -1.0);
    this->declare_parameter<double>("zoom_center_y", -1.0);
    this->declare_parameter<double>("zoom_width", 500.0);
    this->declare_parameter<double>("zoom_height", 500.0);
    this->declare_parameter<double>("zoom_upscale", 1.5);
    this->declare_parameter<bool>("zoom_rescale_distortion", false);
    this->declare_parameter<bool>("zoom_debug_enabled", false);
  }

  bool run()
  {
    std::string action_server_name = this->get_parameter("action_server_name").as_string();
    std::string image_topic = this->get_parameter("image_topic").as_string();
    std::vector<std::string> dictionaries = this->get_parameter("dictionaries").as_string_array();
    std::vector<double> marker_sizes = this->get_parameter("marker_sizes").as_double_array();
    std::string camera_params_file = this->get_parameter("camera_params_file").as_string();
    std::string detection_id = this->get_parameter("detection_id").as_string();
    std::string reference_frame = this->get_parameter("reference_frame").as_string();
    bool zoom_enabled = this->get_parameter("zoom_enabled").as_bool();
    double zoom_center_x = this->get_parameter("zoom_center_x").as_double();
    double zoom_center_y = this->get_parameter("zoom_center_y").as_double();
    double zoom_width = this->get_parameter("zoom_width").as_double();
    double zoom_height = this->get_parameter("zoom_height").as_double();
    double zoom_upscale = this->get_parameter("zoom_upscale").as_double();
    bool zoom_rescale_distortion = this->get_parameter("zoom_rescale_distortion").as_bool();
    bool zoom_debug_enabled = this->get_parameter("zoom_debug_enabled").as_bool();

    if (dictionaries.empty() || dictionaries.size() != marker_sizes.size()) {
      RCLCPP_ERROR(this->get_logger(),
        "dictionaries must be non-empty and have the same length as marker_sizes");
      return false;
    }
    if (camera_params_file.empty()) {
      RCLCPP_ERROR(this->get_logger(), "camera_params_file parameter is required");
      return false;
    }

    cv::Mat cam_mtx, dist_coeffs;
    if (!aruco_ros_cv::loadCameraIntrinsicsFromYAML(camera_params_file, cam_mtx, dist_coeffs)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load camera params: %s",
        camera_params_file.c_str());
      return false;
    }

    sensor_msgs::msg::CameraInfo camera_info;
    camera_info.k[0] = cam_mtx.at<double>(0, 0);
    camera_info.k[2] = cam_mtx.at<double>(0, 2);
    camera_info.k[4] = cam_mtx.at<double>(1, 1);
    camera_info.k[5] = cam_mtx.at<double>(1, 2);
    camera_info.k[8] = 1.0;
    camera_info.d.resize(static_cast<size_t>(dist_coeffs.total()));
    for (size_t i = 0; i < camera_info.d.size(); ++i) {
      camera_info.d[i] = dist_coeffs.at<double>(static_cast<int>(i));
    }

    std::string action_name = "/" + action_server_name + "/start_detection";
    auto client = rclcpp_action::create_client<ArucoRtStart>(this, action_name);

    RCLCPP_INFO(this->get_logger(), "Waiting for action server: %s", action_name.c_str());
    if (!client->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(this->get_logger(), "Action server not available: %s", action_name.c_str());
      return false;
    }

    ArucoRtStart::Goal goal;
    goal.image_topic = image_topic;
    goal.dictionaries = dictionaries;
    goal.marker_sizes = marker_sizes;
    goal.camera_info = camera_info;
    goal.detection_id = detection_id;
    goal.reference_frame = reference_frame;
    goal.zoom_enabled = zoom_enabled;
    goal.zoom_center_x = zoom_center_x;
    goal.zoom_center_y = zoom_center_y;
    goal.zoom_width = zoom_width;
    goal.zoom_height = zoom_height;
    goal.zoom_upscale = zoom_upscale;
    goal.zoom_rescale_distortion = zoom_rescale_distortion;
    goal.zoom_debug_enabled = zoom_debug_enabled;

    RCLCPP_INFO(this->get_logger(), "Requesting detection on %s (server: %s)",
      image_topic.c_str(), action_server_name.c_str());

    auto goal_handle_future = client->async_send_goal(
      goal, rclcpp_action::Client<ArucoRtStart>::SendGoalOptions());
    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), goal_handle_future) !=
      rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to send goal to %s", action_name.c_str());
      return false;
    }

    auto goal_handle = goal_handle_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(this->get_logger(),
        "Goal was rejected — a detection may already be running on this topic/id, "
        "or the max_processing_threads limit has been reached");
      return false;
    }

    auto result_future = client->async_get_result(goal_handle);
    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), result_future) !=
      rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to get result from %s", action_name.c_str());
      return false;
    }

    auto wrapped_result = result_future.get();
    if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED || !wrapped_result.result) {
      RCLCPP_ERROR(this->get_logger(), "Detection failed to start: %s",
        wrapped_result.result ? wrapped_result.result->message.c_str() : "unknown error");
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "%s", wrapped_result.result->message.c_str());
    RCLCPP_INFO(this->get_logger(),
      "To stop this detection: ros2 run aruco_ros_cv aruco_stop_detection --ros-args "
      "-p action_server_name:=%s -p image_topic:=%s -p detection_id:=%s",
      action_server_name.c_str(), image_topic.c_str(),
      wrapped_result.result->detection_id.c_str());
    return true;
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoStartDetectionNode>();
  bool ok = node->run();
  rclcpp::shutdown();
  return ok ? 0 : 1;
}
