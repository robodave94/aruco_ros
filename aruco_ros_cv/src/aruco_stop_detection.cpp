/**
 * @file aruco_stop_detection.cpp
 * @brief One-shot CLI node that calls the 'stop_detection' service on aruco_action_server.
 *
 * Parameters:
 *   action_server_name  - Name of the aruco_action_server node (default "aruco_action_server")
 *   image_topic         - Camera image topic to stop detection on
 *   detection_id        - Optional id of the specific detection to stop
 *                          (default: empty, which stops every detection on the topic)
 */

#include <chrono>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "aruco_ros_cv_interfaces/srv/aruco_rt_stop.hpp"

class ArucoStopDetectionNode : public rclcpp::Node
{
public:
  using ArucoRtStop = aruco_ros_cv_interfaces::srv::ArucoRtStop;

  ArucoStopDetectionNode()
  : Node("aruco_stop_detection")
  {
    this->declare_parameter<std::string>("action_server_name", "aruco_action_server");
    this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
    this->declare_parameter<std::string>("detection_id", "");
  }

  bool run()
  {
    std::string action_server_name = this->get_parameter("action_server_name").as_string();
    std::string image_topic = this->get_parameter("image_topic").as_string();
    std::string detection_id = this->get_parameter("detection_id").as_string();

    std::string service_name = "/" + action_server_name + "/stop_detection";
    auto client = this->create_client<ArucoRtStop>(service_name);

    RCLCPP_INFO(this->get_logger(), "Waiting for service: %s", service_name.c_str());
    if (!client->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_ERROR(this->get_logger(), "Service not available: %s", service_name.c_str());
      return false;
    }

    auto request = std::make_shared<ArucoRtStop::Request>();
    request->image_topic = image_topic;
    request->detection_id = detection_id;

    if (detection_id.empty()) {
      RCLCPP_INFO(this->get_logger(), "Requesting stop of ALL detections on %s",
        image_topic.c_str());
    } else {
      RCLCPP_INFO(this->get_logger(), "Requesting stop of detection '%s' on %s",
        detection_id.c_str(), image_topic.c_str());
    }

    auto future = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) !=
      rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to call service: %s", service_name.c_str());
      return false;
    }

    auto response = future.get();
    if (!response->success) {
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      return false;
    }

    RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
    return true;
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoStopDetectionNode>();
  bool ok = node->run();
  rclcpp::shutdown();
  return ok ? 0 : 1;
}
