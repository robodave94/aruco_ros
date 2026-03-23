/**
 * @file aruco_action_server.cpp
 * @brief ROS2 action server for dynamic multi-camera ArUco marker detection.
 *
 * Action: ~/start_detection  (ArucoRtStart)  - Start RT detection on a camera topic
 * Service: ~/stop_detection  (ArucoRtStop)   - Stop RT detection on a camera topic
 * Diagnostics: ~/diagnostics (diagnostic_msgs/DiagnosticArray)
 *
 * Features:
 *   - Dynamically spawn/stop processing threads per camera topic
 *   - Max thread limit (default 10, configurable via 'max_processing_threads' param)
 *   - Watchdog: warns if no image received for 60s, then shuts down that thread
 *   - Each thread publishes <topic>/arucofeed and <topic>/arucofeed/markers
 */

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#if __has_include("cv_bridge/cv_bridge.hpp")
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>

#include "aruco_ros_cv_interfaces/action/aruco_rt_start.hpp"
#include "aruco_ros_cv_interfaces/srv/aruco_rt_stop.hpp"
#include "aruco_msgs/msg/marker.hpp"
#include "aruco_msgs/msg/marker_array.hpp"
#include "aruco_ros_cv/aruco_cv_utils.hpp"

using ArucoRtStart = aruco_ros_cv_interfaces::action::ArucoRtStart;
using ArucoRtStop = aruco_ros_cv_interfaces::srv::ArucoRtStop;
using GoalHandleStart = rclcpp_action::ServerGoalHandle<ArucoRtStart>;

/** Per-topic processing context. */
struct ProcessingThread
{
  std::string image_topic;
  std::vector<std::string> dictionaries;
  std::vector<double> marker_sizes;
  cv::Mat cam_mtx;
  cv::Mat dist_coeffs;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feed_pub;
  rclcpp::Publisher<aruco_msgs::msg::MarkerArray>::SharedPtr markers_pub;

  std::atomic<bool> active{true};
  std::chrono::steady_clock::time_point last_image_time;
  std::mutex time_mutex;
};

class ArucoActionServerNode : public rclcpp::Node
{
public:
  ArucoActionServerNode()
  : Node("aruco_action_server")
  {
    this->declare_parameter<int>("max_processing_threads", 10);
    max_threads_ = static_cast<size_t>(this->get_parameter("max_processing_threads").as_int());

    // Action server for starting detection
    action_server_ = rclcpp_action::create_server<ArucoRtStart>(
      this, "~/start_detection",
      std::bind(&ArucoActionServerNode::handle_goal, this,
      std::placeholders::_1, std::placeholders::_2),
      std::bind(&ArucoActionServerNode::handle_cancel, this, std::placeholders::_1),
      std::bind(&ArucoActionServerNode::handle_accepted, this, std::placeholders::_1));

    // Service for stopping detection
    stop_service_ = this->create_service<ArucoRtStop>(
      "~/stop_detection",
      std::bind(&ArucoActionServerNode::handle_stop, this,
      std::placeholders::_1, std::placeholders::_2));

    // Diagnostics publisher
    diag_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "~/diagnostics", 10);

    // Diagnostics timer (publish every 2 seconds)
    diag_timer_ = this->create_wall_timer(
      std::chrono::seconds(2),
      std::bind(&ArucoActionServerNode::publish_diagnostics, this));

    // Watchdog timer (check every 5 seconds)
    watchdog_timer_ = this->create_wall_timer(
      std::chrono::seconds(5),
      std::bind(&ArucoActionServerNode::watchdog_check, this));

    RCLCPP_INFO(this->get_logger(), "ArUco action server started (max threads: %zu)",
      max_threads_);
  }

private:
  size_t max_threads_;
  std::mutex threads_mutex_;
  std::map<std::string, std::shared_ptr<ProcessingThread>> threads_;

  rclcpp_action::Server<ArucoRtStart>::SharedPtr action_server_;
  rclcpp::Service<ArucoRtStop>::SharedPtr stop_service_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;

  // -------------------------------------------------------------------------
  // Action server callbacks
  // -------------------------------------------------------------------------

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const ArucoRtStart::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received start request for topic: %s",
      goal->image_topic.c_str());

    std::lock_guard<std::mutex> lock(threads_mutex_);

    // Check if already running
    if (threads_.count(goal->image_topic)) {
      RCLCPP_WARN(this->get_logger(), "Detection already running on topic: %s",
        goal->image_topic.c_str());
      return rclcpp_action::GoalResponse::REJECT;
    }

    // Check thread limit
    if (threads_.size() >= max_threads_) {
      RCLCPP_ERROR(this->get_logger(),
        "The amount of RGB processing RT limits has been reached (%zu). "
        "Please shutdown a feed before adding another.", max_threads_);
      return rclcpp_action::GoalResponse::REJECT;
    }

    // Validate inputs
    if (goal->dictionaries.size() != goal->marker_sizes.size()) {
      RCLCPP_ERROR(this->get_logger(), "dictionaries and marker_sizes must have equal length");
      return rclcpp_action::GoalResponse::REJECT;
    }

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleStart>)
  {
    RCLCPP_INFO(this->get_logger(), "Received cancel request");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleStart> goal_handle)
  {
    // Execute in a detached thread to avoid blocking the action server
    std::thread([this, goal_handle]() {
      execute_start(goal_handle);
    }).detach();
  }

  void execute_start(const std::shared_ptr<GoalHandleStart> goal_handle)
  {
    auto goal = goal_handle->get_goal();
    auto result = std::make_shared<ArucoRtStart::Result>();

    try {
      // Validate dictionaries
      for (const auto & d : goal->dictionaries) {
        aruco_ros_cv::dictionaryFromString(d);
      }

      // Create processing thread context
      auto ctx = std::make_shared<ProcessingThread>();
      ctx->image_topic = goal->image_topic;
      ctx->dictionaries = goal->dictionaries;
      ctx->marker_sizes = goal->marker_sizes;
      const auto & ci = goal->camera_info;
      ctx->cam_mtx = aruco_ros_cv::buildCameraMatrix(ci.k[0], ci.k[4], ci.k[2], ci.k[5]);
      ctx->dist_coeffs = cv::Mat(ci.d, true);
      ctx->last_image_time = std::chrono::steady_clock::now();

      // Create publishers
      std::string feed_topic = goal->image_topic + "/arucofeed";
      std::string markers_topic = goal->image_topic + "/arucofeed/markers";

      ctx->feed_pub = this->create_publisher<sensor_msgs::msg::Image>(feed_topic, 10);
      ctx->markers_pub = this->create_publisher<aruco_msgs::msg::MarkerArray>(
        markers_topic, 10);

      // Create subscriber
      ctx->image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        goal->image_topic, rclcpp::SensorDataQoS(),
        [this, ctx](const sensor_msgs::msg::Image::ConstSharedPtr & msg) {
          process_image(ctx, msg);
        });

      // Register
      {
        std::lock_guard<std::mutex> lock(threads_mutex_);
        threads_[goal->image_topic] = ctx;
      }

      // Send feedback periodically while goal is active
      auto feedback = std::make_shared<ArucoRtStart::Feedback>();
      feedback->status = "Detection started on " + goal->image_topic;

      {
        std::lock_guard<std::mutex> lock(threads_mutex_);
        feedback->active_threads = static_cast<uint32_t>(threads_.size());
      }
      goal_handle->publish_feedback(feedback);

      // Wait briefly then succeed — the thread continues running independently
      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      result->success = true;
      result->message = "Detection started on " + goal->image_topic;
      goal_handle->succeed(result);

      RCLCPP_INFO(this->get_logger(), "Detection thread started for: %s",
        goal->image_topic.c_str());

    } catch (const std::exception & e) {
      result->success = false;
      result->message = std::string("Failed to start detection: ") + e.what();
      goal_handle->abort(result);
      RCLCPP_ERROR(this->get_logger(), "%s", result->message.c_str());
    }
  }

  // -------------------------------------------------------------------------
  // Stop service
  // -------------------------------------------------------------------------

  void handle_stop(
    const std::shared_ptr<ArucoRtStop::Request> request,
    std::shared_ptr<ArucoRtStop::Response> response)
  {
    RCLCPP_INFO(this->get_logger(), "Stop request for topic: %s", request->image_topic.c_str());

    std::lock_guard<std::mutex> lock(threads_mutex_);
    auto it = threads_.find(request->image_topic);
    if (it == threads_.end()) {
      response->success = false;
      response->message = "No active detection on topic: " + request->image_topic;
      RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    shutdown_thread(it->second);
    threads_.erase(it);

    response->success = true;
    response->message = "Detection stopped on " + request->image_topic;
    RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
  }

  // -------------------------------------------------------------------------
  // Image processing
  // -------------------------------------------------------------------------

  void process_image(
    std::shared_ptr<ProcessingThread> ctx,
    const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    if (!ctx->active) return;

    // Update last image timestamp
    {
      std::lock_guard<std::mutex> lock(ctx->time_mutex);
      ctx->last_image_time = std::chrono::steady_clock::now();
    }

    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "cv_bridge exception on %s: %s", ctx->image_topic.c_str(), e.what());
      return;
    }

    cv::Mat image = cv_ptr->image;
    aruco_ros_cv::DetectionResult det;
    try {
      det = aruco_ros_cv::detectMultiDictMarkers(
        image, ctx->dictionaries, ctx->marker_sizes, ctx->cam_mtx, ctx->dist_coeffs);
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "Detection error on %s: %s", ctx->image_topic.c_str(), e.what());
      return;
    }

    auto stamp = msg->header.stamp;

    // Publish visualized feed
    if (ctx->feed_pub->get_subscription_count() > 0) {
      cv::Mat vis = aruco_ros_cv::draw2DVisualization(image, det, ctx->dictionaries);
      for (const auto & m : det.markers) {
        float axis_len = static_cast<float>(m.marker_size) * 0.75f;
        cv::drawFrameAxes(vis, ctx->cam_mtx, ctx->dist_coeffs, m.rvec, m.tvec, axis_len);
      }
      cv_bridge::CvImage out;
      out.header.stamp = stamp;
      out.header.frame_id = msg->header.frame_id;
      out.encoding = sensor_msgs::image_encodings::BGR8;
      out.image = vis;
      ctx->feed_pub->publish(*out.toImageMsg());
    }

    // Publish marker array
    if (ctx->markers_pub->get_subscription_count() > 0) {
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

      ctx->markers_pub->publish(marker_array);
    }
  }

  // -------------------------------------------------------------------------
  // Diagnostics
  // -------------------------------------------------------------------------

  void publish_diagnostics()
  {
    diagnostic_msgs::msg::DiagnosticArray diag_array;
    diag_array.header.stamp = this->now();

    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "aruco_action_server";

    std::lock_guard<std::mutex> lock(threads_mutex_);
    status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    status.message = "Active detection threads: " + std::to_string(threads_.size()) +
      "/" + std::to_string(max_threads_);

    diagnostic_msgs::msg::KeyValue kv_count;
    kv_count.key = "active_threads";
    kv_count.value = std::to_string(threads_.size());
    status.values.push_back(kv_count);

    diagnostic_msgs::msg::KeyValue kv_limit;
    kv_limit.key = "max_threads";
    kv_limit.value = std::to_string(max_threads_);
    status.values.push_back(kv_limit);

    for (const auto & [topic, ctx] : threads_) {
      diagnostic_msgs::msg::KeyValue kv;
      kv.key = "topic";
      kv.value = topic;
      status.values.push_back(kv);
    }

    diag_array.status.push_back(status);
    diag_pub_->publish(diag_array);
  }

  // -------------------------------------------------------------------------
  // Watchdog
  // -------------------------------------------------------------------------

  void watchdog_check()
  {
    auto now = std::chrono::steady_clock::now();
    std::vector<std::string> to_remove;

    std::lock_guard<std::mutex> lock(threads_mutex_);
    for (auto & [topic, ctx] : threads_) {
      std::chrono::steady_clock::time_point last;
      {
        std::lock_guard<std::mutex> tlock(ctx->time_mutex);
        last = ctx->last_image_time;
      }

      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last);

      if (elapsed.count() > 60) {
        RCLCPP_ERROR(this->get_logger(),
          "Camera feed for aruco processing '%s' not receiving data. "
          "Timeout exceeded (60s). Shutting down processing thread.", topic.c_str());
        shutdown_thread(ctx);
        to_remove.push_back(topic);
      } else if (elapsed.count() > 30) {
        RCLCPP_WARN(this->get_logger(),
          "Camera feed for aruco processing '%s' not receiving data. "
          "Shutting down in %ld seconds.",
          topic.c_str(), 60 - elapsed.count());

        // Also publish on diagnostics
        diagnostic_msgs::msg::DiagnosticArray diag_array;
        diag_array.header.stamp = this->now();
        diagnostic_msgs::msg::DiagnosticStatus status;
        status.name = "aruco_action_server/watchdog";
        status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
        status.message = "Camera feed '" + topic + "' not receiving data. " +
          "Shutting down in " + std::to_string(60 - elapsed.count()) + " seconds.";
        diag_array.status.push_back(status);
        diag_pub_->publish(diag_array);
      }
    }

    for (const auto & topic : to_remove) {
      threads_.erase(topic);
    }
  }

  void shutdown_thread(std::shared_ptr<ProcessingThread> ctx)
  {
    ctx->active = false;
    ctx->image_sub.reset();
    ctx->feed_pub.reset();
    ctx->markers_pub.reset();
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoActionServerNode>();
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
