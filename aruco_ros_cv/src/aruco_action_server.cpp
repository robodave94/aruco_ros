/**
 * @file aruco_action_server.cpp
 * @brief ROS2 action server for dynamic multi-camera, multi-detection ArUco marker detection.
 *
 * Action: ~/start_detection  (ArucoRtStart)  - Start RT detection on a camera topic
 * Service: ~/stop_detection  (ArucoRtStop)   - Stop RT detection on a camera topic
 * Diagnostics: ~/diagnostics (diagnostic_msgs/DiagnosticArray)
 *
 * Features:
 *   - Dynamically spawn/stop processing threads per camera topic
 *   - Multiple detections (different dictionaries/ids) can share one camera topic's
 *     thread/subscription — only distinct topics count against 'max_processing_threads'
 *   - Max thread limit (default 10, configurable via 'max_processing_threads' param)
 *   - Watchdog: warns if no image received for 60s, then shuts down that topic's thread
 *   - Each detection publishes <topic>/arucofeed/<detection_id> and
 *     <topic>/arucofeed/markers/<detection_id>
 */

#include <atomic>
#include <cctype>
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

#include <std_msgs/msg/header.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "aruco_ros_cv_interfaces/action/aruco_rt_start.hpp"
#include "aruco_ros_cv_interfaces/srv/aruco_rt_stop.hpp"
#include "aruco_msgs/msg/marker.hpp"
#include "aruco_msgs/msg/marker_array.hpp"
#include "aruco_ros_cv/aruco_cv_utils.hpp"

using ArucoRtStart = aruco_ros_cv_interfaces::action::ArucoRtStart;
using ArucoRtStop = aruco_ros_cv_interfaces::srv::ArucoRtStop;
using GoalHandleStart = rclcpp_action::ServerGoalHandle<ArucoRtStart>;

/** Replace any character unsafe for a ROS topic token with '_'. */
inline std::string sanitizeToken(const std::string & s)
{
  std::string out = s;
  for (auto & c : out) {
    if (!std::isalnum(static_cast<unsigned char>(c))) {
      c = '_';
    }
  }
  return out;
}

/** Resolve the id used to key/name a detection: explicit id, or dictionaries joined by '_'. */
inline std::string resolveDetectionId(
  const std::string & requested_id,
  const std::vector<std::string> & dictionaries)
{
  if (!requested_id.empty()) {
    return sanitizeToken(requested_id);
  }
  std::string joined;
  for (size_t i = 0; i < dictionaries.size(); ++i) {
    if (i > 0) joined += "_";
    joined += dictionaries[i];
  }
  return sanitizeToken(joined);
}

/** A single detection running on a camera topic's shared thread. */
struct DetectionJob
{
  std::string detection_id;
  std::vector<std::string> dictionaries;
  std::vector<double> marker_sizes;
  cv::Mat cam_mtx;
  cv::Mat dist_coeffs;

  std::string reference_frame;   // TF parent; empty => image header frame_id
  bool zoom_enabled{false};
  bool zoom_debug_enabled{false};
  aruco_ros_cv::ZoomConfig zoom;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr feed_pub;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr zoom_debug_pub;
  rclcpp::Publisher<aruco_msgs::msg::MarkerArray>::SharedPtr markers_pub;
};

/** Per-topic processing context — one subscription shared by all of its detection jobs. */
struct CameraThread
{
  std::string image_topic;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;

  std::atomic<bool> active{true};
  std::chrono::steady_clock::time_point last_image_time;
  std::mutex time_mutex;

  std::map<std::string, std::shared_ptr<DetectionJob>> jobs;
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

    // TF broadcaster + listener for marker transforms
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    RCLCPP_INFO(this->get_logger(), "ArUco action server started (max threads: %zu)",
      max_threads_);
  }

private:
  size_t max_threads_;
  std::mutex threads_mutex_;
  std::map<std::string, std::shared_ptr<CameraThread>> camera_threads_;

  rclcpp_action::Server<ArucoRtStart>::SharedPtr action_server_;
  rclcpp::Service<ArucoRtStop>::SharedPtr stop_service_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // -------------------------------------------------------------------------
  // Action server callbacks
  // -------------------------------------------------------------------------

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const ArucoRtStart::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received start request for topic: %s",
      goal->image_topic.c_str());

    // Validate inputs
    if (goal->dictionaries.empty() ||
      goal->dictionaries.size() != goal->marker_sizes.size())
    {
      RCLCPP_ERROR(this->get_logger(),
        "dictionaries must be non-empty and have the same length as marker_sizes");
      return rclcpp_action::GoalResponse::REJECT;
    }

    std::string detection_id = resolveDetectionId(goal->detection_id, goal->dictionaries);

    std::lock_guard<std::mutex> lock(threads_mutex_);

    auto it = camera_threads_.find(goal->image_topic);
    if (it != camera_threads_.end()) {
      // Topic already has a processing thread — joining it doesn't affect the thread limit.
      if (it->second->jobs.count(detection_id)) {
        RCLCPP_WARN(this->get_logger(), "Detection '%s' already running on topic: %s",
          detection_id.c_str(), goal->image_topic.c_str());
        return rclcpp_action::GoalResponse::REJECT;
      }
    } else if (camera_threads_.size() >= max_threads_) {
      // This would spawn a brand-new processing thread — enforce the thread limit.
      RCLCPP_ERROR(this->get_logger(),
        "The amount of RGB processing RT limits has been reached (%zu). "
        "Please shutdown a feed before adding another.", max_threads_);
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

      std::string detection_id = resolveDetectionId(goal->detection_id, goal->dictionaries);

      // Find or create the camera thread for this topic
      std::shared_ptr<CameraThread> cam;
      bool created_thread = false;
      {
        std::lock_guard<std::mutex> lock(threads_mutex_);
        auto it = camera_threads_.find(goal->image_topic);
        if (it != camera_threads_.end()) {
          cam = it->second;
          if (cam->jobs.count(detection_id)) {
            throw std::invalid_argument(
                    "Detection '" + detection_id + "' already running on topic: " +
                    goal->image_topic);
          }
        } else {
          if (camera_threads_.size() >= max_threads_) {
            throw std::invalid_argument(
                    "Max processing threads (" + std::to_string(max_threads_) + ") reached");
          }
          cam = std::make_shared<CameraThread>();
          cam->image_topic = goal->image_topic;
          cam->last_image_time = std::chrono::steady_clock::now();
          created_thread = true;
        }
      }

      // Build the new detection job
      auto job = std::make_shared<DetectionJob>();
      job->detection_id = detection_id;
      job->dictionaries = goal->dictionaries;
      job->marker_sizes = goal->marker_sizes;
      const auto & ci = goal->camera_info;
      job->cam_mtx = aruco_ros_cv::buildCameraMatrix(ci.k[0], ci.k[4], ci.k[2], ci.k[5]);
      job->dist_coeffs = cv::Mat(ci.d, true);

      job->reference_frame = goal->reference_frame;
      job->zoom_enabled = goal->zoom_enabled;
      job->zoom_debug_enabled = goal->zoom_debug_enabled;
      job->zoom.center_x = goal->zoom_center_x;
      job->zoom.center_y = goal->zoom_center_y;
      job->zoom.width = (goal->zoom_width > 0.0) ? goal->zoom_width : 500.0;
      job->zoom.height = (goal->zoom_height > 0.0) ? goal->zoom_height : 500.0;
      job->zoom.upscale = (goal->zoom_upscale > 0.0) ? goal->zoom_upscale : 1.5;
      job->zoom.rescale_distortion = goal->zoom_rescale_distortion;

      std::string feed_topic = goal->image_topic + "/arucofeed/" + detection_id;
      std::string markers_topic = goal->image_topic + "/arucofeed/markers/" + detection_id;
      job->feed_pub = this->create_publisher<sensor_msgs::msg::Image>(feed_topic, 10);
      job->markers_pub = this->create_publisher<aruco_msgs::msg::MarkerArray>(
        markers_topic, 10);

      if (job->zoom_enabled && job->zoom_debug_enabled) {
        std::string zoom_debug_topic =
          goal->image_topic + "/arucofeed/zoomdebug/" + detection_id;
        job->zoom_debug_pub =
          this->create_publisher<sensor_msgs::msg::Image>(zoom_debug_topic, 10);
      }

      // Only a brand-new camera thread needs its own subscription
      if (created_thread) {
        cam->image_sub = this->create_subscription<sensor_msgs::msg::Image>(
          goal->image_topic, rclcpp::SensorDataQoS(),
          [this, cam](const sensor_msgs::msg::Image::ConstSharedPtr & msg) {
            process_image(cam, msg);
          });
      }

      // Register
      size_t active_thread_count;
      {
        std::lock_guard<std::mutex> lock(threads_mutex_);
        cam->jobs[detection_id] = job;
        if (created_thread) {
          camera_threads_[goal->image_topic] = cam;
        }
        active_thread_count = camera_threads_.size();
      }

      // Send feedback periodically while goal is active
      auto feedback = std::make_shared<ArucoRtStart::Feedback>();
      feedback->status = "Detection '" + detection_id + "' started on " + goal->image_topic;
      feedback->active_threads = static_cast<uint32_t>(active_thread_count);
      goal_handle->publish_feedback(feedback);

      // Wait briefly then succeed — the thread continues running independently
      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      result->success = true;
      result->detection_id = detection_id;
      result->message = "Detection '" + detection_id + "' started on " + goal->image_topic;
      goal_handle->succeed(result);

      RCLCPP_INFO(this->get_logger(), "Detection '%s' started for: %s",
        detection_id.c_str(), goal->image_topic.c_str());

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
    std::lock_guard<std::mutex> lock(threads_mutex_);
    auto it = camera_threads_.find(request->image_topic);
    if (it == camera_threads_.end()) {
      response->success = false;
      response->message = "No active detection on topic: " + request->image_topic;
      RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    auto cam = it->second;

    if (request->detection_id.empty()) {
      // No id given — stop every detection on this topic and tear down its thread.
      RCLCPP_INFO(this->get_logger(), "Stop request for all detections on topic: %s",
        request->image_topic.c_str());
      shutdown_thread(cam);
      camera_threads_.erase(it);
      response->success = true;
      response->message = "All detections stopped on " + request->image_topic;
      RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Stop request for detection '%s' on topic: %s",
      request->detection_id.c_str(), request->image_topic.c_str());

    auto job_it = cam->jobs.find(request->detection_id);
    if (job_it == cam->jobs.end()) {
      response->success = false;
      response->message = "No active detection '" + request->detection_id +
        "' on topic: " + request->image_topic;
      RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    job_it->second->feed_pub.reset();
    job_it->second->markers_pub.reset();
    job_it->second->zoom_debug_pub.reset();
    cam->jobs.erase(job_it);

    response->success = true;
    response->message = "Detection '" + request->detection_id + "' stopped on " +
      request->image_topic;
    RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());

    if (cam->jobs.empty()) {
      // Last detection on this topic — no reason to keep the subscription alive.
      shutdown_thread(cam);
      camera_threads_.erase(it);
      RCLCPP_INFO(this->get_logger(),
        "No detections remain on %s — processing thread shut down",
        request->image_topic.c_str());
    }
  }

  // -------------------------------------------------------------------------
  // Image processing
  // -------------------------------------------------------------------------

  void process_image(
    std::shared_ptr<CameraThread> cam,
    const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    if (!cam->active) return;

    // Update last image timestamp
    {
      std::lock_guard<std::mutex> lock(cam->time_mutex);
      cam->last_image_time = std::chrono::steady_clock::now();
    }

    // Snapshot the current jobs so detection/publishing doesn't hold the shared lock
    std::vector<std::shared_ptr<DetectionJob>> jobs_snapshot;
    {
      std::lock_guard<std::mutex> lock(threads_mutex_);
      jobs_snapshot.reserve(cam->jobs.size());
      for (const auto & [id, job] : cam->jobs) {
        jobs_snapshot.push_back(job);
      }
    }
    if (jobs_snapshot.empty()) return;

    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(*msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "cv_bridge exception on %s: %s", cam->image_topic.c_str(), e.what());
      return;
    }

    cv::Mat image = cv_ptr->image;
    auto stamp = msg->header.stamp;

    for (const auto & job : jobs_snapshot) {
      aruco_ros_cv::DetectionResult det;
      cv::Mat zoom_debug;
      bool want_zoom_debug =
        job->zoom_enabled && job->zoom_debug_pub &&
        job->zoom_debug_pub->get_subscription_count() > 0;
      try {
        if (job->zoom_enabled) {
          det = aruco_ros_cv::detectMultiDictMarkersZoomed(
            image, job->dictionaries, job->marker_sizes, job->cam_mtx, job->dist_coeffs,
            job->zoom, want_zoom_debug ? &zoom_debug : nullptr);
        } else {
          det = aruco_ros_cv::detectMultiDictMarkers(
            image, job->dictionaries, job->marker_sizes, job->cam_mtx, job->dist_coeffs);
        }
      } catch (const std::exception & e) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
          "Detection error on %s [%s]: %s", cam->image_topic.c_str(),
          job->detection_id.c_str(), e.what());
        continue;
      }

      // Publish zoomed-crop debug feed
      if (want_zoom_debug && !zoom_debug.empty()) {
        cv_bridge::CvImage dbg;
        dbg.header.stamp = msg->header.stamp;
        dbg.header.frame_id = msg->header.frame_id;
        dbg.encoding = sensor_msgs::image_encodings::BGR8;
        dbg.image = zoom_debug;
        job->zoom_debug_pub->publish(*dbg.toImageMsg());
      }

      // Publish visualized feed
      if (job->feed_pub->get_subscription_count() > 0) {
        cv::Mat vis = aruco_ros_cv::draw2DVisualization(image, det, job->dictionaries);
        for (const auto & m : det.markers) {
          float axis_len = static_cast<float>(m.marker_size) * 0.75f;
          cv::drawFrameAxes(vis, job->cam_mtx, job->dist_coeffs, m.rvec, m.tvec, axis_len);
        }
        cv_bridge::CvImage out;
        out.header.stamp = stamp;
        out.header.frame_id = msg->header.frame_id;
        out.encoding = sensor_msgs::image_encodings::BGR8;
        out.image = vis;
        job->feed_pub->publish(*out.toImageMsg());
      }

      // Publish marker array
      if (job->markers_pub->get_subscription_count() > 0) {
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

        job->markers_pub->publish(marker_array);
      }

      broadcast_marker_tfs(job, det, msg->header);
    }
  }

  /** Broadcast one TF per detected marker; transform into reference_frame via tf2. */
  void broadcast_marker_tfs(
    const std::shared_ptr<DetectionJob> & job,
    const aruco_ros_cv::DetectionResult & det,
    const std_msgs::msg::Header & header)
  {
    if (det.markers.empty()) return;

    const std::string & camera_frame = header.frame_id;
    std::string parent_frame =
      job->reference_frame.empty() ? camera_frame : job->reference_frame;
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

    size_t total_jobs = 0;
    for (const auto & [topic, cam] : camera_threads_) {
      total_jobs += cam->jobs.size();
    }

    status.message = "Active threads: " + std::to_string(camera_threads_.size()) +
      "/" + std::to_string(max_threads_) + ", active detections: " +
      std::to_string(total_jobs);

    diagnostic_msgs::msg::KeyValue kv_threads;
    kv_threads.key = "active_threads";
    kv_threads.value = std::to_string(camera_threads_.size());
    status.values.push_back(kv_threads);

    diagnostic_msgs::msg::KeyValue kv_limit;
    kv_limit.key = "max_threads";
    kv_limit.value = std::to_string(max_threads_);
    status.values.push_back(kv_limit);

    diagnostic_msgs::msg::KeyValue kv_jobs;
    kv_jobs.key = "active_detections";
    kv_jobs.value = std::to_string(total_jobs);
    status.values.push_back(kv_jobs);

    for (const auto & [topic, cam] : camera_threads_) {
      for (const auto & [id, job] : cam->jobs) {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = "detection";
        kv.value = topic + "#" + id;
        status.values.push_back(kv);
      }
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
    for (auto & [topic, cam] : camera_threads_) {
      std::chrono::steady_clock::time_point last;
      {
        std::lock_guard<std::mutex> tlock(cam->time_mutex);
        last = cam->last_image_time;
      }

      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last);

      if (elapsed.count() > 60) {
        RCLCPP_ERROR(this->get_logger(),
          "Camera feed for aruco processing '%s' not receiving data. "
          "Timeout exceeded (60s). Shutting down processing thread (%zu detections).",
          topic.c_str(), cam->jobs.size());
        shutdown_thread(cam);
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
      camera_threads_.erase(topic);
    }
  }

  void shutdown_thread(std::shared_ptr<CameraThread> cam)
  {
    cam->active = false;
    for (auto & [id, job] : cam->jobs) {
      job->feed_pub.reset();
      job->markers_pub.reset();
      job->zoom_debug_pub.reset();
    }
    cam->jobs.clear();
    cam->image_sub.reset();
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
