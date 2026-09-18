/**
 * @file pose_filter.hpp
 * @brief Header-only temporal smoothing of ArUco marker poses (position, orientation, corners).
 *
 * A PoseFilter keeps a short per-marker-id history and, on each frame, replaces every
 * DetectedMarker's tvec/rvec/corners with a smoothed estimate. Because the position, the
 * orientation and the 2D corners are all rewritten in place, every downstream consumer
 * (2D visualization, drawn axes, MarkerArray, TF broadcast) sees the same smoothed pose.
 *
 * Methods (smoothing_type):
 *   - moving_average : mean of the last N samples (default, window = 10)
 *   - median         : per-component median of the last N samples (mean orientation)
 *   - ema            : exponential moving average weighted by ema_alpha
 *
 * Orientation is handled in quaternion space with sign alignment so opposite-hemisphere
 * quaternions do not cancel; ema orientation uses SLERP.
 */

#ifndef ARUCO_ROS_CV__POSE_FILTER_HPP_
#define ARUCO_ROS_CV__POSE_FILTER_HPP_

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include "aruco_ros_cv/aruco_cv_utils.hpp"

namespace aruco_ros_cv
{

enum class SmoothingType
{
  None,
  MovingAverage,
  Median,
  Ema
};

/** Parse a smoothing_type string; unknown/empty falls back to MovingAverage. */
inline SmoothingType smoothingTypeFromString(const std::string & name)
{
  if (name == "none" || name == "off") {return SmoothingType::None;}
  if (name == "median") {return SmoothingType::Median;}
  if (name == "ema" || name == "exponential") {return SmoothingType::Ema;}
  return SmoothingType::MovingAverage;
}

/** Runtime configuration for a PoseFilter. */
struct PoseFilterConfig
{
  bool enabled{false};
  SmoothingType type{SmoothingType::MovingAverage};
  int window{10};                // samples for moving_average/median
  double ema_alpha{0.3};         // newest-sample weight for ema
  double reset_timeout{0.5};     // sec; clear history after a gap this long
};

/** A quaternion stored as (w, x, y, z), always normalized. */
using Quat = cv::Vec4d;

/** rvec (Rodrigues) -> unit quaternion (w, x, y, z). */
inline Quat rvecToQuat(const cv::Vec3d & rvec)
{
  const double angle = std::sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]);
  if (angle < 1e-9) {
    return Quat(1.0, 0.0, 0.0, 0.0);
  }
  const double half = angle * 0.5;
  const double s = std::sin(half) / angle;
  return Quat(std::cos(half), rvec[0] * s, rvec[1] * s, rvec[2] * s);
}

/** unit quaternion (w, x, y, z) -> rvec (Rodrigues). */
inline cv::Vec3d quatToRvec(const Quat & q)
{
  Quat n = q;
  const double norm = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2] + n[3] * n[3]);
  if (norm < 1e-12) {
    return cv::Vec3d(0.0, 0.0, 0.0);
  }
  n /= norm;
  double w = n[0];
  if (w > 1.0) {w = 1.0;}
  if (w < -1.0) {w = -1.0;}
  const double angle = 2.0 * std::acos(w);
  const double s = std::sqrt(std::max(0.0, 1.0 - w * w));
  if (s < 1e-9) {
    return cv::Vec3d(0.0, 0.0, 0.0);
  }
  const double scale = angle / s;
  return cv::Vec3d(n[1] * scale, n[2] * scale, n[3] * scale);
}

inline double quatDot(const Quat & a, const Quat & b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

inline Quat quatNormalize(const Quat & q)
{
  const double norm = std::sqrt(quatDot(q, q));
  return (norm < 1e-12) ? Quat(1.0, 0.0, 0.0, 0.0) : Quat(q / norm);
}

/** Flip q to the same hemisphere as ref so an average does not cancel. */
inline Quat quatAlign(const Quat & q, const Quat & ref)
{
  return (quatDot(q, ref) < 0.0) ? Quat(-q[0], -q[1], -q[2], -q[3]) : q;
}

/** Shortest-path SLERP from a to b by t in [0,1]. */
inline Quat quatSlerp(const Quat & a, const Quat & b_in, double t)
{
  Quat b = quatAlign(b_in, a);
  double d = quatDot(a, b);
  if (d > 1.0) {d = 1.0;}
  if (d > 0.9995) {
    return quatNormalize(Quat(a + t * (b - a)));   // nearly parallel: linear is safe
  }
  const double theta0 = std::acos(d);
  const double theta = theta0 * t;
  const double sin_theta0 = std::sin(theta0);
  const double s0 = std::sin(theta0 - theta) / sin_theta0;
  const double s1 = std::sin(theta) / sin_theta0;
  return quatNormalize(Quat(s0 * a + s1 * b));
}

/** Temporal smoother maintaining one history per marker id. */
class PoseFilter
{
public:
  void configure(const PoseFilterConfig & cfg) {cfg_ = cfg;}
  const PoseFilterConfig & config() const {return cfg_;}

  /** Smooth every marker in det in place using the frame timestamp (seconds). */
  void apply(DetectionResult & det, double stamp_sec)
  {
    if (!cfg_.enabled || cfg_.type == SmoothingType::None) {
      return;
    }
    for (auto & m : det.markers) {
      smoothMarker(m, stamp_sec);
    }
  }

private:
  struct Sample
  {
    cv::Vec3d t;
    Quat q;
    std::vector<cv::Point2f> corners;
  };

  struct MarkerState
  {
    std::deque<Sample> history;
    cv::Vec3d ema_t;
    Quat ema_q;
    std::vector<cv::Point2f> ema_corners;
    bool has_ema{false};
    double last_stamp{-1.0};
  };

  void smoothMarker(DetectedMarker & m, double stamp_sec)
  {
    MarkerState & st = states_[m.id];

    // A marker that vanished and returned should not average across the gap.
    if (st.last_stamp >= 0.0 && cfg_.reset_timeout > 0.0 &&
      (stamp_sec - st.last_stamp) > cfg_.reset_timeout)
    {
      st = MarkerState{};
    }
    st.last_stamp = stamp_sec;

    Sample cur;
    cur.t = m.tvec;
    cur.q = quatNormalize(rvecToQuat(m.rvec));
    cur.corners = m.corners;

    cv::Vec3d out_t;
    Quat out_q;
    std::vector<cv::Point2f> out_corners;

    if (cfg_.type == SmoothingType::Ema) {
      smoothEma(st, cur, out_t, out_q, out_corners);
    } else {
      smoothWindow(st, cur, out_t, out_q, out_corners);
    }

    m.tvec = out_t;
    m.rvec = quatToRvec(out_q);
    if (out_corners.size() == m.corners.size()) {
      m.corners = out_corners;
    }
  }

  void smoothEma(
    MarkerState & st, const Sample & cur,
    cv::Vec3d & out_t, Quat & out_q, std::vector<cv::Point2f> & out_corners)
  {
    double a = cfg_.ema_alpha;
    if (a <= 0.0 || a > 1.0) {a = 0.3;}

    if (!st.has_ema || st.ema_corners.size() != cur.corners.size()) {
      st.ema_t = cur.t;
      st.ema_q = cur.q;
      st.ema_corners = cur.corners;
      st.has_ema = true;
    } else {
      st.ema_t = a * cur.t + (1.0 - a) * st.ema_t;
      st.ema_q = quatSlerp(st.ema_q, cur.q, a);
      for (size_t i = 0; i < st.ema_corners.size(); ++i) {
        st.ema_corners[i].x =
          static_cast<float>(a * cur.corners[i].x + (1.0 - a) * st.ema_corners[i].x);
        st.ema_corners[i].y =
          static_cast<float>(a * cur.corners[i].y + (1.0 - a) * st.ema_corners[i].y);
      }
    }
    out_t = st.ema_t;
    out_q = quatNormalize(st.ema_q);
    out_corners = st.ema_corners;
  }

  void smoothWindow(
    MarkerState & st, const Sample & cur,
    cv::Vec3d & out_t, Quat & out_q, std::vector<cv::Point2f> & out_corners)
  {
    const std::size_t win = static_cast<std::size_t>(std::max(1, cfg_.window));
    st.history.push_back(cur);
    while (st.history.size() > win) {
      st.history.pop_front();
    }

    const bool median = (cfg_.type == SmoothingType::Median);
    const bool corners_consistent = cornersConsistent(st.history);

    if (median) {
      out_t = medianVec3(st.history);
      out_corners = corners_consistent ? medianCorners(st.history) : cur.corners;
    } else {
      out_t = meanVec3(st.history);
      out_corners = corners_consistent ? meanCorners(st.history) : cur.corners;
    }
    // Orientation always uses a sign-aligned quaternion mean (a median quaternion
    // is not well defined), which stays stable for both window methods.
    out_q = meanQuat(st.history);
  }

  static bool cornersConsistent(const std::deque<Sample> & h)
  {
    if (h.empty()) {return false;}
    const std::size_t n = h.front().corners.size();
    if (n == 0) {return false;}
    for (const auto & s : h) {
      if (s.corners.size() != n) {return false;}
    }
    return true;
  }

  static cv::Vec3d meanVec3(const std::deque<Sample> & h)
  {
    cv::Vec3d acc(0.0, 0.0, 0.0);
    for (const auto & s : h) {acc += s.t;}
    return (h.empty()) ? acc : cv::Vec3d(acc / static_cast<double>(h.size()));
  }

  static cv::Vec3d medianVec3(const std::deque<Sample> & h)
  {
    cv::Vec3d out(0.0, 0.0, 0.0);
    for (int axis = 0; axis < 3; ++axis) {
      std::vector<double> vals;
      vals.reserve(h.size());
      for (const auto & s : h) {vals.push_back(s.t[axis]);}
      out[axis] = median1d(vals);
    }
    return out;
  }

  static Quat meanQuat(const std::deque<Sample> & h)
  {
    if (h.empty()) {return Quat(1.0, 0.0, 0.0, 0.0);}
    const Quat ref = h.back().q;
    Quat acc(0.0, 0.0, 0.0, 0.0);
    for (const auto & s : h) {
      acc += quatAlign(s.q, ref);
    }
    return quatNormalize(acc);
  }

  static std::vector<cv::Point2f> meanCorners(const std::deque<Sample> & h)
  {
    const std::size_t n = h.front().corners.size();
    std::vector<cv::Point2f> out(n, cv::Point2f(0.f, 0.f));
    for (const auto & s : h) {
      for (std::size_t i = 0; i < n; ++i) {
        out[i].x += s.corners[i].x;
        out[i].y += s.corners[i].y;
      }
    }
    const float inv = 1.0f / static_cast<float>(h.size());
    for (auto & p : out) {p.x *= inv; p.y *= inv;}
    return out;
  }

  static std::vector<cv::Point2f> medianCorners(const std::deque<Sample> & h)
  {
    const std::size_t n = h.front().corners.size();
    std::vector<cv::Point2f> out(n);
    for (std::size_t i = 0; i < n; ++i) {
      std::vector<double> xs;
      std::vector<double> ys;
      xs.reserve(h.size());
      ys.reserve(h.size());
      for (const auto & s : h) {
        xs.push_back(s.corners[i].x);
        ys.push_back(s.corners[i].y);
      }
      out[i].x = static_cast<float>(median1d(xs));
      out[i].y = static_cast<float>(median1d(ys));
    }
    return out;
  }

  static double median1d(std::vector<double> & v)
  {
    if (v.empty()) {return 0.0;}
    std::sort(v.begin(), v.end());
    const std::size_t mid = v.size() / 2;
    return (v.size() % 2 == 1) ? v[mid] : 0.5 * (v[mid - 1] + v[mid]);
  }

  PoseFilterConfig cfg_;
  std::map<int, MarkerState> states_;
};

}  // namespace aruco_ros_cv

#endif  // ARUCO_ROS_CV__POSE_FILTER_HPP_
