# aruco_ros_cv

ROS 2 package providing ArUco marker detection nodes built directly on OpenCV's
`cv::aruco` API. Unlike `aruco_ros`/`aruco_ros_cv` (the original wrapper),
this package has no shared library — every node includes the header-only
utility library [`include/aruco_ros_cv/aruco_cv_utils.hpp`](include/aruco_ros_cv/aruco_cv_utils.hpp)
directly, which keeps each node self-contained and easy to reason about.

## Table of Contents

- [Quick Start](#quick-start)
- [Package Layout](#package-layout)
- [Dependencies](#dependencies)
- [Building](#building)
- [Core Utility Library](#core-utility-library)
- [Nodes](#nodes)
  - [aruco_action_server](#aruco_action_server)
  - [aruco_start_detection](#aruco_start_detection)
  - [aruco_stop_detection](#aruco_stop_detection)
  - [aruco_rt_capture](#aruco_rt_capture)
  - [aruco_detect_service](#aruco_detect_service)
  - [single_aruco_image_detection](#single_aruco_image_detection)
  - [camera_calibration](#camera_calibration)
- [Launch Files](#launch-files)
- [Camera Intrinsics File Format](#camera-intrinsics-file-format)
- [Adding a New ArUco Dictionary](#adding-a-new-aruco-dictionary)
- [Extending the Action Server: Adding a New Detector](#extending-the-action-server-adding-a-new-detector)
- [Troubleshooting](#troubleshooting)

## Quick Start

The easiest way to run marker detection is: launch the action server once, then
use the `aruco_start_detection` / `aruco_stop_detection` helper nodes to
start/stop individual detections — no need to hand-write `ros2 action
send_goal` commands.

```bash
# Terminal 1 — boot the long-running action server
ros2 launch aruco_ros_cv aruco_action_server.launch
```

```bash
# Terminal 2 — start a detection on /camera/image_raw using DICT_6X6_250
ros2 launch aruco_ros_cv aruco_start_detection.launch \
  image_topic:=/camera/image_raw dictionaries:="['DICT_6X6_250']" marker_sizes:="[0.05]"
```

This prints the resolved `detection_id` (e.g. `DICT_6X6_250`) and publishes to
`/camera/image_raw/arucofeed/DICT_6X6_250` and
`/camera/image_raw/arucofeed/markers/DICT_6X6_250`.

Because detections are keyed by `(image_topic, detection_id)` rather than by
topic alone, you can start a **second, independent detection on the same
camera feed** (e.g. a different dictionary) without using an extra processing
thread — the existing subscription for `/camera/image_raw` is reused:

```bash
# Terminal 3 — add a second detector on the SAME topic, different dictionary
ros2 launch aruco_ros_cv aruco_start_detection.launch \
  image_topic:=/camera/image_raw dictionaries:="['DICT_4X4_50']" marker_sizes:="[0.03]"
```

`~/diagnostics` will now show 1 active thread but 2 active detections. Stop
either one individually, or stop everything on the topic at once:

```bash
# Stop just the DICT_4X4_50 detection
ros2 launch aruco_ros_cv aruco_stop_detection.launch \
  image_topic:=/camera/image_raw detection_id:=DICT_4X4_50

# Stop every detection left on the topic (detection_id omitted = stop all)
ros2 launch aruco_ros_cv aruco_stop_detection.launch image_topic:=/camera/image_raw
```

## Package Layout

```
include/aruco_ros_cv/aruco_cv_utils.hpp   Header-only detection/visualization/conversion helpers
src/aruco_action_server.cpp               Multi-camera, multi-detection action-server based detector
src/aruco_start_detection.cpp             CLI helper: sends a start_detection goal
src/aruco_stop_detection.cpp              CLI helper: calls the stop_detection service
src/aruco_rt_capture.cpp                  Single-camera real-time detector node
src/aruco_detect_service.cpp              Stateless single-image detection service
src/single_aruco_image_detection.cpp      One-shot detection on an image file (CLI-style)
src/camera_calibration.cpp                Checkerboard-based camera intrinsics calibration
launch/                                   One launch file per node
resources/camera_params.yml               Example OpenCV FileStorage camera intrinsics file
```

## Dependencies

- `rclcpp`, `rclcpp_action`
- `sensor_msgs`, `geometry_msgs`, `std_msgs`, `diagnostic_msgs`
- `aruco_msgs` (Marker / MarkerArray)
- `aruco_ros_cv_interfaces` (ArucoRtStart action, ArucoDetect / ArucoRtStop services)
- `cv_bridge`, `image_transport`
- `OpenCV` (built with the `aruco` and `calib3d` modules)

## Building

From the workspace root:

```bash
colcon build --packages-up-to aruco_ros_cv
source install/setup.bash
```

Build just this package (interfaces must already be built, since it depends on
`aruco_ros_cv_interfaces` and `aruco_msgs`):

```bash
colcon build --packages-select aruco_ros_cv
```

If you changed the `.action`/`.srv` files in `aruco_ros_cv_interfaces`, rebuild
that package first:

```bash
colcon build --packages-select aruco_ros_cv_interfaces aruco_msgs
colcon build --packages-select aruco_ros_cv
```

## Core Utility Library

All detection logic lives in `aruco_cv_utils.hpp` (namespace `aruco_ros_cv`) so
every node shares identical behavior:

| Function | Purpose |
|---|---|
| `dictionaryFromString(name)` | Maps a string like `"DICT_6X6_250"` to the OpenCV `cv::aruco` dictionary enum. Throws `std::invalid_argument` on unknown names. |
| `buildCameraMatrix(fx, fy, cx, cy)` / `buildDistCoeffs(...)` | Build intrinsic matrices from raw numbers. |
| `loadCameraIntrinsicsFromYAML` / `saveCameraIntrinsicsToYAML` | Read/write OpenCV `FileStorage` YAML camera params. |
| `detectMultiDictMarkers(image, dictionaries, marker_sizes, camera_matrix, dist_coeffs)` | Runs `cv::aruco::detectMarkers` + `cv::aruco::estimatePoseSingleMarkers` once per dictionary and returns a merged `DetectionResult`. |
| `draw2DVisualization` / `draw3DVisualization` | Overlay bounding boxes/labels or 3D pose axes for a detection result. |
| `rvecTvecToPositionQuat` | Converts an OpenCV Rodrigues rotation + translation vector into a ROS-style position + quaternion. |

Key types:

```cpp
struct DetectedMarker {
  int id;
  std::string dictionary_name;
  double marker_size;
  std::vector<cv::Point2f> corners;
  cv::Vec3d rvec, tvec;
};

struct DetectionResult {
  std::vector<DetectedMarker> markers;
};
```

Every node calls `detectMultiDictMarkers` and then converts the resulting
`DetectionResult` into `aruco_msgs/msg/MarkerArray` and/or visualization
images — see [Extending the Action Server](#extending-the-action-server-adding-a-new-detector)
for how to plug in a different detection backend.

## Nodes

### aruco_action_server

The primary long-running node. Manages an arbitrary number of **independent,
dynamically started/stopped detections**, each identified by
`(image_topic, detection_id)`, through a ROS 2 action + service interface.

**Threads vs. detections** — a *processing thread* (one image subscription)
is created per unique `image_topic`. Starting a second detection on a topic
that's already being processed (e.g. a different dictionary) reuses that
same thread/subscription instead of creating a new one; only brand-new
topics count against `max_processing_threads`.

- **Action** `~/start_detection` (`aruco_ros_cv_interfaces/action/ArucoRtStart`)
  Starts a detection on a camera topic.
  - Goal: `string image_topic`, `string[] dictionaries`, `float64[] marker_sizes`, `sensor_msgs/CameraInfo camera_info`, `string detection_id`
  - Result: `bool success`, `string message`, `string detection_id`
  - Feedback: `string status`, `uint32 active_threads`
  - `detection_id` is optional — if empty, it's auto-generated by joining `dictionaries` with `_` (e.g. `DICT_6X6_250`), and the resolved id is always echoed back in `Result.detection_id`.
  - Rejected if `(image_topic, detection_id)` is already active, or if the topic is new and `max_processing_threads` has been reached.
- **Service** `~/stop_detection` (`aruco_ros_cv_interfaces/srv/ArucoRtStop`)
  - Request: `string image_topic`, `string detection_id` → Response: `bool success`, `string message`
  - If `detection_id` is empty, **every** detection on `image_topic` is stopped and its processing thread is torn down (this is the only behavior when `detection_id` is omitted, and matches stopping a topic that only ever had one detection).
  - If `detection_id` is set, only that detection is stopped; the thread itself is only torn down once its last detection is removed.
- **Topic** `~/diagnostics` (`diagnostic_msgs/DiagnosticArray`, published every 2s) — reports active thread count, active detection count, and one `detection` key/value per `<topic>#<detection_id>` pair.
- **Per-detection outputs**, created once a goal is accepted for `(image_topic, detection_id)`:
  - `<image_topic>/arucofeed/<detection_id>` (`sensor_msgs/Image`) — annotated 2D+3D visualization (only rendered if there's a subscriber)
  - `<image_topic>/arucofeed/markers/<detection_id>` (`aruco_msgs/MarkerArray`) — detected marker poses (only published if there's a subscriber)
- **Watchdog**: every 5s, any camera topic idle > 30s gets a `WARN` diagnostic; idle > 60s causes that topic's thread (and all of its detections) to be automatically shut down.

Parameters:

| Name | Default | Description |
|---|---|---|
| `max_processing_threads` | `10` | Max number of concurrent per-topic processing threads (not detections). |

Example — start two detections on the same topic with the CLI (prefer the
[aruco_start_detection](#aruco_start_detection) / [aruco_stop_detection](#aruco_stop_detection)
nodes for day-to-day use; raw commands shown here for reference):

```bash
ros2 launch aruco_ros_cv aruco_action_server.launch
```

```bash
ros2 action send_goal /aruco_action_server/start_detection \
  aruco_ros_cv_interfaces/action/ArucoRtStart \
  "{image_topic: /camera/image_raw, dictionaries: ['DICT_6X6_250'], marker_sizes: [0.05], camera_info: {k: [951.26,0,644.98,0,944.24,360.02,0,0,1], d: [0.094,-0.27,0.0018,0.0027,0.403]}}"

# Second detection on the SAME topic — succeeds, reuses the existing thread
ros2 action send_goal /aruco_action_server/start_detection \
  aruco_ros_cv_interfaces/action/ArucoRtStart \
  "{image_topic: /camera/image_raw, dictionaries: ['DICT_4X4_50'], marker_sizes: [0.03], camera_info: {k: [951.26,0,644.98,0,944.24,360.02,0,0,1], d: [0.094,-0.27,0.0018,0.0027,0.403]}}"

# Repeating the FIRST call again is rejected (same image_topic + detection_id)
```

```bash
# Stop just one detection
ros2 service call /aruco_action_server/stop_detection \
  aruco_ros_cv_interfaces/srv/ArucoRtStop \
  "{image_topic: /camera/image_raw, detection_id: DICT_4X4_50}"

# Stop everything left on the topic (detection_id omitted)
ros2 service call /aruco_action_server/stop_detection \
  aruco_ros_cv_interfaces/srv/ArucoRtStop "{image_topic: /camera/image_raw}"
```

### aruco_start_detection

One-shot CLI helper that builds and sends an `ArucoRtStart` goal for you —
no need to hand-write the `camera_info` block or remember the action name.
It loads camera intrinsics from a YAML file, sends the goal, waits for the
result, and prints the resolved `detection_id` plus a ready-to-use
`aruco_stop_detection` command for stopping it later.

Parameters:

| Name | Default | Description |
|---|---|---|
| `action_server_name` | `aruco_action_server` | Node name of the running action server. |
| `image_topic` | `/camera/image_raw` | Camera image topic to start detection on. |
| `dictionaries` | `['DICT_6X6_250']` | List of ArUco dictionary names. |
| `marker_sizes` | `[0.05]` | List of marker sizes in meters, parallel to `dictionaries`. |
| `camera_params_file` | `resources/camera_params.yml` | Path to an OpenCV YAML camera intrinsics file. |
| `detection_id` | `""` | Optional id; auto-generated from `dictionaries` if empty. |

```bash
ros2 launch aruco_ros_cv aruco_start_detection.launch \
  image_topic:=/camera/image_raw \
  dictionaries:="['DICT_6X6_250']" marker_sizes:="[0.05]" \
  camera_params_file:=/path/to/camera_params.yml
```

### aruco_stop_detection

One-shot CLI helper that calls the `stop_detection` service for you.

Parameters:

| Name | Default | Description |
|---|---|---|
| `action_server_name` | `aruco_action_server` | Node name of the running action server. |
| `image_topic` | `/camera/image_raw` | Camera image topic to stop detection on. |
| `detection_id` | `""` | Id of the detection to stop; if empty, **all** detections on `image_topic` are stopped. |

```bash
# Stop one specific detection
ros2 launch aruco_ros_cv aruco_stop_detection.launch \
  image_topic:=/camera/image_raw detection_id:=DICT_6X6_250

# Stop everything on that topic
ros2 launch aruco_ros_cv aruco_stop_detection.launch image_topic:=/camera/image_raw
```

### aruco_rt_capture

Simpler, single-camera always-on real-time detector. Subscribes to one image
topic + one camera info topic (params, not dynamic), and republishes the same
`arucofeed` / `arucofeed/markers` outputs as the action server.

Parameters: `image_topic`, `dictionaries`, `marker_sizes`, `camera_info_topic`.
Camera intrinsics are taken from the first message on `camera_info_topic`
(latched, `transient_local` QoS) and then cached.

```bash
ros2 launch aruco_ros_cv aruco_rt_capture.launch \
  image_topic:=/camera/image_raw \
  dictionaries:="['DICT_6X6_250']" marker_sizes:="[0.05]" \
  camera_info_topic:=/camera/camera_info
```

### aruco_detect_service

Stateless request/response detection: send one image + intrinsics, get back
2D/3D visualization images and a `MarkerArray`. No launch file ships for this
node by default — run it directly:

```bash
ros2 run aruco_ros_cv aruco_detect_service
```

Service `/aruco_detect` (`aruco_ros_cv_interfaces/srv/ArucoDetect`):
- Request: `sensor_msgs/Image image`, `string[] dictionaries`, `float64[] marker_sizes`, `sensor_msgs/CameraInfo camera_info`
- Response: `sensor_msgs/Image visualization_2d`, `sensor_msgs/Image visualization_3d`, `aruco_msgs/MarkerArray markers`

### single_aruco_image_detection

One-shot CLI-style node: reads an image file from disk, detects markers,
logs every marker's corners/pose, and saves `<name>_2dvis.<ext>` and
`<name>_3dvis.<ext>` next to (or in `output_dir`) the source image.

```bash
ros2 launch aruco_ros_cv single_aruco_image_detection.launch \
  image_path:=/path/to/image.jpg \
  camera_params_file:=/path/to/camera_params.yml \
  dictionaries:="['DICT_6X6_250']" marker_sizes:="[0.05]"
```

### camera_calibration

Collects checkerboard frames from an image topic, runs
`cv::calibrateCamera`, and writes the intrinsics to an OpenCV YAML file
consumable by `loadCameraIntrinsicsFromYAML` (and thus by
`single_aruco_image_detection` / `aruco_detect_service`).

- **Topic** `<image_topic>/calibration_feed` (`sensor_msgs/Image`) — every incoming
  frame with the detected checkerboard corners overlaid (green text/corners when
  found, red text when not) plus a running `frames collected/num_frames` counter.
  Only rendered if there's a subscriber; view it live with
  `ros2 run rqt_image_view rqt_image_view` to frame the board correctly.
- The subscription keeps only the latest frame (`keep_last(1)`), so a slow
  callback processes the newest image instead of falling behind on a backlog.

```bash
ros2 launch aruco_ros_cv camera_calibration.launch \
  image_topic:=/camera/image_raw \
  checkerboard_rows:=6 checkerboard_cols:=9 square_size:=0.025 \
  num_frames:=20 output_file:=my_camera_params.yml
```

The node shuts itself down once `num_frames` valid checkerboard detections
have been collected and calibration succeeds.

## Launch Files

| Launch file | Node | Key args |
|---|---|---|
| `aruco_action_server.launch` | `aruco_action_server` | `ns`, `max_processing_threads` |
| `aruco_start_detection.launch` | `aruco_start_detection` | `action_server_name`, `image_topic`, `dictionaries`, `marker_sizes`, `camera_params_file`, `detection_id` |
| `aruco_stop_detection.launch` | `aruco_stop_detection` | `action_server_name`, `image_topic`, `detection_id` |
| `aruco_rt_capture.launch` | `aruco_rt_capture` | `image_topic`, `dictionaries`, `marker_sizes`, `camera_info_topic` |
| `single_aruco_image_detection.launch` | `single_aruco_image_detection` | `image_path`, `camera_params_file`, `dictionaries`, `marker_sizes`, `output_dir` |
| `camera_calibration.launch` | `camera_calibration` | `image_topic`, `checkerboard_rows`, `checkerboard_cols`, `square_size`, `output_file`, `num_frames` |

All launch files use ROS 2 XML launch syntax and expose every node parameter
as a launch argument with a default value — inspect the file under `launch/`
for the exact defaults, or override any of them with `name:=value` on the
`ros2 launch` command line.

## Camera Intrinsics File Format

`resources/camera_params.yml` is an OpenCV `FileStorage` YAML file:

```yaml
%YAML:1.0
cameraMatrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
distCoeffs: !!opencv-matrix
   rows: 5
   cols: 1
   dt: d
   data: [k1, k2, p1, p2, k3]
```

Generate one with `camera_calibration`, or hand-edit it from a known
calibration. Note: `saveCameraIntrinsicsToYAML` writes keys `camera_matrix` /
`dist_coeffs` while `loadCameraIntrinsicsFromYAML` reads `cameraMatrix` /
`distCoeffs` — if you calibrate with this package and then load the result
back with `single_aruco_image_detection`, rename the keys in the saved file
(or align the key names in `aruco_cv_utils.hpp` if you change one side).

## Adding a New ArUco Dictionary

If OpenCV already defines the dictionary you need (any `cv::aruco::DICT_*`
constant), just add it to the map in `dictionaryFromString`:

```cpp
// include/aruco_ros_cv/aruco_cv_utils.hpp
static const std::map<std::string, int> dict_map = {
  ...
  {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
  {"DICT_MY_NEW_ONE", cv::aruco::DICT_MY_NEW_ONE},  // add here
};
```

No other changes are needed — every node (including the action server)
already accepts dictionary names as a runtime parameter/goal field, so a
rebuild of `aruco_ros_cv` is sufficient.

## Extending the Action Server: Adding a New Detector

"Detector" here means the algorithm that turns an image into
`DetectionResult`-shaped data (currently always
`aruco_ros_cv::detectMultiDictMarkers`, which wraps `cv::aruco`). Use this
section if you want the action server to support a different marker family
or a completely different detection algorithm (e.g., a custom CNN detector,
a different fiducial system, or a non-`cv::aruco` implementation).

Recall the action server's two-level model: a `CameraThread` (one per
`image_topic`, owns the subscription) holds a map of `DetectionJob`s (one per
`detection_id`). The detector choice below is a per-job setting, so different
detections on the same topic could even use different detectors.

### 1. Implement a detector function with the same shape

Add a new function to `aruco_cv_utils.hpp` (or a new header alongside it)
that takes an image + configuration and returns a `DetectionResult`:

```cpp
// include/aruco_ros_cv/aruco_cv_utils.hpp
inline DetectionResult detectWithMyDetector(
  const cv::Mat & image,
  const MyDetectorConfig & config,
  const cv::Mat & camera_matrix,
  const cv::Mat & dist_coeffs)
{
  DetectionResult result;
  // ... run your detector, fill result.markers with DetectedMarker entries ...
  return result;
}
```

Reusing `DetectedMarker`/`DetectionResult` means `draw2DVisualization`,
`draw3DVisualization`, and the `aruco_msgs::msg::MarkerArray` conversion code
in `aruco_action_server.cpp` keep working unmodified.

### 2. Expose the choice of detector on the action goal

`ArucoRtStart.action` currently has no field to select a detector — every
goal implicitly uses `detectMultiDictMarkers`. Add one:

```
# aruco_ros_cv_interfaces/action/ArucoRtStart.action
# Goal
string image_topic
string[] dictionaries
float64[] marker_sizes
sensor_msgs/CameraInfo camera_info
string detector_type   # NEW: e.g. "aruco" (default) or "my_detector"
---
# Result
bool success
string message
---
# Feedback
string status
uint32 active_threads
```

Rebuild the interfaces package after editing the `.action` file:

```bash
colcon build --packages-select aruco_ros_cv_interfaces
```

### 3. Store the detector choice per detection job

In `aruco_action_server.cpp`, add the field to `DetectionJob` and set it in
`execute_start` when the job is built:

```cpp
struct DetectionJob
{
  std::string detection_id;
  std::string detector_type;   // NEW
  std::vector<std::string> dictionaries;
  std::vector<double> marker_sizes;
  cv::Mat cam_mtx, dist_coeffs;
  ...
};
```

```cpp
job->detector_type = goal->detector_type.empty() ? "aruco" : goal->detector_type;
```

### 4. Dispatch to the right detector in `process_image`

`process_image` already loops over a snapshot of each `CameraThread`'s jobs.
Replace the single call to `detectMultiDictMarkers` inside that loop with a
small dispatch on `job->detector_type`:

```cpp
aruco_ros_cv::DetectionResult det;
try {
  if (job->detector_type == "aruco") {
    det = aruco_ros_cv::detectMultiDictMarkers(
      image, job->dictionaries, job->marker_sizes, job->cam_mtx, job->dist_coeffs);
  } else if (job->detector_type == "my_detector") {
    det = aruco_ros_cv::detectWithMyDetector(
      image, my_detector_config_, job->cam_mtx, job->dist_coeffs);
  } else {
    throw std::invalid_argument("Unknown detector_type: " + job->detector_type);
  }
} catch (const std::exception & e) {
  RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
    "Detection error on %s [%s]: %s", cam->image_topic.c_str(),
    job->detection_id.c_str(), e.what());
  continue;
}
```

Validate `detector_type` in `handle_goal()` too (alongside the existing
dictionary/size length check), so bad requests are rejected immediately
instead of failing later inside the detached thread.

### 5. Update CMakeLists.txt if you added new source files or dependencies

If the new detector needs an additional library (e.g. a CNN inference
runtime), add `find_package(...)` near the top of `CMakeLists.txt` and append
the dependency name to the `aruco_action_server` target's
`add_aruco_node(...)` call:

```cmake
find_package(my_detector_lib REQUIRED)
...
add_aruco_node(aruco_action_server
  src/aruco_action_server.cpp
  ${COMMON_DEPS} rclcpp_action aruco_ros_cv_interfaces diagnostic_msgs my_detector_lib
)
```

### 6. Rebuild and test

```bash
colcon build --packages-select aruco_ros_cv_interfaces aruco_ros_cv
source install/setup.bash
ros2 launch aruco_ros_cv aruco_action_server.launch
ros2 action send_goal /aruco_action_server/start_detection \
  aruco_ros_cv_interfaces/action/ArucoRtStart \
  "{image_topic: /camera/image_raw, detector_type: my_detector, ...}"
```

If your new detector only needs a *different ArUco dictionary* (not a whole
new algorithm), skip all of the above and see
[Adding a New ArUco Dictionary](#adding-a-new-aruco-dictionary) instead.

## Troubleshooting

- **`GoalResponse::REJECT` with "already running"** — a detection with the
  same `(image_topic, detection_id)` is already active. Either stop it first
  (`aruco_stop_detection` / `~/stop_detection`), or start the new one with a
  different `detection_id` (or a different `dictionaries` list, since the id
  auto-generates from it) — that's how you add a second detector to the same
  camera feed. Check `~/diagnostics` for the full list of active detections.
- **`GoalResponse::REJECT` with thread-limit error** — this only triggers
  when starting detection on a **brand-new** topic; raise
  `max_processing_threads` on launch or stop unused feeds. Adding another
  detection to a topic that's already active never hits this limit.
- **No `arucofeed` output** — the action server/`aruco_rt_capture` only
  render/publish the visualization image when something is subscribed to
  `<image_topic>/arucofeed/<detection_id>`; `ros2 topic echo`/RViz/`rqt_image_view`
  on that topic to trigger publishing.
- **`Unknown ArUco dictionary` exception** — the dictionary string doesn't
  match any key in `dictionaryFromString`; check spelling against the table
  in [Core Utility Library](#core-utility-library) or extend the map as
  described above.
- **Detection watchdog shuts down a feed** — no image arrived on that topic
  for 60s; verify the camera driver is publishing and the topic name matches
  exactly (including namespace). This tears down every detection on that
  topic, not just one.
- **Stopping one detection also stopped the others** — you called
  `~/stop_detection` (or `aruco_stop_detection`) with an empty `detection_id`,
  which intentionally stops *every* detection on that topic. Pass the
  specific `detection_id` to stop just one.
