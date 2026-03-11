std_msgs/Header header
uint32 id
geometry_msgs/PoseWithCovariance pose
float64 confidence

# Corner points in image coordinates (4 corners, z=0)
geometry_msgs/Point[] corners

# Dictionary name used for detection (e.g. "DICT_6X6_250")
string dictionary

# Physical marker size in meters
float64 marker_size
