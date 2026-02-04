# pc2_tools_cpp

A high-performance ROS 2 package for point cloud transformation and downsampling with optional CUDA GPU acceleration. Replaced by NVBLOX built in capability for onboard jetson.

## Overview

This package provides the `pc2reheader_transform` node, which subscribes to PointCloud2 messages, transforms them from a source coordinate frame to a target frame using TF2, applies downsampling, and republishes the result. It is designed for real-time robotics applications where low-latency point cloud processing is critical.

## Features

- **Frame transformation**: Transforms point clouds between coordinate frames using TF2
- **Downsampling**: Configurable point skip factor to reduce point cloud density
- **CUDA acceleration**: Optional GPU-accelerated transformation (auto-detected at build time)
- **CPU fallback**: Automatically falls back to CPU processing if CUDA is unavailable or fails
- **NaN filtering**: Removes invalid (NaN/Inf) points during processing
- **Configurable QoS**: Uses best-effort subscription and reliable publication

## Dependencies

- ROS 2 (tested with Humble/Iron/Jazzy)
- rclcpp
- sensor_msgs
- tf2_ros
- tf2
- tf2_eigen
- Eigen3
- CUDA Toolkit (optional, for GPU acceleration)

## Building

### Standard Build (CPU only)

```bash
cd ~/your_ros2_ws
colcon build --packages-select pc2_tools_cpp
source install/setup.bash
```

### Build with CUDA Support

CUDA support is automatically enabled if the CUDA Toolkit is detected. The build system looks for CUDA 12.6 at `/usr/local/cuda-12.6` (for Jetpack 6.2 compatibility) or falls back to the default CUDA installation.

```bash
# Ensure CUDA is installed and in PATH
export PATH=/usr/local/cuda/bin:$PATH

cd ~/your_ros2_ws
colcon build --packages-select pc2_tools_cpp
source install/setup.bash
```

## Usage

### Running the Node

```bash
ros2 run pc2_tools_cpp pc2reheader_transform
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in` | string | `/depth_camera/points` | Input PointCloud2 topic |
| `out` | string | `/depth_camera/points_world` | Output PointCloud2 topic |
| `source_frame` | string | `depth_camera_link` | Source TF frame of the input point cloud |
| `target_frame` | string | `world` | Target TF frame for the output point cloud |
| `startup_delay` | double | `3.0` | Seconds to wait for TF tree before subscribing |
| `skip_points` | int | `4` | Downsample factor (process every Nth point) |

### Example with Parameters

```bash
ros2 run pc2_tools_cpp pc2reheader_transform --ros-args \
  -p in:=/camera/depth/points \
  -p out:=/camera/points_world \
  -p source_frame:=camera_depth_optical_frame \
  -p target_frame:=odom \
  -p skip_points:=2 \
  -p startup_delay:=5.0
```

### Launch File Integration

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pc2_tools_cpp',
            executable='pc2reheader_transform',
            name='pc2_transform',
            parameters=[{
                'in': '/depth_camera/points',
                'out': '/depth_camera/points_world',
                'source_frame': 'depth_camera_link',
                'target_frame': 'world',
                'skip_points': 4,
                'startup_delay': 3.0,
            }],
            output='screen',
        ),
    ])
```

## Architecture

### Processing Pipeline

1. **Startup**: Wait for TF tree to become available (configurable delay)
2. **Subscribe**: Listen to input PointCloud2 topic with best-effort QoS
3. **Transform Lookup**: Query TF2 for the source→target transformation
4. **Process Points**:
   - Extract XYZ coordinates from input
   - Apply downsampling (every Nth point)
   - Filter out NaN/Inf values
   - Apply rotation and translation: `p_out = R * p_in + T`
5. **Publish**: Output transformed PointCloud2 with XYZ fields only

### CUDA Kernel (when available)

The CUDA implementation performs the following in a single GPU kernel:
- Point downsampling
- NaN/Inf filtering
- Rigid body transformation (rotation + translation)

This minimizes memory transfers and provides significant speedup for large point clouds.

## Output Format

The output PointCloud2 message contains only XYZ fields (3 × float32 = 12 bytes per point):
- `x`: float32, offset 0
- `y`: float32, offset 4
- `z`: float32, offset 8

Other fields from the input (intensity, RGB, etc.) are not preserved.

## Performance Notes

- **CUDA**: Provides best performance for point clouds with >10,000 points
- **CPU**: Uses Eigen for efficient matrix operations
- **Memory**: Pre-allocates output buffers to minimize runtime allocations
- **Logging**: Progress logged every 100 processed clouds

## Troubleshooting

### TF Transform Errors

If you see frequent "Transform error" warnings:
- Verify your TF tree is publishing transforms between `source_frame` and `target_frame`
- Increase `startup_delay` if transforms take time to become available
- Check frame names for typos

### CUDA Fallback

If the node reports "CUDA transform failed, falling back to CPU":
- Verify GPU drivers are installed correctly
- Check `nvidia-smi` output
- Ensure sufficient GPU memory is available

### No Output Points

If the output topic has no messages:
- Verify the input topic is publishing
- Check that points in the input are valid (not all NaN)
- Reduce `skip_points` value

## License

Apache-2.0

## Maintainer

Rohan (rninamdar@wpi.edu)
