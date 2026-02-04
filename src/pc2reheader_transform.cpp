#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/convert.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Dense>
#include <memory>
#include <chrono>
#include <vector>

#ifdef USE_CUDA
#include <pc2_tools_cpp/cuda_transform.h>
#endif

using namespace std::chrono_literals;

class PC2ReheaderTransform : public rclcpp::Node
{
public:
  PC2ReheaderTransform()
    : Node("pc2_reheader_transform")
    , tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock()))
    , tf_listener_(std::make_shared<tf2_ros::TransformListener>(*tf_buffer_))
    , count_(0)
    , error_count_(0)
    , tf_ready_(false)
  {
    // Declare parameters
    this->declare_parameter<std::string>("in", "/depth_camera/points");
    this->declare_parameter<std::string>("out", "/depth_camera/points_world");
    this->declare_parameter<std::string>("source_frame", "depth_camera_link");
    this->declare_parameter<std::string>("target_frame", "world");
    this->declare_parameter<double>("startup_delay", 3.0);
    this->declare_parameter<int>("skip_points", 4);

    // Get parameters
    std::string in_topic = this->get_parameter("in").as_string();
    std::string out_topic = this->get_parameter("out").as_string();
    source_frame_ = this->get_parameter("source_frame").as_string();
    target_frame_ = this->get_parameter("target_frame").as_string();
    double startup_delay = this->get_parameter("startup_delay").as_double();
    skip_points_ = this->get_parameter("skip_points").as_int();

    // QoS profiles
    rclcpp::QoS sensor_qos(10);
    sensor_qos.best_effort();
    sensor_qos.durability_volatile();

    rclcpp::QoS pub_qos(10);
    pub_qos.reliable();
    pub_qos.durability_volatile();

    // Publisher
    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(out_topic, pub_qos);

    #ifdef USE_CUDA
    RCLCPP_INFO(this->get_logger(),
                "PC2ReheaderTransform (CUDA): %s -> %s (%s -> %s), skip=%d",
                in_topic.c_str(), out_topic.c_str(),
                source_frame_.c_str(), target_frame_.c_str(), skip_points_);
    #else
    RCLCPP_INFO(this->get_logger(),
                "PC2ReheaderTransform (CPU): %s -> %s (%s -> %s), skip=%d",
                in_topic.c_str(), out_topic.c_str(),
                source_frame_.c_str(), target_frame_.c_str(), skip_points_);
    #endif
    RCLCPP_INFO(this->get_logger(), "Waiting %.1fs for TF tree...", startup_delay);

    // Startup timer to wait for TF tree
    startup_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(startup_delay),
      [this]() { this->checkTFAndSubscribe(); });

    // Store input topic for later subscription
    in_topic_ = in_topic;
  }

private:
  void checkTFAndSubscribe()
  {
    startup_timer_->cancel();

    // Check if TF is available
    try {
      if (tf_buffer_->canTransform(target_frame_, source_frame_,
                                    tf2::TimePointZero,
                                    tf2::durationFromSec(1.0))) {
        RCLCPP_INFO(this->get_logger(), "TF ready! Starting point cloud processing.");
        tf_ready_ = true;
      } else {
        RCLCPP_WARN(this->get_logger(), "TF not ready after startup delay. Creating subscription anyway...");
      }
    } catch (const std::exception& e) {
      RCLCPP_WARN(this->get_logger(), "TF check failed: %s. Creating subscription anyway...", e.what());
    }

    // Create subscription now
    rclcpp::QoS sub_qos(10);
    sub_qos.best_effort();
    sub_qos.durability_volatile();
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      in_topic_, sub_qos,
      std::bind(&PC2ReheaderTransform::pointCloudCallback, this, std::placeholders::_1));
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    try {
      // Get transform
      geometry_msgs::msg::TransformStamped transform_stamped;
      try {
        transform_stamped = tf_buffer_->lookupTransform(
          target_frame_, source_frame_,
          tf2::TimePointZero,
          tf2::durationFromSec(0.1));
      } catch (const tf2::TransformException& ex) {
        error_count_++;
        if (error_count_ % 50 == 0) {
          RCLCPP_WARN(this->get_logger(), "Transform error (%lu total): %s",
                      error_count_, ex.what());
        }
        return;
      }

      // Convert transform to Eigen
      Eigen::Isometry3d transform_eigen = tf2::transformToEigen(transform_stamped.transform);
      Eigen::Matrix4f transform_matrix = transform_eigen.matrix().cast<float>();

      // Extract rotation and translation
      Eigen::Matrix3f R = transform_matrix.block<3, 3>(0, 0);
      Eigen::Vector3f T = transform_matrix.block<3, 1>(0, 3);

      // Prepare output message
      sensor_msgs::msg::PointCloud2 output_msg;
      output_msg.header.frame_id = target_frame_;
      output_msg.header.stamp = this->now();
      output_msg.height = 1;
      output_msg.is_bigendian = msg->is_bigendian;
      output_msg.is_dense = true;

      // Set fields (x, y, z only)
      output_msg.fields.resize(3);
      output_msg.fields[0].name = "x";
      output_msg.fields[0].offset = 0;
      output_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
      output_msg.fields[0].count = 1;
      output_msg.fields[1].name = "y";
      output_msg.fields[1].offset = 4;
      output_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
      output_msg.fields[1].count = 1;
      output_msg.fields[2].name = "z";
      output_msg.fields[2].offset = 8;
      output_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
      output_msg.fields[2].count = 1;

      output_msg.point_step = 12;  // 3 floats * 4 bytes

      size_t input_points = msg->width * msg->height;
      size_t out_idx = 0;

      #ifdef USE_CUDA
      // CUDA path: GPU-accelerated transformation
      try {
        // Extract all points to host array
        std::vector<float3> h_points_in(input_points);
        sensor_msgs::PointCloud2Iterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(*msg, "z");
        
        for (size_t i = 0; i < input_points; ++i) {
          h_points_in[i].x = *iter_x;
          h_points_in[i].y = *iter_y;
          h_points_in[i].z = *iter_z;
          ++iter_x;
          ++iter_y;
          ++iter_z;
        }
        
        // Prepare rotation matrix (row-major)
        float R_array[9] = {
          R(0,0), R(0,1), R(0,2),
          R(1,0), R(1,1), R(1,2),
          R(2,0), R(2,1), R(2,2)
        };
        float3 T_cuda = {T.x(), T.y(), T.z()};
        
        // Allocate output
        size_t estimated_output = input_points / skip_points_ + 100;
        std::vector<float3> h_points_out(estimated_output);
        int n_valid = 0;
        
        // Call CUDA kernel
        transformPointsCUDA(
          h_points_in.data(),
          h_points_out.data(),
          R_array,
          T_cuda,
          input_points,
          skip_points_,
          &n_valid
        );
        
        if (n_valid > 0) {
          // Copy transformed points to output message
          output_msg.width = n_valid;
          output_msg.row_step = output_msg.point_step * output_msg.width;
          output_msg.data.resize(output_msg.row_step);
          
          float* output_data = reinterpret_cast<float*>(output_msg.data.data());
          for (int i = 0; i < n_valid; ++i) {
            output_data[i * 3 + 0] = h_points_out[i].x;
            output_data[i * 3 + 1] = h_points_out[i].y;
            output_data[i * 3 + 2] = h_points_out[i].z;
          }
          out_idx = n_valid;
        } else {
          // CUDA failed or no valid points, fall through to CPU
          throw std::runtime_error("CUDA returned 0 valid points");
        }
      } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "CUDA transform failed, falling back to CPU: %s", e.what());
        // Fall through to CPU implementation
        #ifndef NO_CUDA
        #define NO_CUDA  // Force CPU path
        #endif
      }
      #endif
      
      #ifndef USE_CUDA
      // CPU fallback path
      #endif
      #if defined(NO_CUDA) || !defined(USE_CUDA)
      // CPU path: Sequential processing
      sensor_msgs::PointCloud2Iterator<float> iter_x(*msg, "x");
      sensor_msgs::PointCloud2Iterator<float> iter_y(*msg, "y");
      sensor_msgs::PointCloud2Iterator<float> iter_z(*msg, "z");
      
      size_t points_processed = 0;
      size_t estimated_output = input_points / skip_points_ + 100;
      output_msg.data.reserve(estimated_output * output_msg.point_step);
      
      for (size_t i = 0; i < input_points; ++i) {
        float x = *iter_x;
        float y = *iter_y;
        float z = *iter_z;
        
        // Process every skip_points_-th point
        if (points_processed % skip_points_ == 0) {
          if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
            // Transform point: R * p + T
            Eigen::Vector3f point_in(x, y, z);
            Eigen::Vector3f point_out = R * point_in + T;
            
            // Grow buffer if needed
            if (out_idx * output_msg.point_step >= output_msg.data.size()) {
              output_msg.data.resize((out_idx + 100) * output_msg.point_step);
            }
            
            float* output_data = reinterpret_cast<float*>(output_msg.data.data() + out_idx * output_msg.point_step);
            output_data[0] = point_out.x();
            output_data[1] = point_out.y();
            output_data[2] = point_out.z();
            out_idx++;
          }
        }
        
        points_processed++;
        ++iter_x;
        ++iter_y;
        ++iter_z;
      }
      #endif

      if (out_idx == 0) {
        return;
      }

      // Set final size (if not already set by CUDA path)
      if (output_msg.width == 0) {
        output_msg.width = out_idx;
        output_msg.row_step = output_msg.point_step * output_msg.width;
        output_msg.data.resize(output_msg.row_step);
      }

      // Publish
      pub_->publish(output_msg);
      count_++;

      if (count_ % 100 == 0) {
        RCLCPP_INFO(this->get_logger(), "Processed %lu clouds (%lu pts)",
                    count_, out_idx);
      }

    } catch (const std::exception& e) {
      error_count_++;
      if (error_count_ % 50 == 0) {
        RCLCPP_ERROR(this->get_logger(), "Error processing point cloud: %s", e.what());
      }
    }
  }

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  
  std::string source_frame_;
  std::string target_frame_;
  std::string in_topic_;
  int skip_points_;
  
  rclcpp::TimerBase::SharedPtr startup_timer_;
  
  size_t count_;
  size_t error_count_;
  bool tf_ready_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PC2ReheaderTransform>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
