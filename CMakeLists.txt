cmake_minimum_required(VERSION 3.8)
project(pc2_tools_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try to enable CUDA if available (optional)
# CUDA 12.6 installed for Jetpack 6.2 compatibility
if(EXISTS "/usr/local/cuda-12.6")
  set(CUDAToolkit_ROOT "/usr/local/cuda-12.6")
endif()
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)

# Set dependencies list
set(dependencies
  rclcpp
  sensor_msgs
  tf2_ros
  tf2
  tf2_eigen
  Eigen3
)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(include)

# CUDA configuration (optional)
if(CUDAToolkit_FOUND AND CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
  add_definitions(-DUSE_CUDA)
  
  # Find CUDA runtime library explicitly
  find_library(CUDART_LIBRARY
    NAMES cudart
    PATHS ${CUDAToolkit_LIBRARY_DIR}
    NO_DEFAULT_PATH
  )
  
  if(NOT CUDART_LIBRARY)
    # Fallback: try standard locations
    find_library(CUDART_LIBRARY
      NAMES cudart
      PATHS ${CUDAToolkit_ROOT}/lib64
            ${CUDAToolkit_ROOT}/lib
            /usr/local/cuda/lib64
            /usr/local/cuda/lib
    )
  endif()
  
  if(CUDART_LIBRARY)
    message(STATUS "Found CUDA runtime: ${CUDART_LIBRARY}")
  else()
    message(WARNING "CUDA runtime library not found, but CUDA toolkit is available")
  endif()
  
  # CUDA library
  add_library(cuda_transform_lib STATIC
    src/pc2reheader_transform.cu
  )
  # Set CUDA architecture (common architectures: 75=Turing, 80=Ampere, 86=Ada, 89=Blackwell)
  # Use 75 (Turing) as a safe default that works on most modern GPUs
  set_property(TARGET cuda_transform_lib PROPERTY CUDA_ARCHITECTURES 75)
  
  set_target_properties(cuda_transform_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(cuda_transform_lib PUBLIC
    include
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(cuda_transform_lib PUBLIC
    ${CUDAToolkit_LIBRARIES}
  )
  
  # Explicitly link CUDA runtime if found
  if(CUDART_LIBRARY)
    target_link_libraries(cuda_transform_lib PUBLIC ${CUDART_LIBRARY})
  endif()
else()
  message(STATUS "CUDA not found - will use CPU fallback")
  add_definitions(-DNO_CUDA)
endif()

# Create executable for pc2reheader_transform
add_executable(pc2reheader_transform src/pc2reheader_transform.cpp)

# Set dependencies first (uses plain signature)
ament_target_dependencies(pc2reheader_transform ${dependencies})

# Link CUDA library if available (after ament_target_dependencies, but use plain signature to match)
if(CUDAToolkit_FOUND)
  target_link_libraries(pc2reheader_transform cuda_transform_lib)
  
  # Also link CUDA runtime directly to executable to ensure symbols are resolved
  if(CUDART_LIBRARY)
    target_link_libraries(pc2reheader_transform ${CUDART_LIBRARY})
  endif()
endif()

# Install executable
install(TARGETS
  pc2reheader_transform
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files if any
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
  )
endif()

ament_package()

