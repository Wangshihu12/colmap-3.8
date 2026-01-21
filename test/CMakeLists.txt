# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Test executable CMakeLists.txt for test_vpgo
# 用于编译SE3位姿图优化测试程序

# 设置目标名称
set(TEST_VPGO_TARGET test_vpgo)

# 添加可执行文件
add_executable(${TEST_VPGO_TARGET} test_vpgo.cc)

# 设置include路径
target_include_directories(${TEST_VPGO_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/lib
    ${COLMAP_INCLUDE_DIRS}
)

# 链接COLMAP库和外部依赖
target_link_libraries(${TEST_VPGO_TARGET}
    colmap
    ${COLMAP_INTERNAL_LIBRARIES}
    ${COLMAP_EXTERNAL_LIBRARIES}
)

# 如果启用CUDA，还需要链接colmap_cuda库
if(CUDA_ENABLED)
    target_link_libraries(${TEST_VPGO_TARGET} colmap_cuda)
endif()

# 设置输出目录
set_target_properties(${TEST_VPGO_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 打印编译信息
message(STATUS "Configuring test_vpgo executable")
