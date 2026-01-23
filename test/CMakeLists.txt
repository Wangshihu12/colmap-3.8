# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Test executable CMakeLists.txt for test_vpgo
# 用于编译SE3位姿图优化测试程序

# 设置CMake策略，允许跨目录链接库
cmake_policy(SET CMP0079 NEW)

# 设置目标名称
set(TEST_VPGO_TARGET test_vpgo)
set(TEST_VPGO_SIM3_TARGET test_vpgo_sim3)
set(TEST_LOOP_EDGES_EVAL_TARGET test_loop_edges_eval)

# 添加可执行文件
add_executable(${TEST_VPGO_TARGET} test_vpgo.cc)
add_executable(${TEST_VPGO_SIM3_TARGET} test_vpgo_sim3.cc)
add_executable(${TEST_LOOP_EDGES_EVAL_TARGET} test_loop_edges_eval.cc)

# 设置include路径
target_include_directories(${TEST_VPGO_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/lib
    ${COLMAP_INCLUDE_DIRS}
)

target_include_directories(${TEST_VPGO_SIM3_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/lib
    ${COLMAP_INCLUDE_DIRS}
)

target_include_directories(${TEST_LOOP_EDGES_EVAL_TARGET} PRIVATE
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

target_link_libraries(${TEST_VPGO_SIM3_TARGET}
    colmap
    ${COLMAP_INTERNAL_LIBRARIES}
    ${COLMAP_EXTERNAL_LIBRARIES}
)

target_link_libraries(${TEST_LOOP_EDGES_EVAL_TARGET}
    colmap
    ${COLMAP_INTERNAL_LIBRARIES}
    ${COLMAP_EXTERNAL_LIBRARIES}
)

# 如果启用CUDA，还需要链接colmap_cuda库
if(CUDA_ENABLED)
    target_link_libraries(${TEST_VPGO_TARGET} colmap_cuda)
    target_link_libraries(${TEST_VPGO_SIM3_TARGET} colmap_cuda)
    target_link_libraries(${TEST_LOOP_EDGES_EVAL_TARGET} colmap_cuda)
endif()

# 设置输出目录
set_target_properties(${TEST_VPGO_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(${TEST_VPGO_SIM3_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(${TEST_LOOP_EDGES_EVAL_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 打印编译信息
message(STATUS "Configuring test_vpgo executable")
