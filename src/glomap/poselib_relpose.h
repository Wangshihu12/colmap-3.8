// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#ifndef COLMAP_SRC_GLOMAP_POSELIB_RELPOSE_H_
#define COLMAP_SRC_GLOMAP_POSELIB_RELPOSE_H_

#include <cstddef>
#include <vector>

namespace colmap {

class Reconstruction;
struct ViewEdge;

struct PoseLibRelPoseOptions {
  // PoseLib RANSAC 最大迭代次数。
  size_t max_iterations = 50000;

  // Sampson 误差阈值，单位为像素。
  double max_epipolar_error = 2.0;

  // 非线性优化最大迭代次数。
  size_t max_bundle_iterations = 100;
};

/**
 * @brief 功能描述：使用 PoseLib 对位姿图中的每条边重新估计相对位姿。
 *
 * 这里的输入 edges 仍然沿用当前兼容层里的 ViewEdge 结构。函数会直接更新：
 * 1. `geometry.qvec / geometry.tvec`
 * 2. `geometry.inlier_matches`
 * 3. `weight`
 *
 * 注意：当前只处理仍然有效的边，避免将 calibration 已经剔除的边重新放回图中。
 * @param options PoseLib 相对位姿求解选项。
 * @param edges 输入输出位姿图边集合，函数会原位更新相对位姿和内点。
 * @param reconstruction 当前重建对象，用于读取相机与 2D 特征。
 * @return 返回值说明：成功重估的边数。
 */
size_t EstimateRelativePosesWithPoseLib(const PoseLibRelPoseOptions& options,
                                        std::vector<ViewEdge>* edges,
                                        const Reconstruction& reconstruction);

}  // namespace colmap

#endif  // COLMAP_SRC_GLOMAP_POSELIB_RELPOSE_H_
