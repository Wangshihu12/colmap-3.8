// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "glomap/poselib_relpose.h"

#include "glomap/global_mapper.h"

#include <array>
#include <exception>
#include <vector>

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust.h>

#include "base/reconstruction.h"
#include "util/logging.h"

namespace colmap {
namespace {

// 将 COLMAP 相机模型转换为 PoseLib 可识别的相机描述。
// 这里直接复用相机模型名和参数向量，尽量避免二次解释带来的模型偏差。
poselib::Camera ConvertCameraToPoseLibCamera(const Camera& camera) {
  return poselib::Camera(
      camera.ModelName(),
      camera.Params(),
      static_cast<int>(camera.Width()),
      static_cast<int>(camera.Height()));
}

// 从当前位姿图边的内点匹配中提取两张图上的像素坐标。
// points1 / points2 形状：
// - points1: [N, 2]，N 为该边当前内点数
// - points2: [N, 2]，与 points1 按匹配顺序一一对应
void CollectMatchedPoints(const ViewEdge& edge,
                          const Reconstruction& reconstruction,
                          std::vector<Eigen::Vector2d>* points1,
                          std::vector<Eigen::Vector2d>* points2) {
  points1->clear();
  points2->clear();
  points1->reserve(edge.geometry.inlier_matches.size());
  points2->reserve(edge.geometry.inlier_matches.size());

  const Image& image1 = reconstruction.Image(edge.image_id1);
  const Image& image2 = reconstruction.Image(edge.image_id2);
  for (const FeatureMatch& match : edge.geometry.inlier_matches) {
    points1->push_back(image1.Point2D(match.point2D_idx1).XY());
    points2->push_back(image2.Point2D(match.point2D_idx2).XY());
  }
}

}  // namespace

size_t EstimateRelativePosesWithPoseLib(const PoseLibRelPoseOptions& options,
                                        std::vector<ViewEdge>* edges,
                                        const Reconstruction& reconstruction) {
  CHECK_NOTNULL(edges);

  size_t num_success = 0;
  for (ViewEdge& edge : *edges) {
    // 只重估当前仍然有效的边，避免把 calibration 已经判掉的异常边重新放回图里。
    if (!edge.valid || edge.weight <= 0.0) {
      continue;
    }
    if (edge.geometry.inlier_matches.size() < 5) {
      continue;
    }

    std::vector<Eigen::Vector2d> points1;
    std::vector<Eigen::Vector2d> points2;
    CollectMatchedPoints(edge, reconstruction, &points1, &points2);

    const Camera& camera1 = reconstruction.Camera(
        reconstruction.Image(edge.image_id1).CameraId());
    const Camera& camera2 = reconstruction.Camera(
        reconstruction.Image(edge.image_id2).CameraId());

    poselib::RelativePoseOptions relpose_options;
    relpose_options.ransac.max_iterations = options.max_iterations;
    relpose_options.max_error = options.max_epipolar_error;
    relpose_options.bundle.max_iterations = options.max_bundle_iterations;

    poselib::CameraPose relative_pose;
    std::vector<char> inlier_mask;
    try {
      // PoseLib 内部会做鲁棒相对位姿估计和非线性优化。
      // inlier_mask 形状为 [N]，1 表示该匹配被保留为内点。
      poselib::estimate_relative_pose(points1,
                                      points2,
                                      ConvertCameraToPoseLibCamera(camera1),
                                      ConvertCameraToPoseLibCamera(camera2),
                                      relpose_options,
                                      &relative_pose,
                                      &inlier_mask);
    } catch (const std::exception& e) {
      LOG(WARNING) << "PoseLib relative pose failed for pair "
                   << edge.image_id1 << "-" << edge.image_id2
                   << ": " << e.what();
      continue;
    }

    if (inlier_mask.empty()) {
      continue;
    }

    std::vector<FeatureMatch> refined_inliers;
    refined_inliers.reserve(inlier_mask.size());
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
      if (inlier_mask[i]) {
        refined_inliers.push_back(edge.geometry.inlier_matches[i]);
      }
    }
    if (refined_inliers.size() < 5) {
      continue;
    }

    edge.geometry.inlier_matches = std::move(refined_inliers);
    edge.weight = static_cast<double>(edge.geometry.inlier_matches.size());

    // 使用旧位姿图的平移方向作为符号参考，避免边与边之间出现方向翻转。
    // 相对位姿的平移本身只确定到符号，如果不做这一步，图里相邻边可能朝向相反。
    if (edge.geometry.tvec.dot(relative_pose.t) < 0.0) {
      relative_pose.t *= -1.0;
    }

    edge.geometry.qvec(0) = relative_pose.q(0);
    edge.geometry.qvec(1) = relative_pose.q(1);
    edge.geometry.qvec(2) = relative_pose.q(2);
    edge.geometry.qvec(3) = relative_pose.q(3);
    edge.geometry.tvec = relative_pose.t;
    edge.geometry.config = TwoViewGeometry::CALIBRATED;
    ++num_success;
  }

  return num_success;
}

}  // namespace colmap
