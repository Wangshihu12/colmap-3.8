// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "estimators/two_view_geometry.h"

#include <unordered_set>

#include "base/camera.h"
#include "base/essential_matrix.h"
#include "base/homography_matrix.h"
#include "base/pose.h"
#include "base/projection.h"
#include "base/triangulation.h"
#include "estimators/essential_matrix.h"
#include "estimators/fundamental_matrix.h"
#include "estimators/homography_matrix.h"
#include "estimators/translation_transform.h"
#include "optim/loransac.h"
#include "optim/ransac.h"
#include "util/random.h"

namespace colmap {
namespace {

FeatureMatches ExtractInlierMatches(const FeatureMatches& matches,
                                    const size_t num_inliers,
                                    const std::vector<char>& inlier_mask) {
  FeatureMatches inlier_matches(num_inliers);
  size_t j = 0;
  for (size_t i = 0; i < matches.size(); ++i) {
    if (inlier_mask[i]) {
      inlier_matches[j] = matches[i];
      j += 1;
    }
  }
  return inlier_matches;
}

FeatureMatches ExtractOutlierMatches(const FeatureMatches& matches,
                                     const FeatureMatches& inlier_matches) {
  CHECK_GE(matches.size(), inlier_matches.size());

  std::unordered_set<std::pair<point2D_t, point2D_t>> inlier_matches_set;
  inlier_matches_set.reserve(inlier_matches.size());
  for (const auto& match : inlier_matches) {
    inlier_matches_set.emplace(match.point2D_idx1, match.point2D_idx2);
  }

  FeatureMatches outlier_matches;
  outlier_matches.reserve(matches.size() - inlier_matches.size());

  for (const auto& match : matches) {
    if (inlier_matches_set.count(
            std::make_pair(match.point2D_idx1, match.point2D_idx2)) == 0) {
      outlier_matches.push_back(match);
    }
  }

  return outlier_matches;
}

inline bool IsImagePointInBoundingBox(const Eigen::Vector2d& point,
                                      const double minx, const double maxx,
                                      const double miny, const double maxy) {
  return point.x() >= minx && point.x() <= maxx && point.y() >= miny &&
         point.y() <= maxy;
}

}  // namespace

void TwoViewGeometry::Invert() {
  F.transposeInPlace();
  E.transposeInPlace();
  H = H.inverse().eval();

  const Eigen::Vector4d orig_qvec = qvec;
  const Eigen::Vector3d orig_tvec = tvec;
  InvertPose(orig_qvec, orig_tvec, &qvec, &tvec);

  for (auto& match : inlier_matches) {
    std::swap(match.point2D_idx1, match.point2D_idx2);
  }
}

void TwoViewGeometry::Estimate(const Camera& camera1,
                               const std::vector<Eigen::Vector2d>& points1,
                               const Camera& camera2,
                               const std::vector<Eigen::Vector2d>& points2,
                               const FeatureMatches& matches,
                               const Options& options) {
  if (options.force_H_use) {
    EstimateHomography(camera1, points1, camera2, points2, matches, options);
  } else if (camera1.HasPriorFocalLength() && camera2.HasPriorFocalLength()) {
    EstimateCalibrated(camera1, points1, camera2, points2, matches, options);
  } else {
    EstimateUncalibrated(camera1, points1, camera2, points2, matches, options);
  }
}

void TwoViewGeometry::EstimateMultiple(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches, const Options& options) {
  FeatureMatches remaining_matches = matches;
  std::vector<TwoViewGeometry> two_view_geometries;
  while (true) {
    TwoViewGeometry two_view_geometry;
    two_view_geometry.Estimate(camera1, points1, camera2, points2,
                               remaining_matches, options);
    if (two_view_geometry.config == ConfigurationType::DEGENERATE) {
      break;
    }

    if (options.multiple_ignore_watermark) {
      if (two_view_geometry.config != ConfigurationType::WATERMARK) {
        two_view_geometries.push_back(two_view_geometry);
      }
    } else {
      two_view_geometries.push_back(two_view_geometry);
    }

    remaining_matches = ExtractOutlierMatches(remaining_matches,
                                              two_view_geometry.inlier_matches);
  }

  if (two_view_geometries.empty()) {
    config = ConfigurationType::DEGENERATE;
  } else if (two_view_geometries.size() == 1) {
    *this = two_view_geometries[0];
  } else {
    config = ConfigurationType::MULTIPLE;

    for (const auto& two_view_geometry : two_view_geometries) {
      inlier_matches.insert(inlier_matches.end(),
                            two_view_geometry.inlier_matches.begin(),
                            two_view_geometry.inlier_matches.end());
    }
  }
}

/**
 * [功能描述]：估计两个视图之间的相对位姿（旋转和平移）。
 *            根据不同的几何配置类型，使用本质矩阵或单应矩阵来恢复相对位姿。
 * @param camera1：第一个相机的内参模型。
 * @param points1：第一张图像中的2D特征点坐标集合。
 * @param camera2：第二个相机的内参模型。
 * @param points2：第二张图像中的2D特征点坐标集合。
 * @return bool：位姿估计成功返回true，否则返回false。
 */
bool TwoViewGeometry::EstimateRelativePose(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2) {
  // 检查几何配置类型是否有效，只有以下配置才能估计相对位姿：
  // CALIBRATED: 已标定相机，UNCALIBRATED: 未标定相机
  // PLANAR: 平面场景，PANORAMIC: 全景/纯旋转场景
  // PLANAR_OR_PANORAMIC: 待确定是平面还是全景场景
  if (config != CALIBRATED && config != UNCALIBRATED && config != PLANAR &&
      config != PANORAMIC && config != PLANAR_OR_PANORAMIC) {
    return false;
  }

  // 提取归一化的内点坐标（从图像坐标转换到归一化相机坐标）
  std::vector<Eigen::Vector2d> inlier_points1_normalized;
  inlier_points1_normalized.reserve(inlier_matches.size());
  std::vector<Eigen::Vector2d> inlier_points2_normalized;
  inlier_points2_normalized.reserve(inlier_matches.size());
  
  // 遍历所有内点匹配，将图像坐标转换为归一化坐标
  for (const auto& match : inlier_matches) {
    const point2D_t idx1 = match.point2D_idx1;  // 第一张图像中的点索引
    const point2D_t idx2 = match.point2D_idx2;  // 第二张图像中的点索引
    // ImageToWorld: 去除相机内参影响，得到归一化平面上的坐标
    inlier_points1_normalized.push_back(camera1.ImageToWorld(points1[idx1]));
    inlier_points2_normalized.push_back(camera2.ImageToWorld(points2[idx2]));
  }

  Eigen::Matrix3d R;                      // 旋转矩阵 [3x3]
  std::vector<Eigen::Vector3d> points3D;  // 三角化得到的3D点集合

  // 根据几何配置类型选择不同的位姿恢复方法
  if (config == CALIBRATED || config == UNCALIBRATED) {
    // 对于已标定和未标定配置，使用本质矩阵E恢复位姿
    // 注意：未标定情况下可能导致病态重建，但有时经过后续的BA优化仍可成功
    PoseFromEssentialMatrix(E, inlier_points1_normalized,
                            inlier_points2_normalized, &R, &tvec, &points3D);
  } else if (config == PLANAR || config == PANORAMIC ||
             config == PLANAR_OR_PANORAMIC) {
    // 对于平面/全景配置，使用单应矩阵H恢复位姿
    Eigen::Vector3d n;  // 平面法向量（仅平面场景有意义）
    PoseFromHomographyMatrix(
        H, camera1.CalibrationMatrix(), camera2.CalibrationMatrix(),
        inlier_points1_normalized, inlier_points2_normalized, &R, &tvec, &n,
        &points3D);
  } else {
    return false;
  }

  // 将旋转矩阵转换为四元数表示，存储到成员变量qvec中
  qvec = RotationMatrixToQuaternion(R);

  // 计算三角化角度（用于评估重建质量）
  if (points3D.empty()) {
    tri_angle = 0;  // 无3D点时角度为0
  } else {
    // 计算所有3D点的三角化角度的中值
    // 第一个相机在原点，第二个相机位置为 -R^T * t
    tri_angle = Median(CalculateTriangulationAngles(
        Eigen::Vector3d::Zero(), -R.transpose() * tvec, points3D));
  }

  // 对于待确定类型的配置，根据平移向量判断具体是平面还是全景场景
  if (config == PLANAR_OR_PANORAMIC) {
    if (tvec.norm() == 0) {
      // 平移为零，说明是纯旋转（全景场景）
      config = PANORAMIC;
      tri_angle = 0;  // 纯旋转无法三角化，角度为0
    } else {
      // 有平移，说明是平面场景
      config = PLANAR;
    }
  }

  return true;
}

/**
 * [功能描述]：估计已标定相机的两视图几何关系。
 *            通过RANSAC估计本质矩阵E、基础矩阵F和单应矩阵H，
 *            并根据各模型的内点比例自动判断场景的几何配置类型。
 * @param camera1：第一个相机的内参模型。
 * @param points1：第一张图像中的2D特征点坐标集合。
 * @param camera2：第二个相机的内参模型。
 * @param points2：第二张图像中的2D特征点坐标集合。
 * @param matches：两张图像之间的特征点匹配关系。
 * @param options：两视图几何估计的配置选项。
 */
void TwoViewGeometry::EstimateCalibrated(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches, const Options& options) {
  // 验证配置选项的有效性
  options.Check();

  // 如果匹配点数量不足，标记为退化配置并返回
  if (matches.size() < options.min_num_inliers) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // ========== 第一步：提取匹配点坐标 ==========
  // 分别存储原始图像坐标和归一化相机坐标
  std::vector<Eigen::Vector2d> matched_points1(matches.size());           // 图像1的原始坐标
  std::vector<Eigen::Vector2d> matched_points2(matches.size());           // 图像2的原始坐标
  std::vector<Eigen::Vector2d> matched_points1_normalized(matches.size()); // 图像1的归一化坐标
  std::vector<Eigen::Vector2d> matched_points2_normalized(matches.size()); // 图像2的归一化坐标
  
  for (size_t i = 0; i < matches.size(); ++i) {
    const point2D_t idx1 = matches[i].point2D_idx1;
    const point2D_t idx2 = matches[i].point2D_idx2;
    matched_points1[i] = points1[idx1];
    matched_points2[i] = points2[idx2];
    // ImageToWorld: 将图像坐标转换为归一化相机坐标（去除内参影响）
    matched_points1_normalized[i] = camera1.ImageToWorld(points1[idx1]);
    matched_points2_normalized[i] = camera2.ImageToWorld(points2[idx2]);
  }

  // ========== 第二步：估计对极几何模型 ==========

  // 2.1 估计本质矩阵E（使用归一化坐标，需要将误差阈值也转换到归一化空间）
  auto E_ransac_options = options.ransac_options;
  // 将图像空间的误差阈值转换为归一化空间的阈值（取两相机的平均值）
  E_ransac_options.max_error =
      (camera1.ImageToWorldThreshold(options.ransac_options.max_error) +
       camera2.ImageToWorldThreshold(options.ransac_options.max_error)) /
      2;

  // 使用五点法估计本质矩阵，配合LO-RANSAC进行鲁棒估计
  LORANSAC<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator>
      E_ransac(E_ransac_options);
  const auto E_report =
      E_ransac.Estimate(matched_points1_normalized, matched_points2_normalized);
  E = E_report.model;  // 存储估计的本质矩阵 [3x3]

  // 2.2 估计基础矩阵F（使用原始图像坐标）
  // 使用七点法估计，八点法进行局部优化
  LORANSAC<FundamentalMatrixSevenPointEstimator,
           FundamentalMatrixEightPointEstimator>
      F_ransac(options.ransac_options);
  const auto F_report = F_ransac.Estimate(matched_points1, matched_points2);
  F = F_report.model;  // 存储估计的基础矩阵 [3x3]

  // 2.3 估计单应矩阵H（用于检测平面场景或纯旋转）
  LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(
      options.ransac_options);
  const auto H_report = H_ransac.Estimate(matched_points1, matched_points2);
  H = H_report.model;  // 存储估计的单应矩阵 [3x3]

  // 如果所有模型都估计失败，或者所有模型的内点数都不足，标记为退化配置
  if ((!E_report.success && !F_report.success && !H_report.success) ||
      (E_report.support.num_inliers < options.min_num_inliers &&
       F_report.support.num_inliers < options.min_num_inliers &&
       H_report.support.num_inliers < options.min_num_inliers)) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // ========== 第三步：计算各模型之间的内点比例 ==========
  // 这些比例用于判断场景的几何配置类型

  // E相对于F的内点比例（用于判断是否为已标定场景）
  const double E_F_inlier_ratio =
      static_cast<double>(E_report.support.num_inliers) /
      F_report.support.num_inliers;
  // H相对于F的内点比例（用于检测平面/全景场景）
  const double H_F_inlier_ratio =
      static_cast<double>(H_report.support.num_inliers) /
      F_report.support.num_inliers;
  // H相对于E的内点比例（用于检测平面/全景场景）
  const double H_E_inlier_ratio =
      static_cast<double>(H_report.support.num_inliers) /
      E_report.support.num_inliers;

  const std::vector<char>* best_inlier_mask = nullptr;  // 最佳模型的内点掩码
  size_t num_inliers = 0;                                // 最佳模型的内点数量

  // ========== 第四步：根据内点比例判断几何配置类型 ==========

  if (E_report.success && E_F_inlier_ratio > options.min_E_F_inlier_ratio &&
      E_report.support.num_inliers >= options.min_num_inliers) {
    // 情况1：已标定配置（本质矩阵E有效且E/F内点比例足够高）

    // 选择内点数最多的模型（E或F）
    if (E_report.support.num_inliers >= F_report.support.num_inliers) {
      num_inliers = E_report.support.num_inliers;
      best_inlier_mask = &E_report.inlier_mask;
    } else {
      num_inliers = F_report.support.num_inliers;
      best_inlier_mask = &F_report.inlier_mask;
    }

    // 检查是否为平面/全景场景（H内点比例过高表明可能是平面或纯旋转）
    if (H_E_inlier_ratio > options.max_H_inlier_ratio) {
      config = PLANAR_OR_PANORAMIC;
      // 如果H的内点更多，使用H的内点掩码
      if (H_report.support.num_inliers > num_inliers) {
        num_inliers = H_report.support.num_inliers;
        best_inlier_mask = &H_report.inlier_mask;
      }
    } else {
      config = ConfigurationType::CALIBRATED;
    }
  } else if (F_report.success &&
             F_report.support.num_inliers >= options.min_num_inliers) {
    // 情况2：未标定配置（本质矩阵E不满足条件，但基础矩阵F有效）

    num_inliers = F_report.support.num_inliers;
    best_inlier_mask = &F_report.inlier_mask;

    // 同样检查是否为平面/全景场景
    if (H_F_inlier_ratio > options.max_H_inlier_ratio) {
      config = ConfigurationType::PLANAR_OR_PANORAMIC;
      if (H_report.support.num_inliers > num_inliers) {
        num_inliers = H_report.support.num_inliers;
        best_inlier_mask = &H_report.inlier_mask;
      }
    } else {
      config = ConfigurationType::UNCALIBRATED;
    }
  } else if (H_report.success &&
             H_report.support.num_inliers >= options.min_num_inliers) {
    // 情况3：只有单应矩阵H有效（纯平面场景或全景场景）
    num_inliers = H_report.support.num_inliers;
    best_inlier_mask = &H_report.inlier_mask;
    config = ConfigurationType::PLANAR_OR_PANORAMIC;
  } else {
    // 情况4：所有模型都不满足条件，标记为退化配置
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // ========== 第五步：后处理 ==========
  if (best_inlier_mask != nullptr) {
    // 提取内点匹配
    inlier_matches =
        ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);

    // 可选：检测水印（某些图像可能包含固定水印导致虚假匹配）
    if (options.detect_watermark &&
        DetectWatermark(camera1, matched_points1, camera2, matched_points2,
                        num_inliers, *best_inlier_mask, options)) {
      config = ConfigurationType::WATERMARK;
    }

    // 可选：计算相对位姿（旋转和平移）
    if (options.compute_relative_pose) {
      EstimateRelativePose(camera1, points1, camera2, points2);
    }
  }
}

void TwoViewGeometry::EstimateUncalibrated(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches, const Options& options) {
  options.Check();

  if (matches.size() < options.min_num_inliers) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // Extract corresponding points.
  std::vector<Eigen::Vector2d> matched_points1(matches.size());
  std::vector<Eigen::Vector2d> matched_points2(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    matched_points1[i] = points1[matches[i].point2D_idx1];
    matched_points2[i] = points2[matches[i].point2D_idx2];
  }

  // Estimate epipolar model.

  LORANSAC<FundamentalMatrixSevenPointEstimator,
           FundamentalMatrixEightPointEstimator>
      F_ransac(options.ransac_options);
  const auto F_report = F_ransac.Estimate(matched_points1, matched_points2);
  F = F_report.model;

  // Estimate planar or panoramic model.

  LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(
      options.ransac_options);
  const auto H_report = H_ransac.Estimate(matched_points1, matched_points2);
  H = H_report.model;

  if ((!F_report.success && !H_report.success) ||
      (F_report.support.num_inliers < options.min_num_inliers &&
       H_report.support.num_inliers < options.min_num_inliers)) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // Determine inlier ratios of different models.

  const double H_F_inlier_ratio =
      static_cast<double>(H_report.support.num_inliers) /
      F_report.support.num_inliers;

  const std::vector<char>* best_inlier_mask = &F_report.inlier_mask;
  size_t num_inliers = F_report.support.num_inliers;

  if (H_F_inlier_ratio > options.max_H_inlier_ratio) {
    config = ConfigurationType::PLANAR_OR_PANORAMIC;
    if (H_report.support.num_inliers >= F_report.support.num_inliers) {
      num_inliers = H_report.support.num_inliers;
      best_inlier_mask = &H_report.inlier_mask;
    }
  } else {
    config = ConfigurationType::UNCALIBRATED;
  }

  inlier_matches =
      ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);

  if (options.detect_watermark &&
      DetectWatermark(camera1, matched_points1, camera2, matched_points2,
                      num_inliers, *best_inlier_mask, options)) {
    config = ConfigurationType::WATERMARK;
  }

  if (options.compute_relative_pose) {
      EstimateRelativePose(camera1, points1, camera2, points2);
  }
}

void TwoViewGeometry::EstimateHomography(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const FeatureMatches& matches, const Options& options) {
  options.Check();

  if (matches.size() < options.min_num_inliers) {
    config = ConfigurationType::DEGENERATE;
    return;
  }

  // Extract corresponding points.
  std::vector<Eigen::Vector2d> matched_points1(matches.size());
  std::vector<Eigen::Vector2d> matched_points2(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    matched_points1[i] = points1[matches[i].point2D_idx1];
    matched_points2[i] = points2[matches[i].point2D_idx2];
  }

  // Estimate planar or panoramic model.

  LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(
      options.ransac_options);
  const auto H_report = H_ransac.Estimate(matched_points1, matched_points2);
  H = H_report.model;

  if (!H_report.success ||
      H_report.support.num_inliers < options.min_num_inliers) {
    config = ConfigurationType::DEGENERATE;
    return;
  } else {
    config = ConfigurationType::PLANAR_OR_PANORAMIC;
  }

  inlier_matches = ExtractInlierMatches(matches, H_report.support.num_inliers,
                                        H_report.inlier_mask);
  if (options.detect_watermark &&
      DetectWatermark(camera1, matched_points1, camera2, matched_points2,
                      H_report.support.num_inliers, H_report.inlier_mask,
                      options)) {
    config = ConfigurationType::WATERMARK;
  }

  if (options.compute_relative_pose) {
      EstimateRelativePose(camera1, points1, camera2, points2);
  }
}

bool TwoViewGeometry::DetectWatermark(
    const Camera& camera1, const std::vector<Eigen::Vector2d>& points1,
    const Camera& camera2, const std::vector<Eigen::Vector2d>& points2,
    const size_t num_inliers, const std::vector<char>& inlier_mask,
    const Options& options) {
  options.Check();

  // Check if inlier points in border region and extract inlier matches.

  const double diagonal1 = std::sqrt(camera1.Width() * camera1.Width() +
                                     camera1.Height() * camera1.Height());
  const double diagonal2 = std::sqrt(camera2.Width() * camera2.Width() +
                                     camera2.Height() * camera2.Height());
  const double minx1 = options.watermark_border_size * diagonal1;
  const double miny1 = minx1;
  const double maxx1 = camera1.Width() - minx1;
  const double maxy1 = camera1.Height() - miny1;
  const double minx2 = options.watermark_border_size * diagonal2;
  const double miny2 = minx2;
  const double maxx2 = camera2.Width() - minx2;
  const double maxy2 = camera2.Height() - miny2;

  std::vector<Eigen::Vector2d> inlier_points1(num_inliers);
  std::vector<Eigen::Vector2d> inlier_points2(num_inliers);

  size_t num_matches_in_border = 0;

  size_t j = 0;
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const auto& point1 = points1[i];
      const auto& point2 = points2[i];

      inlier_points1[j] = point1;
      inlier_points2[j] = point2;
      j += 1;

      if (!IsImagePointInBoundingBox(point1, minx1, maxx1, miny1, maxy1) &&
          !IsImagePointInBoundingBox(point2, minx2, maxx2, miny2, maxy2)) {
        num_matches_in_border += 1;
      }
    }
  }

  const double matches_in_border_ratio =
      static_cast<double>(num_matches_in_border) / num_inliers;

  if (matches_in_border_ratio < options.watermark_min_inlier_ratio) {
    return false;
  }

  // Check if matches follow a translational model.

  RANSACOptions ransac_options = options.ransac_options;
  ransac_options.min_inlier_ratio = options.watermark_min_inlier_ratio;

  LORANSAC<TranslationTransformEstimator<2>, TranslationTransformEstimator<2>>
      ransac(ransac_options);
  const auto report = ransac.Estimate(inlier_points1, inlier_points2);

  const double inlier_ratio =
      static_cast<double>(report.support.num_inliers) / num_inliers;

  return inlier_ratio >= options.watermark_min_inlier_ratio;
}

}  // namespace colmap
