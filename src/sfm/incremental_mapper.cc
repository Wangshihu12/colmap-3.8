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

#include "sfm/incremental_mapper.h"

#include <array>
#include <fstream>

#include "base/pose.h"
#include "base/projection.h"
#include "base/triangulation.h"
#include "estimators/pose.h"
#include "util/bitmap.h"
#include "util/misc.h"

namespace colmap {
namespace {

void SortAndAppendNextImages(std::vector<std::pair<image_t, float>> image_ranks,
                             std::vector<image_t>* sorted_images_ids) {
  std::sort(image_ranks.begin(), image_ranks.end(),
            [](const std::pair<image_t, float>& image1,
               const std::pair<image_t, float>& image2) {
              return image1.second > image2.second;
            });

  sorted_images_ids->reserve(sorted_images_ids->size() + image_ranks.size());
  for (const auto& image : image_ranks) {
    sorted_images_ids->push_back(image.first);
  }

  image_ranks.clear();
}

float RankNextImageMaxVisiblePointsNum(const Image& image) {
  return static_cast<float>(image.NumVisiblePoints3D());
}

float RankNextImageMaxVisiblePointsRatio(const Image& image) {
  return static_cast<float>(image.NumVisiblePoints3D()) /
         static_cast<float>(image.NumObservations());
}

float RankNextImageMinUncertainty(const Image& image) {
  return static_cast<float>(image.Point3DVisibilityScore());
}

}  // namespace

bool IncrementalMapper::Options::Check() const {
  CHECK_OPTION_GT(init_min_num_inliers, 0);
  CHECK_OPTION_GT(init_max_error, 0.0);
  CHECK_OPTION_GE(init_max_forward_motion, 0.0);
  CHECK_OPTION_LE(init_max_forward_motion, 1.0);
  CHECK_OPTION_GE(init_min_tri_angle, 0.0);
  CHECK_OPTION_GE(init_max_reg_trials, 1);
  CHECK_OPTION_GT(abs_pose_max_error, 0.0);
  CHECK_OPTION_GT(abs_pose_min_num_inliers, 0);
  CHECK_OPTION_GE(abs_pose_min_inlier_ratio, 0.0);
  CHECK_OPTION_LE(abs_pose_min_inlier_ratio, 1.0);
  CHECK_OPTION_GE(local_ba_num_images, 2);
  CHECK_OPTION_GE(local_ba_min_tri_angle, 0.0);
  CHECK_OPTION_GE(min_focal_length_ratio, 0.0);
  CHECK_OPTION_GE(max_focal_length_ratio, min_focal_length_ratio);
  CHECK_OPTION_GE(max_extra_param, 0.0);
  CHECK_OPTION_GE(filter_max_reproj_error, 0.0);
  CHECK_OPTION_GE(filter_min_tri_angle, 0.0);
  CHECK_OPTION_GE(max_reg_trials, 1);
  return true;
}

IncrementalMapper::IncrementalMapper(const DatabaseCache* database_cache)
    : database_cache_(database_cache),
      reconstruction_(nullptr),
      triangulator_(nullptr),
      num_total_reg_images_(0),
      num_shared_reg_images_(0),
      prev_init_image_pair_id_(kInvalidImagePairId) {}

/**
 * [功能描述]：开始一个新的增量式重建过程。
 *            该函数初始化重建所需的各种数据结构，包括加载数据库信息、
 *            设置对应关系图、创建三角化器，并重置各种状态变量。
 * @param reconstruction：指向重建对象的指针，用于存储和管理重建结果
 */
void IncrementalMapper::BeginReconstruction(Reconstruction* reconstruction) {
  // 确保当前没有正在进行的重建（防止重复调用）
  CHECK(reconstruction_ == nullptr);
  
  // 保存重建对象指针
  reconstruction_ = reconstruction;
  
  // 从数据库缓存加载图像、相机和特征点等信息到重建对象
  reconstruction_->Load(*database_cache_);
  
  // 设置重建对象，关联对应关系图（特征点匹配关系）
  reconstruction_->SetUp(&database_cache_->CorrespondenceGraph());
  
  // 创建增量三角化器，用于后续的3D点三角化操作
  // 三角化器需要对应关系图和重建对象来工作
  triangulator_ = std::make_unique<IncrementalTriangulator>(
      &database_cache_->CorrespondenceGraph(), reconstruction);

  // 初始化共享注册图像计数器（用于多模型重建时跟踪模型间的重叠）
  num_shared_reg_images_ = 0;
  
  // 清空每个相机的注册图像数量映射表
  num_reg_images_per_camera_.clear();
  
  // 遍历已注册的图像，为每个图像触发注册事件
  // 这会更新相关的统计信息（如每个相机的注册图像数）
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    RegisterImageEvent(image_id);
  }

  // 记录重建开始时已存在的图像ID集合
  // 用于区分导入的图像和新注册的图像
  existing_image_ids_ =
      std::unordered_set<image_t>(reconstruction->RegImageIds().begin(),
                                  reconstruction->RegImageIds().end());

  // 重置上一次初始化图像对的ID为无效值
  prev_init_image_pair_id_ = kInvalidImagePairId;
  // 重置上一次初始化的双视图几何信息
  prev_init_two_view_geometry_ = TwoViewGeometry();

  // 清空被过滤图像的集合（记录因质量问题被排除的图像）
  filtered_images_.clear();
  // 清空每张图像的注册尝试次数记录
  num_reg_trials_.clear();
}

void IncrementalMapper::EndReconstruction(const bool discard) {
  CHECK_NOTNULL(reconstruction_);

  if (discard) {
    for (const image_t image_id : reconstruction_->RegImageIds()) {
      DeRegisterImageEvent(image_id);
    }
  }

  reconstruction_->TearDown();
  reconstruction_ = nullptr;
  triangulator_.reset();
}

bool IncrementalMapper::FindInitialImagePair(const Options& options,
                                             image_t* image_id1,
                                             image_t* image_id2) {
  CHECK(options.Check());

  std::vector<image_t> image_ids1;
  if (*image_id1 != kInvalidImageId && *image_id2 == kInvalidImageId) {
    // Only *image_id1 provided.
    if (!database_cache_->ExistsImage(*image_id1)) {
      return false;
    }
    image_ids1.push_back(*image_id1);
  } else if (*image_id1 == kInvalidImageId && *image_id2 != kInvalidImageId) {
    // Only *image_id2 provided.
    if (!database_cache_->ExistsImage(*image_id2)) {
      return false;
    }
    image_ids1.push_back(*image_id2);
  } else {
    // No initial seed image provided.
    image_ids1 = FindFirstInitialImage(options);
  }

  // Try to find good initial pair.
  for (size_t i1 = 0; i1 < image_ids1.size(); ++i1) {
    *image_id1 = image_ids1[i1];

    const std::vector<image_t> image_ids2 =
        FindSecondInitialImage(options, *image_id1);

    for (size_t i2 = 0; i2 < image_ids2.size(); ++i2) {
      *image_id2 = image_ids2[i2];

      const image_pair_t pair_id =
          Database::ImagePairToPairId(*image_id1, *image_id2);

      // Try every pair only once.
      if (init_image_pairs_.count(pair_id) > 0) {
        continue;
      }

      init_image_pairs_.insert(pair_id);

      if (EstimateInitialTwoViewGeometry(options, *image_id1, *image_id2)) {
        return true;
      }
    }
  }

  // No suitable pair found in entire dataset.
  *image_id1 = kInvalidImageId;
  *image_id2 = kInvalidImageId;

  return false;
}

std::vector<image_t> IncrementalMapper::FindNextImages(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  std::function<float(const Image&)> rank_image_func;
  switch (options.image_selection_method) {
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_NUM:
      rank_image_func = RankNextImageMaxVisiblePointsNum;
      break;
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_RATIO:
      rank_image_func = RankNextImageMaxVisiblePointsRatio;
      break;
    case Options::ImageSelectionMethod::MIN_UNCERTAINTY:
      rank_image_func = RankNextImageMinUncertainty;
      break;
  }

  std::vector<std::pair<image_t, float>> image_ranks;
  std::vector<std::pair<image_t, float>> other_image_ranks;

  // Append images that have not failed to register before.
  for (const auto& image : reconstruction_->Images()) {
    // Skip images that are already registered.
    if (image.second.IsRegistered()) {
      continue;
    }

    // Only consider images with a sufficient number of visible points.
    if (image.second.NumVisiblePoints3D() <
        static_cast<size_t>(options.abs_pose_min_num_inliers)) {
      continue;
    }

    // Only try registration for a certain maximum number of times.
    const size_t num_reg_trials = num_reg_trials_[image.first];
    if (num_reg_trials >= static_cast<size_t>(options.max_reg_trials)) {
      continue;
    }

    // If image has been filtered or failed to register, place it in the
    // second bucket and prefer images that have not been tried before.
    const float rank = rank_image_func(image.second);
    if (filtered_images_.count(image.first) == 0 && num_reg_trials == 0) {
      image_ranks.emplace_back(image.first, rank);
    } else {
      other_image_ranks.emplace_back(image.first, rank);
    }
  }

  std::vector<image_t> ranked_images_ids;
  SortAndAppendNextImages(image_ranks, &ranked_images_ids);
  SortAndAppendNextImages(other_image_ranks, &ranked_images_ids);

  return ranked_images_ids;
}

/**
 * [功能描述]：注册初始图像对并进行三角化，建立重建的初始结构。
 *            这是增量式SfM的第一步，通过两张图像估计相对位姿，
 *            并三角化匹配的特征点生成初始的3D点云。
 * @param options：增量映射器的配置选项
 * @param image_id1：第一张图像的ID
 * @param image_id2：第二张图像的ID
 * @return 初始图像对注册是否成功
 */
bool IncrementalMapper::RegisterInitialImagePair(const Options& options,
                                                 const image_t image_id1,
                                                 const image_t image_id2) {
  // 确保重建对象存在
  CHECK_NOTNULL(reconstruction_);
  // 确保当前没有已注册的图像（这是初始化步骤）
  CHECK_EQ(reconstruction_->NumRegImages(), 0);

  // 验证配置选项的有效性
  CHECK(options.Check());

  // 更新两张图像的初始化注册尝试次数
  init_num_reg_trials_[image_id1] += 1;
  init_num_reg_trials_[image_id2] += 1;
  // 更新两张图像的总注册尝试次数
  num_reg_trials_[image_id1] += 1;
  num_reg_trials_[image_id2] += 1;

  // 生成图像对的唯一ID并记录到已尝试的初始图像对集合中
  const image_pair_t pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  init_image_pairs_.insert(pair_id);

  // 获取两张图像及其对应相机的引用
  Image& image1 = reconstruction_->Image(image_id1);
  const Camera& camera1 = reconstruction_->Camera(image1.CameraId());

  Image& image2 = reconstruction_->Image(image_id2);
  const Camera& camera2 = reconstruction_->Camera(image2.CameraId());

  //////////////////////////////////////////////////////////////////////////////
  // 估计双视图几何 - 计算两张图像之间的相对位姿
  //////////////////////////////////////////////////////////////////////////////

  // 估计初始双视图几何（本质矩阵/基础矩阵分解得到相对位姿）
  if (!EstimateInitialTwoViewGeometry(options, image_id1, image_id2)) {
    return false;
  }

  // 设置第一张图像的位姿为世界坐标系原点（单位旋转，零平移）
  // Qvec：四元数表示的旋转，形状为(4,)，单位四元数[1,0,0,0]表示无旋转
  image1.Qvec() = ComposeIdentityQuaternion();
  // Tvec：平移向量，形状为(3,)，[0,0,0]表示在原点
  image1.Tvec() = Eigen::Vector3d(0, 0, 0);
  
  // 设置第二张图像的位姿为相对于第一张图像的位姿
  // 从双视图几何估计结果中获取
  image2.Qvec() = prev_init_two_view_geometry_.qvec;
  image2.Tvec() = prev_init_two_view_geometry_.tvec;

  // 计算两张图像的投影矩阵 P = K[R|t]，形状为(3,4)
  const Eigen::Matrix3x4d proj_matrix1 = image1.ProjectionMatrix();
  const Eigen::Matrix3x4d proj_matrix2 = image2.ProjectionMatrix();
  // 计算两张图像的相机中心（光心）在世界坐标系中的位置，形状为(3,)
  const Eigen::Vector3d proj_center1 = image1.ProjectionCenter();
  const Eigen::Vector3d proj_center2 = image2.ProjectionCenter();

  //////////////////////////////////////////////////////////////////////////////
  // 更新重建 - 注册图像并三角化3D点
  //////////////////////////////////////////////////////////////////////////////

  // 将两张图像注册到重建中
  reconstruction_->RegisterImage(image_id1);
  reconstruction_->RegisterImage(image_id2);
  // 触发图像注册事件，更新相关统计信息
  RegisterImageEvent(image_id1);
  RegisterImageEvent(image_id2);

  // 获取对应关系图（存储特征点匹配关系）
  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();
  // 获取两张图像之间的特征匹配
  const FeatureMatches& corrs =
      correspondence_graph.FindCorrespondencesBetweenImages(image_id1,
                                                            image_id2);

  // 将最小三角化角度从度转换为弧度
  const double min_tri_angle_rad = DegToRad(options.init_min_tri_angle);

  // 创建3D点的轨迹（Track）结构，用于记录哪些2D点观测到同一个3D点
  Track track;
  track.Reserve(2);  // 预分配空间，初始轨迹包含2个观测
  track.AddElement(TrackElement());
  track.AddElement(TrackElement());
  // 设置轨迹元素对应的图像ID
  track.Element(0).image_id = image_id1;
  track.Element(1).image_id = image_id2;
  
  // 遍历所有匹配的特征点对，进行三角化
  for (const auto& corr : corrs) {
    // 将图像坐标转换为归一化相机坐标（去除相机内参的影响）
    // point1_N, point2_N 形状为(2,)，表示归一化平面上的点
    const Eigen::Vector2d point1_N =
        camera1.ImageToWorld(image1.Point2D(corr.point2D_idx1).XY());
    const Eigen::Vector2d point2_N =
        camera2.ImageToWorld(image2.Point2D(corr.point2D_idx2).XY());
    
    // 通过三角化计算3D点坐标
    // xyz 形状为(3,)，表示世界坐标系中的3D点位置
    const Eigen::Vector3d& xyz =
        TriangulatePoint(proj_matrix1, proj_matrix2, point1_N, point2_N);
    
    // 计算三角化角度（两条视线的夹角）
    // 角度越大，三角化精度越高；角度过小会导致深度估计不准确
    const double tri_angle =
        CalculateTriangulationAngle(proj_center1, proj_center2, xyz);
    
    // 验证三角化结果的有效性：
    // 1. 三角化角度需大于最小阈值（确保足够的基线）
    // 2. 3D点在两个相机前方（正深度检查）
    if (tri_angle >= min_tri_angle_rad &&
        HasPointPositiveDepth(proj_matrix1, xyz) &&
        HasPointPositiveDepth(proj_matrix2, xyz)) {
      // 设置轨迹元素对应的2D点索引
      track.Element(0).point2D_idx = corr.point2D_idx1;
      track.Element(1).point2D_idx = corr.point2D_idx2;
      // 将有效的3D点添加到重建中
      reconstruction_->AddPoint3D(xyz, track);
    }
  }

  return true;
}

/**
 * [功能描述]：在增量式重建过程中注册下一张图像。
 *            通过PnP算法（2D-3D对应）估计图像的绝对位姿，
 *            然后将图像注册到重建中并扩展3D点的观测轨迹。
 * @param options：增量映射器的配置选项
 * @param image_id：待注册图像的ID
 * @return 图像注册是否成功
 */
bool IncrementalMapper::RegisterNextImage(const Options& options,
                                          const image_t image_id) {
  // 确保重建对象存在
  CHECK_NOTNULL(reconstruction_);
  // 确保至少已有2张图像注册（初始图像对）
  CHECK_GE(reconstruction_->NumRegImages(), 2);

  // 验证配置选项的有效性
  CHECK(options.Check());

  // 获取待注册图像及其对应相机的引用
  Image& image = reconstruction_->Image(image_id);
  Camera& camera = reconstruction_->Camera(image.CameraId());

  // 确保图像未被重复注册
  CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

  // 增加该图像的注册尝试次数
  num_reg_trials_[image_id] += 1;

  // 快速检查：如果可见的3D点数量不足，直接返回失败
  // 这是一个预筛选，避免不必要的计算
  if (image.NumVisiblePoints3D() <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 搜索2D-3D对应关系 - 建立图像特征点与已有3D点的匹配
  //////////////////////////////////////////////////////////////////////////////

  // 获取对应关系图
  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();

  // 存储2D-3D对应关系的容器
  // tri_corrs: 存储(2D点索引, 3D点ID)对
  std::vector<std::pair<point2D_t, point3D_t>> tri_corrs;
  // tri_points2D: 2D点的图像坐标，形状为(N, 2)
  std::vector<Eigen::Vector2d> tri_points2D;
  // tri_points3D: 对应3D点的世界坐标，形状为(N, 3)
  std::vector<Eigen::Vector3d> tri_points3D;

  // 用于去重的3D点ID集合
  std::unordered_set<point3D_t> corr_point3D_ids;
  
  // 遍历当前图像的所有2D特征点
  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const Point2D& point2D = image.Point2D(point2D_idx);

    // 清空当前2D点对应的3D点ID集合（用于同一2D点的去重）
    corr_point3D_ids.clear();
    
    // 查找与当前2D点匹配的所有对应点
    for (const auto& corr :
         correspondence_graph.FindCorrespondences(image_id, point2D_idx)) {
      // 获取对应点所在的图像
      const Image& corr_image = reconstruction_->Image(corr.image_id);
      // 跳过未注册的图像（这些图像的位姿未知）
      if (!corr_image.IsRegistered()) {
        continue;
      }

      // 获取对应的2D点
      const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
      // 跳过没有关联3D点的2D点
      if (!corr_point2D.HasPoint3D()) {
        continue;
      }

      // 避免重复的对应关系（同一个3D点可能被多张图像观测到）
      if (corr_point3D_ids.count(corr_point2D.Point3DId()) > 0) {
        continue;
      }

      // 获取对应图像的相机参数
      const Camera& corr_camera =
          reconstruction_->Camera(corr_image.CameraId());

      // 跳过具有异常相机参数的图像（如焦距不合理）
      if (corr_camera.HasBogusParams(options.min_focal_length_ratio,
                                     options.max_focal_length_ratio,
                                     options.max_extra_param)) {
        continue;
      }

      // 获取3D点
      const Point3D& point3D =
          reconstruction_->Point3D(corr_point2D.Point3DId());

      // 记录有效的2D-3D对应关系
      tri_corrs.emplace_back(point2D_idx, corr_point2D.Point3DId());
      corr_point3D_ids.insert(corr_point2D.Point3DId());
      tri_points2D.push_back(point2D.XY());      // 2D点坐标
      tri_points3D.push_back(point3D.XYZ());     // 3D点坐标
    }
  }

  // 再次检查2D-3D对应数量是否足够
  // 由于跳过了异常相机参数的图像，实际对应数可能少于预期
  if (tri_points2D.size() <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 2D-3D位姿估计 - 使用PnP+RANSAC估计相机绝对位姿
  //////////////////////////////////////////////////////////////////////////////

  // 焦距估计/优化策略：
  // 仅当焦距未指定（手动或EXIF）且未从共享相机的其他图像估计过时才估计焦距

  // 配置绝对位姿估计选项
  AbsolutePoseEstimationOptions abs_pose_options;
  abs_pose_options.num_threads = options.num_threads;           // 并行线程数
  abs_pose_options.num_focal_length_samples = 30;               // 焦距采样数
  abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;  // 最小焦距比例
  abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;  // 最大焦距比例
  abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;    // RANSAC最大重投影误差
  abs_pose_options.ransac_options.min_inlier_ratio =
      options.abs_pose_min_inlier_ratio;                        // 最小内点比例
  // 使用高置信度避免P3P RANSAC过早终止
  // 过早终止可能导致注册失败
  abs_pose_options.ransac_options.min_num_trials = 100;         // 最小迭代次数
  abs_pose_options.ransac_options.max_num_trials = 10000;       // 最大迭代次数
  abs_pose_options.ransac_options.confidence = 0.99999;         // 置信度阈值

  // 配置位姿优化选项
  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  
  // 根据相机是否已被其他图像优化过来决定是否优化相机参数
  if (num_reg_images_per_camera_[image.CameraId()] > 0) {
    // 相机已被其他共享该相机的图像优化过
    if (camera.HasBogusParams(options.min_focal_length_ratio,
                              options.max_focal_length_ratio,
                              options.max_extra_param)) {
      // 之前优化得到的相机参数异常，重置参数并重新估计
      camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
      abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
      abs_pose_refinement_options.refine_focal_length = true;
      abs_pose_refinement_options.refine_extra_params = true;
    } else {
      // 相机参数正常，不再估计/优化
      abs_pose_options.estimate_focal_length = false;
      abs_pose_refinement_options.refine_focal_length = false;
      abs_pose_refinement_options.refine_extra_params = false;
    }
  } else {
    // 相机之前未被优化过
    // 注意：相机参数可能已被修改但图像被过滤，所以显式重置参数并重新估计
    camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
    abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
    abs_pose_refinement_options.refine_focal_length = true;
    abs_pose_refinement_options.refine_extra_params = true;
  }

  // 如果配置不允许优化焦距，则禁用焦距估计和优化
  if (!options.abs_pose_refine_focal_length) {
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
  }

  // 如果配置不允许优化额外参数（如畸变系数），则禁用
  if (!options.abs_pose_refine_extra_params) {
    abs_pose_refinement_options.refine_extra_params = false;
  }

  // 内点数量和内点掩码
  size_t num_inliers;
  std::vector<char> inlier_mask;  // 标记每个对应是否为内点

  // 使用P3P+RANSAC估计绝对位姿
  // 输出：图像的旋转四元数(Qvec)、平移向量(Tvec)、相机参数、内点信息
  if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D,
                            &image.Qvec(), &image.Tvec(), &camera, &num_inliers,
                            &inlier_mask)) {
    return false;
  }

  // 检查内点数量是否满足最小要求
  if (num_inliers < static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 位姿优化 - 非线性优化精化位姿和相机参数
  //////////////////////////////////////////////////////////////////////////////

  // 使用非线性优化（如Levenberg-Marquardt）精化位姿
  // 仅使用内点进行优化，最小化重投影误差
  if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask,
                          tri_points2D, tri_points3D, &image.Qvec(),
                          &image.Tvec(), &camera)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 扩展轨迹 - 将新观测添加到已有3D点的轨迹中
  //////////////////////////////////////////////////////////////////////////////

  // 正式将图像注册到重建中
  reconstruction_->RegisterImage(image_id);
  // 触发注册事件，更新统计信息
  RegisterImageEvent(image_id);

  // 遍历所有内点对应，将新图像的观测添加到3D点轨迹中
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const point2D_t point2D_idx = tri_corrs[i].first;
      const Point2D& point2D = image.Point2D(point2D_idx);
      // 只有当该2D点还未关联到任何3D点时才添加观测
      // （避免重复添加）
      if (!point2D.HasPoint3D()) {
        const point3D_t point3D_id = tri_corrs[i].second;
        // 创建轨迹元素（记录哪张图像的哪个2D点观测到该3D点）
        const TrackElement track_el(image_id, point2D_idx);
        // 将观测添加到3D点的轨迹中
        reconstruction_->AddObservation(point3D_id, track_el);
        // 标记该3D点已被修改，以便后续三角化更新
        triangulator_->AddModifiedPoint3D(point3D_id);
      }
    }
  }

  return true;
}

size_t IncrementalMapper::TriangulateImage(
    const IncrementalTriangulator::Options& tri_options,
    const image_t image_id) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->TriangulateImage(tri_options, image_id);
}

size_t IncrementalMapper::Retriangulate(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->Retriangulate(tri_options);
}

size_t IncrementalMapper::CompleteTracks(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->CompleteAllTracks(tri_options);
}

size_t IncrementalMapper::MergeTracks(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->MergeAllTracks(tri_options);
}

/**
 * [功能描述]：执行局部光束法平差（Local Bundle Adjustment）优化。
 *            该函数对新注册图像及其邻近图像组成的局部区域进行优化，
 *            包括相机位姿、3D点位置的联合优化，以及轨迹合并、补全和过滤。
 * @param options：增量映射器的配置选项
 * @param ba_options：光束法平差的配置选项（如损失函数类型、迭代次数等）
 * @param tri_options：三角化的配置选项
 * @param image_id：新注册图像的ID，局部BA以该图像为中心
 * @param point3D_ids：被修改的3D点ID集合，这些点需要被优化
 * @return LocalBundleAdjustmentReport：包含优化统计信息的报告
 */
IncrementalMapper::LocalBundleAdjustmentReport
IncrementalMapper::AdjustLocalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options, const image_t image_id,
    const std::unordered_set<point3D_t>& point3D_ids) {
  // 确保重建对象存在
  CHECK_NOTNULL(reconstruction_);
  // 验证配置选项的有效性
  CHECK(options.Check());

  // 创建局部BA报告，用于返回优化统计信息
  LocalBundleAdjustmentReport report;

  // 查找与当前图像共享最多3D点的图像，构成局部图像集
  // 这些图像将参与局部BA优化
  const std::vector<image_t> local_bundle = FindLocalBundle(options, image_id);

  // 只有存在连接的图像时才执行BA
  if (local_bundle.size() > 0) {
    // 配置BA参数
    BundleAdjustmentConfig ba_config;
    // 添加当前图像到BA
    ba_config.AddImage(image_id);
    // 添加局部图像集中的所有图像
    for (const image_t local_image_id : local_bundle) {
      ba_config.AddImage(local_image_id);
    }

    // 添加相对位姿约束（如果有的话）到BA配置中
    AddRelativePoseConstraintsToConfig(&ba_config);

    // 如果选项指定，则固定已存在的图像位姿（用于断点续传场景）
    if (options.fix_existing_images) {
      for (const image_t local_image_id : local_bundle) {
        // 检查图像是否是导入的已存在图像
        if (existing_image_ids_.count(local_image_id)) {
          ba_config.SetConstantPose(local_image_id);
        }
      }
    }

    // 确定哪些相机需要固定
    // 当某相机的部分图像不在局部图像集中时，需要固定该相机参数
    // 统计局部图像集中每个相机的图像数量
    std::unordered_map<camera_t, size_t> num_images_per_camera;
    for (const image_t image_id : ba_config.Images()) {
      const Image& image = reconstruction_->Image(image_id);
      num_images_per_camera[image.CameraId()] += 1;
    }

    // 如果某相机在局部集中的图像数少于其总注册图像数，则固定该相机
    // 原因：部分图像不参与优化，优化相机参数会导致不一致
    for (const auto& camera_id_and_num_images_pair : num_images_per_camera) {
      const size_t num_reg_images_for_camera =
          num_reg_images_per_camera_.at(camera_id_and_num_images_pair.first);
      if (camera_id_and_num_images_pair.second < num_reg_images_for_camera) {
        ba_config.SetConstantCamera(camera_id_and_num_images_pair.first);
      }
    }

    // 固定7个自由度（DoF）以避免BA中的尺度/旋转/平移漂移
    // SfM的7-DoF不确定性：3个平移 + 3个旋转 + 1个尺度
    const bool has_rel_constraints =
        ba_config.NumRelativePoseConstraints() > 0;
    
    if (local_bundle.size() == 1) {
      // 只有一张邻近图像时：固定该图像的位姿（6 DoF）
      ba_config.SetConstantPose(local_bundle[0]);
      // 如果没有相对位姿约束，还需固定当前图像平移的一个分量（1 DoF）
      if (!has_rel_constraints) {
        ba_config.SetConstantTvec(image_id, {0});  // 固定平移向量的第一个分量
      }
    } else if (local_bundle.size() > 1) {
      // 有多张邻近图像时：固定最后一张图像的位姿（6 DoF）
      const image_t image_id1 = local_bundle[local_bundle.size() - 1];
      const image_t image_id2 = local_bundle[local_bundle.size() - 2];
      ba_config.SetConstantPose(image_id1);
      // 固定倒数第二张图像平移的一个分量（1 DoF）以固定尺度
      if (!has_rel_constraints &&
          (!options.fix_existing_images ||
           !existing_image_ids_.count(image_id2))) {
        ba_config.SetConstantTvec(image_id2, {0});
      }
    }

    // 确定哪些3D点参与优化
    // 优化所有新的和短轨迹的3D点，无论它们是否完全包含在局部图像集中
    // 不优化长轨迹3D点，因为它们通常已经非常稳定，
    // 将它们加入BA和轨迹合并/补全会显著降低局部BA的速度
    std::unordered_set<point3D_t> variable_point3D_ids;
    for (const point3D_t point3D_id : point3D_ids) {
      const Point3D& point3D = reconstruction_->Point3D(point3D_id);
      const size_t kMaxTrackLength = 15;  // 轨迹长度阈值
      // 如果点没有误差信息（新点）或轨迹较短，则作为变量参与优化
      if (!point3D.HasError() || point3D.Track().Length() <= kMaxTrackLength) {
        ba_config.AddVariablePoint(point3D_id);
        variable_point3D_ids.insert(point3D_id);
      }
    }

    // 执行局部光束法平差
    BundleAdjuster bundle_adjuster(ba_options, ba_config);
    bundle_adjuster.Solve(reconstruction_);

    // 记录调整的观测数量（残差数除以2，因为每个观测有x和y两个残差）
    report.num_adjusted_observations =
        bundle_adjuster.Summary().num_residuals / 2;

    // 合并优化后的轨迹与其他已有点
    // 可能有不同的3D点实际上是同一个物理点，需要合并
    report.num_merged_observations =
        triangulator_->MergeTracks(tri_options, variable_point3D_ids);
    
    // 补全轨迹：尝试为可能在BA前三角化失败的点添加新观测
    // 原因：BA优化了相机位姿和标定，之前失败的三角化现在可能成功
    // 这可以避免一些点被过滤，并有助于后续图像注册
    report.num_completed_observations =
        triangulator_->CompleteTracks(tri_options, variable_point3D_ids);
    // 为当前图像补全可能的观测
    report.num_completed_observations +=
        triangulator_->CompleteImage(tri_options, image_id);
  }

  // 过滤离群点：对修改过的图像和所有变化的3D点进行过滤
  // 确保模型中没有离群点
  // 注意：这会导致一些重复工作，因为很多3D点可能同时在调整的图像中，
  // 但过滤不是瓶颈，所以可以接受
  std::unordered_set<image_t> filter_image_ids;
  filter_image_ids.insert(image_id);
  filter_image_ids.insert(local_bundle.begin(), local_bundle.end());
  
  // 过滤图像中的3D点（基于重投影误差和三角化角度）
  report.num_filtered_observations = reconstruction_->FilterPoints3DInImages(
      options.filter_max_reproj_error, options.filter_min_tri_angle,
      filter_image_ids);
  // 过滤指定的3D点集合
  report.num_filtered_observations += reconstruction_->FilterPoints3D(
      options.filter_max_reproj_error, options.filter_min_tri_angle,
      point3D_ids);

  return report;
}

bool IncrementalMapper::AdjustGlobalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options) {
  CHECK_NOTNULL(reconstruction_);

  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                       "registered for global "
                                       "bundle-adjustment";

  // Avoid degeneracies in bundle adjustment.
  reconstruction_->FilterObservationsWithNegativeDepth();

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  AddRelativePoseConstraintsToConfig(&ba_config);

  // Fix the existing images, if option specified.
  if (options.fix_existing_images) {
    for (const image_t image_id : reg_image_ids) {
      if (existing_image_ids_.count(image_id)) {
        ba_config.SetConstantPose(image_id);
      }
    }
  }

  // Fix 7-DOFs of the bundle adjustment problem.
  const bool has_rel_constraints =
      ba_config.NumRelativePoseConstraints() > 0;
  ba_config.SetConstantPose(reg_image_ids[0]);
  if (!has_rel_constraints &&
      (!options.fix_existing_images ||
       !existing_image_ids_.count(reg_image_ids[1]))) {
    ba_config.SetConstantTvec(reg_image_ids[1], {0});
  }

  // Run bundle adjustment.
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  if (!bundle_adjuster.Solve(reconstruction_)) {
    return false;
  }

  if (options.normalize_scene) {
    // Normalize scene for numerical stability and
    // to avoid large scale changes in viewer.
    reconstruction_->Normalize();
  }

  return true;
}

bool IncrementalMapper::AdjustParallelGlobalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options,
    const ParallelBundleAdjuster::Options& parallel_ba_options) {
  CHECK_NOTNULL(reconstruction_);

  const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

  CHECK_GE(reg_image_ids.size(), 2)
      << "At least two images must be registered for global bundle-adjustment";

  // Avoid degeneracies in bundle adjustment.
  reconstruction_->FilterObservationsWithNegativeDepth();

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // Run bundle adjustment.
  ParallelBundleAdjuster bundle_adjuster(parallel_ba_options, ba_options,
                                         ba_config);
  if (!bundle_adjuster.Solve(reconstruction_)) {
    return false;
  }

  if (options.normalize_scene) {
    // Normalize scene for numerical stability and
    // to avoid large scale changes in viewer.
    reconstruction_->Normalize();
  }

  return true;
}

size_t IncrementalMapper::FilterImages(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  // Do not filter images in the early stage of the reconstruction, since the
  // calibration is often still refining a lot. Hence, the camera parameters
  // are not stable in the beginning.
  const size_t kMinNumImages = 20;
  if (reconstruction_->NumRegImages() < kMinNumImages) {
    return {};
  }

  const std::vector<image_t> image_ids = reconstruction_->FilterImages(
      options.min_focal_length_ratio, options.max_focal_length_ratio,
      options.max_extra_param);

  for (const image_t image_id : image_ids) {
    DeRegisterImageEvent(image_id);
    filtered_images_.insert(image_id);
  }

  return image_ids.size();
}

size_t IncrementalMapper::FilterPoints(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());
  return reconstruction_->FilterAllPoints3D(options.filter_max_reproj_error,
                                            options.filter_min_tri_angle);
}

const Reconstruction& IncrementalMapper::GetReconstruction() const {
  CHECK_NOTNULL(reconstruction_);
  return *reconstruction_;
}

size_t IncrementalMapper::NumTotalRegImages() const {
  return num_total_reg_images_;
}

size_t IncrementalMapper::NumSharedRegImages() const {
  return num_shared_reg_images_;
}

const std::unordered_set<point3D_t>& IncrementalMapper::GetModifiedPoints3D() {
  return triangulator_->GetModifiedPoints3D();
}

void IncrementalMapper::ClearModifiedPoints3D() {
  triangulator_->ClearModifiedPoints3D();
}

void IncrementalMapper::SetRelativePoseConstraints(
    const std::vector<RelativePoseConstraint>& constraints) {
  relative_pose_constraints_ = constraints;
}

void IncrementalMapper::AddRelativePoseConstraintsToConfig(
    BundleAdjustmentConfig* config) const {
  if (relative_pose_constraints_.empty()) {
    return;
  }

  for (const auto& constraint : relative_pose_constraints_) {
    if (config->HasImage(constraint.image_id1) &&
        config->HasImage(constraint.image_id2)) {
      config->AddRelativePoseConstraint(constraint);
    }
  }
}

std::vector<image_t> IncrementalMapper::FindFirstInitialImage(
    const Options& options) const {
  // Struct to hold meta-data for ranking images.
  struct ImageInfo {
    image_t image_id;
    bool prior_focal_length;
    image_t num_correspondences;
  };

  const size_t init_max_reg_trials =
      static_cast<size_t>(options.init_max_reg_trials);

  // Collect information of all not yet registered images with
  // correspondences.
  std::vector<ImageInfo> image_infos;
  image_infos.reserve(reconstruction_->NumImages());
  for (const auto& image : reconstruction_->Images()) {
    // Only images with correspondences can be registered.
    if (image.second.NumCorrespondences() == 0) {
      continue;
    }

    // Only use images for initialization a maximum number of times.
    if (init_num_reg_trials_.count(image.first) &&
        init_num_reg_trials_.at(image.first) >= init_max_reg_trials) {
      continue;
    }

    // Only use images for initialization that are not registered in any
    // of the other reconstructions.
    if (num_registrations_.count(image.first) > 0 &&
        num_registrations_.at(image.first) > 0) {
      continue;
    }

    const class Camera& camera =
        reconstruction_->Camera(image.second.CameraId());
    ImageInfo image_info;
    image_info.image_id = image.first;
    image_info.prior_focal_length = camera.HasPriorFocalLength();
    image_info.num_correspondences = image.second.NumCorrespondences();
    image_infos.push_back(image_info);
  }

  // Sort images such that images with a prior focal length and more
  // correspondences are preferred, i.e. they appear in the front of the list.
  std::sort(
      image_infos.begin(), image_infos.end(),
      [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
        if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
          return true;
        } else if (!image_info1.prior_focal_length &&
                   image_info2.prior_focal_length) {
          return false;
        } else {
          return image_info1.num_correspondences >
                 image_info2.num_correspondences;
        }
      });

  // Extract image identifiers in sorted order.
  std::vector<image_t> image_ids;
  image_ids.reserve(image_infos.size());
  for (const ImageInfo& image_info : image_infos) {
    image_ids.push_back(image_info.image_id);
  }

  return image_ids;
}

std::vector<image_t> IncrementalMapper::FindSecondInitialImage(
    const Options& options, const image_t image_id1) const {
  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();

  // Collect images that are connected to the first seed image and have
  // not been registered before in other reconstructions.
  const class Image& image1 = reconstruction_->Image(image_id1);
  std::unordered_map<image_t, point2D_t> num_correspondences;
  for (point2D_t point2D_idx = 0; point2D_idx < image1.NumPoints2D();
       ++point2D_idx) {
    for (const auto& corr :
         correspondence_graph.FindCorrespondences(image_id1, point2D_idx)) {
      if (num_registrations_.count(corr.image_id) == 0 ||
          num_registrations_.at(corr.image_id) == 0) {
        num_correspondences[corr.image_id] += 1;
      }
    }
  }

  // Struct to hold meta-data for ranking images.
  struct ImageInfo {
    image_t image_id;
    bool prior_focal_length;
    point2D_t num_correspondences;
  };

  const size_t init_min_num_inliers =
      static_cast<size_t>(options.init_min_num_inliers);

  // Compose image information in a compact form for sorting.
  std::vector<ImageInfo> image_infos;
  image_infos.reserve(reconstruction_->NumImages());
  for (const auto elem : num_correspondences) {
    if (elem.second >= init_min_num_inliers) {
      const class Image& image = reconstruction_->Image(elem.first);
      const class Camera& camera = reconstruction_->Camera(image.CameraId());
      ImageInfo image_info;
      image_info.image_id = elem.first;
      image_info.prior_focal_length = camera.HasPriorFocalLength();
      image_info.num_correspondences = elem.second;
      image_infos.push_back(image_info);
    }
  }

  // Sort images such that images with a prior focal length and more
  // correspondences are preferred, i.e. they appear in the front of the list.
  std::sort(
      image_infos.begin(), image_infos.end(),
      [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
        if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
          return true;
        } else if (!image_info1.prior_focal_length &&
                   image_info2.prior_focal_length) {
          return false;
        } else {
          return image_info1.num_correspondences >
                 image_info2.num_correspondences;
        }
      });

  // Extract image identifiers in sorted order.
  std::vector<image_t> image_ids;
  image_ids.reserve(image_infos.size());
  for (const ImageInfo& image_info : image_infos) {
    image_ids.push_back(image_info.image_id);
  }

  return image_ids;
}

std::vector<image_t> IncrementalMapper::FindLocalBundle(
    const Options& options, const image_t image_id) const {
  CHECK(options.Check());

  const Image& image = reconstruction_->Image(image_id);
  CHECK(image.IsRegistered());

  // Extract all images that have at least one 3D point with the query image
  // in common, and simultaneously count the number of common 3D points.

  std::unordered_map<image_t, size_t> shared_observations;

  std::unordered_set<point3D_t> point3D_ids;
  point3D_ids.reserve(image.NumPoints3D());

  for (const Point2D& point2D : image.Points2D()) {
    if (point2D.HasPoint3D()) {
      point3D_ids.insert(point2D.Point3DId());
      const Point3D& point3D = reconstruction_->Point3D(point2D.Point3DId());
      for (const TrackElement& track_el : point3D.Track().Elements()) {
        if (track_el.image_id != image_id) {
          shared_observations[track_el.image_id] += 1;
        }
      }
    }
  }

  // Sort overlapping images according to number of shared observations.

  std::vector<std::pair<image_t, size_t>> overlapping_images(
      shared_observations.begin(), shared_observations.end());
  std::sort(overlapping_images.begin(), overlapping_images.end(),
            [](const std::pair<image_t, size_t>& image1,
               const std::pair<image_t, size_t>& image2) {
              return image1.second > image2.second;
            });

  // The local bundle is composed of the given image and its most connected
  // neighbor images, hence the subtraction of 1.

  const size_t num_images =
      static_cast<size_t>(options.local_ba_num_images - 1);
  const size_t num_eff_images = std::min(num_images, overlapping_images.size());

  // Extract most connected images and ensure sufficient triangulation angle.

  std::vector<image_t> local_bundle_image_ids;
  local_bundle_image_ids.reserve(num_eff_images);

  // If the number of overlapping images equals the number of desired images in
  // the local bundle, then simply copy over the image identifiers.
  if (overlapping_images.size() == num_eff_images) {
    for (const auto& overlapping_image : overlapping_images) {
      local_bundle_image_ids.push_back(overlapping_image.first);
    }
    return local_bundle_image_ids;
  }

  // In the following iteration, we start with the most overlapping images and
  // check whether it has sufficient triangulation angle. If none of the
  // overlapping images has sufficient triangulation angle, we relax the
  // triangulation angle threshold and start from the most overlapping image
  // again. In the end, if we still haven't found enough images, we simply use
  // the most overlapping images.

  const double min_tri_angle_rad = DegToRad(options.local_ba_min_tri_angle);

  // The selection thresholds (minimum triangulation angle, minimum number of
  // shared observations), which are successively relaxed.
  const std::array<std::pair<double, double>, 8> selection_thresholds = {{
      std::make_pair(min_tri_angle_rad / 1.0, 0.6 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 1.5, 0.6 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 2.0, 0.5 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 2.5, 0.4 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 3.0, 0.3 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 4.0, 0.2 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 5.0, 0.1 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 6.0, 0.1 * image.NumPoints3D()),
  }};

  const Eigen::Vector3d proj_center = image.ProjectionCenter();
  std::vector<Eigen::Vector3d> shared_points3D;
  shared_points3D.reserve(image.NumPoints3D());
  std::vector<double> tri_angles(overlapping_images.size(), -1.0);
  std::vector<char> used_overlapping_images(overlapping_images.size(), false);

  for (const auto& selection_threshold : selection_thresholds) {
    for (size_t overlapping_image_idx = 0;
         overlapping_image_idx < overlapping_images.size();
         ++overlapping_image_idx) {
      // Check if the image has sufficient overlap. Since the images are ordered
      // based on the overlap, we can just skip the remaining ones.
      if (overlapping_images[overlapping_image_idx].second <
          selection_threshold.second) {
        break;
      }

      // Check if the image is already in the local bundle.
      if (used_overlapping_images[overlapping_image_idx]) {
        continue;
      }

      const auto& overlapping_image = reconstruction_->Image(
          overlapping_images[overlapping_image_idx].first);
      const Eigen::Vector3d overlapping_proj_center =
          overlapping_image.ProjectionCenter();

      // In the first iteration, compute the triangulation angle. In later
      // iterations, reuse the previously computed value.
      double& tri_angle = tri_angles[overlapping_image_idx];
      if (tri_angle < 0.0) {
        // Collect the commonly observed 3D points.
        shared_points3D.clear();
        for (const Point2D& point2D : image.Points2D()) {
          if (point2D.HasPoint3D() && point3D_ids.count(point2D.Point3DId())) {
            shared_points3D.push_back(
                reconstruction_->Point3D(point2D.Point3DId()).XYZ());
          }
        }

        // Calculate the triangulation angle at a certain percentile.
        const double kTriangulationAnglePercentile = 75;
        tri_angle = Percentile(
            CalculateTriangulationAngles(proj_center, overlapping_proj_center,
                                         shared_points3D),
            kTriangulationAnglePercentile);
      }

      // Check that the image has sufficient triangulation angle.
      if (tri_angle >= selection_threshold.first) {
        local_bundle_image_ids.push_back(overlapping_image.ImageId());
        used_overlapping_images[overlapping_image_idx] = true;
        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
          break;
        }
      }
    }

    // Check if we already collected enough images.
    if (local_bundle_image_ids.size() >= num_eff_images) {
      break;
    }
  }

  // In case there are not enough images with sufficient triangulation angle,
  // simply fill up the rest with the most overlapping images.

  if (local_bundle_image_ids.size() < num_eff_images) {
    for (size_t overlapping_image_idx = 0;
         overlapping_image_idx < overlapping_images.size();
         ++overlapping_image_idx) {
      // Collect image if it is not yet in the local bundle.
      if (!used_overlapping_images[overlapping_image_idx]) {
        local_bundle_image_ids.push_back(
            overlapping_images[overlapping_image_idx].first);
        used_overlapping_images[overlapping_image_idx] = true;

        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
          break;
        }
      }
    }
  }

  return local_bundle_image_ids;
}

void IncrementalMapper::RegisterImageEvent(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  size_t& num_reg_images_for_camera =
      num_reg_images_per_camera_[image.CameraId()];
  num_reg_images_for_camera += 1;

  size_t& num_regs_for_image = num_registrations_[image_id];
  num_regs_for_image += 1;
  if (num_regs_for_image == 1) {
    num_total_reg_images_ += 1;
  } else if (num_regs_for_image > 1) {
    num_shared_reg_images_ += 1;
  }
}

void IncrementalMapper::DeRegisterImageEvent(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  size_t& num_reg_images_for_camera =
      num_reg_images_per_camera_.at(image.CameraId());
  CHECK_GT(num_reg_images_for_camera, 0);
  num_reg_images_for_camera -= 1;

  size_t& num_regs_for_image = num_registrations_[image_id];
  num_regs_for_image -= 1;
  if (num_regs_for_image == 0) {
    num_total_reg_images_ -= 1;
  } else if (num_regs_for_image > 0) {
    num_shared_reg_images_ -= 1;
  }
}

bool IncrementalMapper::EstimateInitialTwoViewGeometry(
    const Options& options, const image_t image_id1, const image_t image_id2) {
  const image_pair_t image_pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);

  if (prev_init_image_pair_id_ == image_pair_id) {
    return true;
  }

  const Image& image1 = database_cache_->Image(image_id1);
  const Camera& camera1 = database_cache_->Camera(image1.CameraId());

  const Image& image2 = database_cache_->Image(image_id2);
  const Camera& camera2 = database_cache_->Camera(image2.CameraId());

  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();
  const FeatureMatches matches =
      correspondence_graph.FindCorrespondencesBetweenImages(image_id1,
                                                            image_id2);

  std::vector<Eigen::Vector2d> points1;
  points1.reserve(image1.NumPoints2D());
  for (const auto& point : image1.Points2D()) {
    points1.push_back(point.XY());
  }

  std::vector<Eigen::Vector2d> points2;
  points2.reserve(image2.NumPoints2D());
  for (const auto& point : image2.Points2D()) {
    points2.push_back(point.XY());
  }

  TwoViewGeometry two_view_geometry;
  TwoViewGeometry::Options two_view_geometry_options;
  two_view_geometry_options.ransac_options.min_num_trials = 30;
  two_view_geometry_options.ransac_options.max_error = options.init_max_error;
  two_view_geometry.EstimateCalibrated(camera1, points1, camera2, points2,
                                       matches, two_view_geometry_options);

  if (!two_view_geometry.EstimateRelativePose(camera1, points1, camera2,
                                              points2)) {
    return false;
  }

  if (static_cast<int>(two_view_geometry.inlier_matches.size()) >=
          options.init_min_num_inliers &&
      std::abs(two_view_geometry.tvec.z()) < options.init_max_forward_motion &&
      two_view_geometry.tri_angle > DegToRad(options.init_min_tri_angle)) {
    prev_init_image_pair_id_ = image_pair_id;
    prev_init_two_view_geometry_ = two_view_geometry;
    return true;
  }

  return false;
}

}  // namespace colmap
