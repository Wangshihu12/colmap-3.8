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

#include "controllers/incremental_mapper.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <unordered_map>

#include "base/pose.h"
#include "util/misc.h"
#include "util/string.h"

namespace colmap {
namespace {

size_t TriangulateImage(const IncrementalMapperOptions& options,
                        const Image& image, IncrementalMapper* mapper) {
  std::cout << "  => Continued observations: " << image.NumPoints3D()
            << std::endl;
  const size_t num_tris =
      mapper->TriangulateImage(options.Triangulation(), image.ImageId());
  std::cout << "  => Added observations: " << num_tris << std::endl;
  return num_tris;
}

/**
 * [功能描述]：执行全局光束法平差，根据条件选择并行BA或普通BA
 * @param options：增量式建图的配置选项
 * @param mapper：增量式建图器指针
 */
void AdjustGlobalBundle(const IncrementalMapperOptions& options,
                        IncrementalMapper* mapper) {
  // 获取全局BA的配置选项
  BundleAdjustmentOptions custom_ba_options = options.GlobalBundleAdjustment();

  // 获取当前已注册的图像数量
  const size_t num_reg_images = mapper->GetReconstruction().NumRegImages();

  // 对于前几张图像，使用更严格的收敛准则
  // 因为初始阶段的精度对后续重建质量影响较大
  const size_t kMinNumRegImagesForFastBA = 10;
  if (num_reg_images < kMinNumRegImagesForFastBA) {
    // 收紧各项容差阈值（缩小10倍）
    custom_ba_options.solver_options.function_tolerance /= 10;   // 函数值容差
    custom_ba_options.solver_options.gradient_tolerance /= 10;   // 梯度容差
    custom_ba_options.solver_options.parameter_tolerance /= 10;  // 参数容差
    // 增加迭代次数以确保充分收敛
    custom_ba_options.solver_options.max_num_iterations *= 2;
    custom_ba_options.solver_options.max_linear_solver_iterations = 200;
  }

  PrintHeading1("Global bundle adjustment");

  // 判断是否使用并行BA（PBA）
  // 需同时满足以下条件：
  // 1. 用户启用了并行BA选项
  // 2. 未固定已有图像的位姿
  // 3. 未使用外部相对位姿约束文件
  // 4. 未从数据库读取相对位姿约束
  // 5. 已注册图像数量达到阈值
  // 6. 当前重建满足并行BA的支持条件
  if (options.ba_global_use_pba && !options.fix_existing_images &&
      options.relative_pose_path.empty() &&
      !options.relative_pose_from_database &&
      num_reg_images >= kMinNumRegImagesForFastBA &&
      ParallelBundleAdjuster::IsSupported(custom_ba_options,
                                          mapper->GetReconstruction())) {
    // 使用GPU加速的并行BA
    mapper->AdjustParallelGlobalBundle(
        options.Mapper(), custom_ba_options,
        options.ParallelGlobalBundleAdjustment());
  } else {
    // 使用CPU的普通BA
    mapper->AdjustGlobalBundle(options.Mapper(), custom_ba_options);
  }
}

/**
 * [功能描述]：对新注册图像进行迭代式局部光束法平差（Local Bundle Adjustment）优化。
 *            该函数会迭代执行局部BA，直到观测变化率低于阈值或达到最大迭代次数。
 *            每次迭代会合并轨迹、补全观测、过滤离群点，逐步提高局部重建精度。
 * @param options：增量映射器的配置选项，包含BA相关参数
 * @param image_id：新注册图像的ID，局部BA将以该图像为中心
 * @param mapper：增量映射器指针，用于执行BA操作
 */
void IterativeLocalRefinement(const IncrementalMapperOptions& options,
                              const image_t image_id,
                              IncrementalMapper* mapper) {
  // 获取局部BA的配置选项
  auto ba_options = options.LocalBundleAdjustment();
  
  // 迭代执行局部BA，最多执行 ba_local_max_refinements 次
  for (int i = 0; i < options.ba_local_max_refinements; ++i) {
    // 执行局部光束法平差
    // 参数：映射器选项、BA选项、三角化选项、中心图像ID、被修改的3D点集合
    // 返回：包含优化统计信息的报告
    const auto report = mapper->AdjustLocalBundle(
        options.Mapper(), ba_options, options.Triangulation(), image_id,
        mapper->GetModifiedPoints3D());
    
    // 输出本次迭代的统计信息
    // 合并的观测数：将同一3D点的重复轨迹合并
    std::cout << "  => Merged observations: " << report.num_merged_observations
              << std::endl;
    // 补全的观测数：通过重新三角化添加的新观测
    std::cout << "  => Completed observations: "
              << report.num_completed_observations << std::endl;
    // 过滤的观测数：因重投影误差过大而被移除的观测
    std::cout << "  => Filtered observations: "
              << report.num_filtered_observations << std::endl;
    
    // 计算观测变化率 = (合并数 + 补全数 + 过滤数) / 总调整观测数
    // 该比率反映了本次迭代对重建的修改程度
    const double changed =
        report.num_adjusted_observations == 0
            ? 0
            : (report.num_merged_observations +
               report.num_completed_observations +
               report.num_filtered_observations) /
                  static_cast<double>(report.num_adjusted_observations);
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;
    
    // 如果变化率低于阈值，说明重建已经稳定，提前终止迭代
    if (changed < options.ba_local_max_refinement_change) {
      break;
    }
    
    // 仅在第一次迭代使用鲁棒损失函数（如Huber/Cauchy）
    // 后续迭代使用普通损失函数（TRIVIAL即平方损失）
    // 原因：第一次迭代可能有较多离群点，需要鲁棒损失函数抑制；
    //       后续迭代离群点已被过滤，使用普通损失可获得更精确的优化结果
    ba_options.loss_function_type =
        BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  }
  
  // 清空被修改的3D点集合，为下一次图像注册做准备
  mapper->ClearModifiedPoints3D();
}

/**
 * [功能描述]：迭代式全局优化，交替执行BA、轨迹补全/合并、点过滤
 *            直到观测变化率低于阈值或达到最大迭代次数
 * @param options：增量式建图的配置选项
 * @param mapper：增量式建图器指针
 */
void IterativeGlobalRefinement(const IncrementalMapperOptions& options,
                               IncrementalMapper* mapper) {
  PrintHeading1("Retriangulation");

  // 初始阶段：先完成轨迹补全和合并
  CompleteAndMergeTracks(options, mapper);

  // 执行重三角化，恢复之前可能失败的三维点
  std::cout << "  => Retriangulated observations: "
            << mapper->Retriangulate(options.Triangulation()) << std::endl;

  // 迭代优化循环
  for (int i = 0; i < options.ba_global_max_refinements; ++i) {
    // 记录当前观测数量，用于计算变化率
    const size_t num_observations =
        mapper->GetReconstruction().ComputeNumObservations();
    size_t num_changed_observations = 0;

    // 步骤1：执行全局BA优化
    AdjustGlobalBundle(options, mapper);

    // 步骤2：补全和合并轨迹，统计变化的观测数
    num_changed_observations += CompleteAndMergeTracks(options, mapper);

    // 步骤3：过滤不良三维点，统计变化的观测数
    num_changed_observations += FilterPoints(options, mapper);

    // 计算观测变化率
    const double changed =
        num_observations == 0
            ? 0
            : static_cast<double>(num_changed_observations) / num_observations;
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;

    // 如果变化率低于阈值，说明已收敛，提前退出
    if (changed < options.ba_global_max_refinement_change) {
      break;
    }
  }

  // 最后过滤不良图像
  FilterImages(options, mapper);
}

void ExtractColors(const std::string& image_path, const image_t image_id,
                   Reconstruction* reconstruction) {
  if (!reconstruction->ExtractColorsForImage(image_id, image_path)) {
    std::cout << StringPrintf("WARNING: Could not read image %s at path %s.",
                              reconstruction->Image(image_id).Name().c_str(),
                              image_path.c_str())
              << std::endl;
  }
}

void WriteSnapshot(const Reconstruction& reconstruction,
                   const std::string& snapshot_path) {
  PrintHeading1("Creating snapshot");
  // Get the current timestamp in milliseconds.
  const size_t timestamp =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now().time_since_epoch())
          .count();
  // Write reconstruction to unique path with current timestamp.
  const std::string path =
      JoinPaths(snapshot_path, StringPrintf("%010d", timestamp));
  CreateDirIfNotExists(path);
  std::cout << "  => Writing to " << path << std::endl;
  reconstruction.Write(path);
}

bool TryExtractTimestampFromName(const std::string& name,
                                 int64_t* timestamp) {
  std::string base = name;
  const size_t slash_pos = base.find_last_of("/\\");
  if (slash_pos != std::string::npos) {
    base = base.substr(slash_pos + 1);
  }
  const size_t dot_pos = base.find_last_of('.');
  if (dot_pos != std::string::npos) {
    base = base.substr(0, dot_pos);
  }

  bool candidate_float = true;
  bool has_digit = false;
  bool has_dot = false;
  for (const char ch : base) {
    if (std::isdigit(static_cast<unsigned char>(ch)) != 0) {
      has_digit = true;
      continue;
    }
    if (ch == '.' && !has_dot) {
      has_dot = true;
      continue;
    }
    candidate_float = false;
    break;
  }

  if (candidate_float && has_digit) {
    try {
      const double value = std::stod(base);
      *timestamp = static_cast<int64_t>(std::llround(value * 1e6));
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }

  size_t best_pos = std::string::npos;
  size_t best_len = 0;
  size_t run_start = 0;
  size_t run_len = 0;

  for (size_t i = 0; i <= base.size(); ++i) {
    const bool is_digit =
        (i < base.size()) &&
        std::isdigit(static_cast<unsigned char>(base[i])) != 0;
    if (is_digit) {
      if (run_len == 0) {
        run_start = i;
      }
      run_len += 1;
    } else if (run_len > 0) {
      if (run_len > best_len ||
          (run_len == best_len && run_start > best_pos)) {
        best_len = run_len;
        best_pos = run_start;
      }
      run_len = 0;
    }
  }

  if (best_len == 0) {
    return false;
  }

  try {
    *timestamp = std::stoll(base.substr(best_pos, best_len));
  } catch (const std::exception&) {
    return false;
  }

  return true;
}

std::vector<RelativePoseConstraint> ReadRelativePoseConstraints(
    const std::string& path, const DatabaseCache& database_cache,
    const double default_rot_weight, const double default_trans_weight) {
  std::vector<RelativePoseConstraint> constraints;
  if (path.empty()) {
    return constraints;
  }

  const auto lines = ReadTextFileLines(path);
  constraints.reserve(lines.size());

  for (const auto& line : lines) {
    std::string content = line;
    const auto comment_pos = content.find('#');
    if (comment_pos != std::string::npos) {
      content = content.substr(0, comment_pos);
    }
    StringTrim(&content);
    if (content.empty()) {
      continue;
    }

    const auto elems = StringSplit(content, " \t");
    if (elems.size() < 9) {
      std::cout << "WARNING: Skipping invalid relative pose line: " << line
                << std::endl;
      continue;
    }

    const Image* image1 = database_cache.FindImageWithName(elems[0]);
    const Image* image2 = database_cache.FindImageWithName(elems[1]);
    if (image1 == nullptr || image2 == nullptr) {
      std::cout << "WARNING: Skipping relative pose with unknown image: "
                << line << std::endl;
      continue;
    }

    try {
      Eigen::Vector4d qvec;
      Eigen::Vector3d tvec;
      qvec(0) = std::stod(elems[2]);
      qvec(1) = std::stod(elems[3]);
      qvec(2) = std::stod(elems[4]);
      qvec(3) = std::stod(elems[5]);
      tvec(0) = std::stod(elems[6]);
      tvec(1) = std::stod(elems[7]);
      tvec(2) = std::stod(elems[8]);

      const double qvec_norm = qvec.norm();
      if (qvec_norm == 0.0) {
        std::cout << "WARNING: Skipping relative pose with zero rotation: "
                  << line << std::endl;
        continue;
      }
      qvec /= qvec_norm;

      double rot_weight = default_rot_weight;
      double trans_weight = default_trans_weight;
      if (elems.size() >= 11) {
        rot_weight = std::stod(elems[9]);
        trans_weight = std::stod(elems[10]);
      }

      RelativePoseConstraint constraint;
      constraint.image_id1 = image1->ImageId();
      constraint.image_id2 = image2->ImageId();
      constraint.qvec12 = qvec;
      constraint.tvec12 = tvec;
      constraint.rot_weight = rot_weight;
      constraint.trans_weight = trans_weight;
      constraints.push_back(constraint);
    } catch (const std::exception&) {
      std::cout << "WARNING: Skipping invalid relative pose line: " << line
                << std::endl;
    }
  }

  return constraints;
}

std::vector<RelativePoseConstraint> BuildRelativePoseConstraintsFromDatabase(
    const DatabaseCache& database_cache, const double default_rot_weight,
    const double default_trans_weight) {
  std::unordered_map<camera_t, std::vector<image_t>> camera_image_ids;
  camera_image_ids.reserve(database_cache.NumImages());

  size_t num_missing_priors = 0;

  for (const auto& image_pair : database_cache.Images()) {
    const Image& image = image_pair.second;
    if (!image.HasQvecPrior() || !image.HasTvecPrior()) {
      num_missing_priors += 1;
      continue;
    }

    camera_image_ids[image.CameraId()].push_back(image.ImageId());
  }

  if (num_missing_priors > 0) {
    std::cout << "WARNING: " << num_missing_priors
              << " images missing pose priors; skipped." << std::endl;
  }

  std::vector<RelativePoseConstraint> constraints;
  for (auto& pair : camera_image_ids) {
    auto& image_ids = pair.second;
    if (image_ids.size() < 2) {
      continue;
    }
    std::sort(image_ids.begin(), image_ids.end());
    constraints.reserve(constraints.size() + image_ids.size() - 1);

    for (size_t i = 0; i + 1 < image_ids.size(); ++i) {
      const Image& image1 = database_cache.Image(image_ids[i]);
      const Image& image2 = database_cache.Image(image_ids[i + 1]);

      RelativePoseConstraint constraint;
      constraint.image_id1 = image1.ImageId();
      constraint.image_id2 = image2.ImageId();
      ComputeRelativePose(image1.QvecPrior(), image1.TvecPrior(),
                          image2.QvecPrior(), image2.TvecPrior(),
                          &constraint.qvec12, &constraint.tvec12);
      constraint.qvec12 = NormalizeQuaternion(constraint.qvec12);
      constraint.rot_weight = default_rot_weight;
      constraint.trans_weight = default_trans_weight;
      constraints.push_back(constraint);
    }
  }

  return constraints;
}

}  // namespace

size_t FilterPoints(const IncrementalMapperOptions& options,
                    IncrementalMapper* mapper) {
  const size_t num_filtered_observations =
      mapper->FilterPoints(options.Mapper());
  std::cout << "  => Filtered observations: " << num_filtered_observations
            << std::endl;
  return num_filtered_observations;
}

size_t FilterImages(const IncrementalMapperOptions& options,
                    IncrementalMapper* mapper) {
  const size_t num_filtered_images = mapper->FilterImages(options.Mapper());
  std::cout << "  => Filtered images: " << num_filtered_images << std::endl;
  return num_filtered_images;
}

/**
 * [功能描述]：补全和合并特征轨迹
 * @param options：增量式建图的配置选项
 * @param mapper：增量式建图器指针
 * @return 补全和合并的观测总数
 */
size_t CompleteAndMergeTracks(const IncrementalMapperOptions& options,
                              IncrementalMapper* mapper) {
  // 补全轨迹：为已有三维点添加更多的二维观测
  const size_t num_completed_observations =
      mapper->CompleteTracks(options.Triangulation());
  std::cout << "  => Completed observations: " << num_completed_observations
            << std::endl;

  // 合并轨迹：将指向同一三维点的不同轨迹合并
  const size_t num_merged_observations =
      mapper->MergeTracks(options.Triangulation());
  std::cout << "  => Merged observations: " << num_merged_observations
            << std::endl;

  return num_completed_observations + num_merged_observations;
}

IncrementalMapper::Options IncrementalMapperOptions::Mapper() const {
  IncrementalMapper::Options options = mapper;
  options.abs_pose_refine_focal_length = ba_refine_focal_length;
  options.abs_pose_refine_extra_params = ba_refine_extra_params;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  options.num_threads = num_threads;
  options.local_ba_num_images = ba_local_num_images;
  options.fix_existing_images = fix_existing_images;
  options.normalize_scene = normalize_scene;
  return options;
}

IncrementalTriangulator::Options IncrementalMapperOptions::Triangulation()
    const {
  IncrementalTriangulator::Options options = triangulation;
  options.min_focal_length_ratio = min_focal_length_ratio;
  options.max_focal_length_ratio = max_focal_length_ratio;
  options.max_extra_param = max_extra_param;
  return options;
}

BundleAdjustmentOptions IncrementalMapperOptions::LocalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = ba_local_function_tolerance;
  options.solver_options.gradient_tolerance = 10.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_local_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.minimizer_progress_to_stdout = false;
  options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = true;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.min_num_residuals_for_multi_threading =
      ba_min_num_residuals_for_multi_threading;
  options.loss_function_scale = 1.0;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  return options;
}

BundleAdjustmentOptions IncrementalMapperOptions::GlobalBundleAdjustment()
    const {
  BundleAdjustmentOptions options;
  options.solver_options.function_tolerance = ba_global_function_tolerance;
  options.solver_options.gradient_tolerance = 1.0;
  options.solver_options.parameter_tolerance = 0.0;
  options.solver_options.max_num_iterations = ba_global_max_num_iterations;
  options.solver_options.max_linear_solver_iterations = 100;
  options.solver_options.minimizer_progress_to_stdout = false;
  options.solver_options.num_threads = num_threads;
#if CERES_VERSION_MAJOR < 2
  options.solver_options.num_linear_solver_threads = num_threads;
#endif  // CERES_VERSION_MAJOR
  options.print_summary = true;
  options.refine_focal_length = ba_refine_focal_length;
  options.refine_principal_point = ba_refine_principal_point;
  options.refine_extra_params = ba_refine_extra_params;
  options.min_num_residuals_for_multi_threading =
      ba_min_num_residuals_for_multi_threading;
  options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::TRIVIAL;
  return options;
}

ParallelBundleAdjuster::Options
IncrementalMapperOptions::ParallelGlobalBundleAdjustment() const {
  ParallelBundleAdjuster::Options options;
  options.max_num_iterations = ba_global_max_num_iterations;
  options.print_summary = true;
  options.gpu_index = ba_global_pba_gpu_index;
  options.num_threads = num_threads;
  options.min_num_residuals_for_multi_threading =
      ba_min_num_residuals_for_multi_threading;
  return options;
}

bool IncrementalMapperOptions::Check() const {
  CHECK_OPTION_GT(min_num_matches, 0);
  CHECK_OPTION_GT(max_num_models, 0);
  CHECK_OPTION_GT(max_model_overlap, 0);
  CHECK_OPTION_GE(min_model_size, 0);
  CHECK_OPTION_GT(init_num_trials, 0);
  CHECK_OPTION_GT(min_focal_length_ratio, 0);
  CHECK_OPTION_GT(max_focal_length_ratio, 0);
  CHECK_OPTION_GE(max_extra_param, 0);
  CHECK_OPTION_GE(ba_local_num_images, 2);
  CHECK_OPTION_GE(ba_local_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_global_images_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_points_ratio, 1.0);
  CHECK_OPTION_GT(ba_global_images_freq, 0);
  CHECK_OPTION_GT(ba_global_points_freq, 0);
  CHECK_OPTION_GT(ba_global_max_num_iterations, 0);
  CHECK_OPTION_GT(ba_local_max_refinements, 0);
  CHECK_OPTION_GE(ba_local_max_refinement_change, 0);
  CHECK_OPTION_GT(ba_global_max_refinements, 0);
  CHECK_OPTION_GE(ba_global_max_refinement_change, 0);
  CHECK_OPTION_GE(snapshot_images_freq, 0);
  CHECK_OPTION_GE(relative_pose_rotation_weight, 0);
  CHECK_OPTION_GE(relative_pose_translation_weight, 0);
  CHECK_OPTION(Mapper().Check());
  CHECK_OPTION(Triangulation().Check());
  return true;
}

IncrementalMapperController::IncrementalMapperController(
    const IncrementalMapperOptions* options, const std::string& image_path,
    const std::string& database_path,
    ReconstructionManager* reconstruction_manager)
    : options_(options),
      image_path_(image_path),
      database_path_(database_path),
      reconstruction_manager_(reconstruction_manager) {
  CHECK(options_->Check());
  RegisterCallback(INITIAL_IMAGE_PAIR_REG_CALLBACK);
  RegisterCallback(NEXT_IMAGE_REG_CALLBACK);
  RegisterCallback(LAST_IMAGE_REG_CALLBACK);
}

void IncrementalMapperController::Run() {
  if (!LoadDatabase()) {
    return;
  }

  IncrementalMapper::Options init_mapper_options = options_->Mapper();
  Reconstruct(init_mapper_options);

  const size_t kNumInitRelaxations = 2;
  for (size_t i = 0; i < kNumInitRelaxations; ++i) {
    if (reconstruction_manager_->Size() > 0 || IsStopped()) {
      break;
    }

    std::cout << "  => Relaxing the initialization constraints." << std::endl;
    init_mapper_options.init_min_num_inliers /= 2;
    Reconstruct(init_mapper_options);

    if (reconstruction_manager_->Size() > 0 || IsStopped()) {
      break;
    }

    std::cout << "  => Relaxing the initialization constraints." << std::endl;
    init_mapper_options.init_min_tri_angle /= 2;
    Reconstruct(init_mapper_options);
  }

  std::cout << std::endl;
  GetTimer().PrintMinutes();
}

bool IncrementalMapperController::LoadDatabase() {
  PrintHeading1("Loading database");

  // Make sure images of the given reconstruction are also included when
  // manually specifying images for the reconstrunstruction procedure.
  std::unordered_set<std::string> image_names = options_->image_names;
  if (reconstruction_manager_->Size() == 1 && !options_->image_names.empty()) {
    const Reconstruction& reconstruction = reconstruction_manager_->Get(0);
    for (const image_t image_id : reconstruction.RegImageIds()) {
      const auto& image = reconstruction.Image(image_id);
      image_names.insert(image.Name());
    }
  }

  Database database(database_path_);
  Timer timer;
  timer.Start();
  const size_t min_num_matches = static_cast<size_t>(options_->min_num_matches);
  database_cache_.Load(database, min_num_matches, options_->ignore_watermarks,
                       image_names);
  std::cout << std::endl;
  timer.PrintMinutes();

  std::cout << std::endl;

  if (database_cache_.NumImages() == 0) {
    std::cout << "WARNING: No images with matches found in the database."
              << std::endl
              << std::endl;
    return false;
  }

  if (options_->relative_pose_from_database) {
    if (!options_->relative_pose_path.empty()) {
      std::cout << "WARNING: Ignoring Mapper.relative_pose_path because "
                   "Mapper.relative_pose_from_database is enabled."
                << std::endl;
    }
    relative_pose_constraints_ = BuildRelativePoseConstraintsFromDatabase(
        database_cache_, options_->relative_pose_rotation_weight,
        options_->relative_pose_translation_weight);
    std::cout << "Loaded " << relative_pose_constraints_.size()
              << " relative pose constraints from database." << std::endl;
  } else if (!options_->relative_pose_path.empty()) {
    relative_pose_constraints_ = ReadRelativePoseConstraints(
        options_->relative_pose_path, database_cache_,
        options_->relative_pose_rotation_weight,
        options_->relative_pose_translation_weight);
    std::cout << "Loaded " << relative_pose_constraints_.size()
              << " relative pose constraints from file." << std::endl;
  }

  return true;
}

/**
 * [功能描述]：执行增量式三维重建的核心函数。
 *            该函数实现了SfM（Structure from Motion）的增量式重建流程：
 *            1. 首先找到并注册初始图像对
 *            2. 然后逐步注册新图像并三角化新的3D点
 *            3. 周期性地进行局部和全局光束法平差（Bundle Adjustment）优化
 *            4. 支持多模型重建和断点续传
 * @param init_mapper_options：增量映射器的配置选项，包含初始化相关的参数设置
 */
void IncrementalMapperController::Reconstruct(
    const IncrementalMapper::Options& init_mapper_options) {
  // 默认丢弃重建结果的标志，用于初始化失败时清理重建
  const bool kDiscardReconstruction = true;

  //////////////////////////////////////////////////////////////////////////////
  // 主循环 - 增量式重建的核心逻辑
  //////////////////////////////////////////////////////////////////////////////

  // 创建增量映射器对象，使用数据库缓存进行初始化
  IncrementalMapper mapper(&database_cache_);
  // 设置相对位姿约束（如果有的话）
  mapper.SetRelativePoseConstraints(relative_pose_constraints_);

  // 检查是否存在用户导入的初始重建模型
  // 如果用户已经导入了一个现有的重建结果，则从该结果继续
  const bool initial_reconstruction_given = reconstruction_manager_->Size() > 0;
  // 断言检查：只能从单个重建继续，不支持从多个重建继续
  CHECK_LE(reconstruction_manager_->Size(), 1) << "Can only resume from a "
                                                  "single reconstruction, but "
                                                  "multiple are given.";

  // 初始化尝试循环：最多尝试 init_num_trials 次不同的初始图像对
  for (int num_trials = 0; num_trials < options_->init_num_trials;
       ++num_trials) {
    // 如果处于暂停状态则阻塞等待
    BlockIfPaused();
    // 如果收到停止信号则退出循环
    if (IsStopped()) {
      break;
    }

    // 确定当前重建的索引
    size_t reconstruction_idx;
    if (!initial_reconstruction_given || num_trials > 0) {
      // 如果没有初始重建或者不是第一次尝试，则创建新的重建
      reconstruction_idx = reconstruction_manager_->Add();
    } else {
      // 使用已存在的初始重建（索引为0）
      reconstruction_idx = 0;
    }

    // 获取当前重建对象的引用
    Reconstruction& reconstruction =
        reconstruction_manager_->Get(reconstruction_idx);

    // 开始重建过程，将重建对象与映射器关联
    mapper.BeginReconstruction(&reconstruction);

    ////////////////////////////////////////////////////////////////////////////
    // 注册初始图像对 - 增量式重建的第一步
    ////////////////////////////////////////////////////////////////////////////

    // 如果当前没有已注册的图像，则需要初始化
    if (reconstruction.NumRegImages() == 0) {
      // 获取用户指定的初始图像对ID（-1表示未指定）
      image_t image_id1 = static_cast<image_t>(options_->init_image_id1);
      image_t image_id2 = static_cast<image_t>(options_->init_image_id2);

      // 如果用户未指定初始图像对，则自动寻找
      if (options_->init_image_id1 == -1 || options_->init_image_id2 == -1) {
        PrintHeading1("Finding good initial image pair");
        // 自动寻找最佳初始图像对
        // 选择标准：足够的特征匹配、合适的基线长度、良好的几何配置
        const bool find_init_success = mapper.FindInitialImagePair(
            init_mapper_options, &image_id1, &image_id2);
        if (!find_init_success) {
          // 找不到合适的初始图像对，清理并退出
          std::cout << "  => No good initial image pair found." << std::endl;
          mapper.EndReconstruction(kDiscardReconstruction);
          reconstruction_manager_->Delete(reconstruction_idx);
          break;
        }
      } else {
        // 用户手动指定了初始图像对，验证其存在性
        if (!reconstruction.ExistsImage(image_id1) ||
            !reconstruction.ExistsImage(image_id2)) {
          std::cout << StringPrintf(
                           "  => Initial image pair #%d and #%d do not exist.",
                           image_id1, image_id2)
                    << std::endl;
          mapper.EndReconstruction(kDiscardReconstruction);
          reconstruction_manager_->Delete(reconstruction_idx);
          return;
        }
      }

      // 使用选定的图像对进行初始化
      PrintHeading1(StringPrintf("Initializing with image pair #%d and #%d",
                                 image_id1, image_id2));
      // 注册初始图像对：估计相对位姿、三角化初始3D点
      const bool reg_init_success = mapper.RegisterInitialImagePair(
          init_mapper_options, image_id1, image_id2);
      if (!reg_init_success) {
        // 初始化失败，给出可能的解决方案
        std::cout << "  => Initialization failed - possible solutions:"
                  << std::endl
                  << "     - try to relax the initialization constraints"
                  << std::endl
                  << "     - manually select an initial image pair"
                  << std::endl;
        mapper.EndReconstruction(kDiscardReconstruction);
        reconstruction_manager_->Delete(reconstruction_idx);
        break;
      }

      // 对初始重建进行优化和过滤
      AdjustGlobalBundle(*options_, &mapper);  // 全局光束法平差
      FilterPoints(*options_, &mapper);         // 过滤离群3D点
      FilterImages(*options_, &mapper);         // 过滤质量差的图像

      // 检查初始化后是否有有效的重建结果
      if (reconstruction.NumRegImages() == 0 ||
          reconstruction.NumPoints3D() == 0) {
        // 初始图像对注册失败（可能被过滤掉了）
        mapper.EndReconstruction(kDiscardReconstruction);
        reconstruction_manager_->Delete(reconstruction_idx);
        // 如果两个初始图像都是手动指定的，则无需继续尝试其他初始对
        if (options_->init_image_id1 != -1 && options_->init_image_id2 != -1) {
          break;
        } else {
          // 否则继续尝试其他初始图像对
          continue;
        }
      }

      // 如果启用了颜色提取，则从图像中提取3D点的颜色
      if (options_->extract_colors) {
        ExtractColors(image_path_, image_id1, &reconstruction);
      }
    }

    // 触发初始图像对注册完成的回调函数（用于UI更新等）
    Callback(INITIAL_IMAGE_PAIR_REG_CALLBACK);

    ////////////////////////////////////////////////////////////////////////////
    // 增量式映射 - 逐步添加新图像并扩展重建
    ////////////////////////////////////////////////////////////////////////////

    // 记录上次快照时的注册图像数量（用于定期保存快照）
    size_t snapshot_prev_num_reg_images = reconstruction.NumRegImages();
    // 记录上次全局BA时的注册图像数量（用于触发全局BA的条件判断）
    size_t ba_prev_num_reg_images = reconstruction.NumRegImages();
    // 记录上次全局BA时的3D点数量
    size_t ba_prev_num_points = reconstruction.NumPoints3D();

    // 注册状态标志
    bool reg_next_success = true;       // 当前注册是否成功
    bool prev_reg_next_success = true;  // 上一次注册是否成功

    // 增量式注册循环：持续注册新图像直到无法继续
    while (reg_next_success) {
      // 检查暂停和停止状态
      BlockIfPaused();
      if (IsStopped()) {
        break;
      }

      // 重置注册成功标志
      reg_next_success = false;

      // 查找下一批候选图像（按可见3D点数量排序）
      const std::vector<image_t> next_images =
          mapper.FindNextImages(options_->Mapper());

      // 如果没有候选图像，则退出循环
      if (next_images.empty()) {
        break;
      }

      // 遍历候选图像，尝试注册
      for (size_t reg_trial = 0; reg_trial < next_images.size(); ++reg_trial) {
        const image_t next_image_id = next_images[reg_trial];
        const Image& next_image = reconstruction.Image(next_image_id);

        // 输出当前正在注册的图像信息
        PrintHeading1(StringPrintf("Registering image #%d (%d)", next_image_id,
                                   reconstruction.NumRegImages() + 1));

        // 显示该图像能看到多少已有3D点
        std::cout << StringPrintf("  => Image sees %d / %d points",
                                  next_image.NumVisiblePoints3D(),
                                  next_image.NumObservations())
                  << std::endl;

        // 尝试注册该图像（通过PnP求解相机位姿）
        reg_next_success =
            mapper.RegisterNextImage(options_->Mapper(), next_image_id);

        if (reg_next_success) {
          // 注册成功后的处理流程
          
          // 三角化该图像观察到的新3D点
          TriangulateImage(*options_, next_image, &mapper);
          // 局部光束法平差优化（优化新注册图像及其邻近图像）
          IterativeLocalRefinement(*options_, next_image_id, &mapper);

          // 判断是否需要进行全局光束法平差
          // 触发条件：注册图像数或3D点数达到一定比例或增量阈值
          if (reconstruction.NumRegImages() >=
                  options_->ba_global_images_ratio * ba_prev_num_reg_images ||
              reconstruction.NumRegImages() >=
                  options_->ba_global_images_freq + ba_prev_num_reg_images ||
              reconstruction.NumPoints3D() >=
                  options_->ba_global_points_ratio * ba_prev_num_points ||
              reconstruction.NumPoints3D() >=
                  options_->ba_global_points_freq + ba_prev_num_points) {
            // 执行迭代式全局优化
            IterativeGlobalRefinement(*options_, &mapper);
            // 更新全局BA的基准值
            ba_prev_num_points = reconstruction.NumPoints3D();
            ba_prev_num_reg_images = reconstruction.NumRegImages();
          }

          // 提取新注册图像的颜色信息
          if (options_->extract_colors) {
            ExtractColors(image_path_, next_image_id, &reconstruction);
          }

          // 检查是否需要保存快照（用于断点续传或中间结果查看）
          if (options_->snapshot_images_freq > 0 &&
              reconstruction.NumRegImages() >=
                  options_->snapshot_images_freq +
                      snapshot_prev_num_reg_images) {
            snapshot_prev_num_reg_images = reconstruction.NumRegImages();
            WriteSnapshot(reconstruction, options_->snapshot_path);
          }

          // 触发图像注册完成的回调函数
          Callback(NEXT_IMAGE_REG_CALLBACK);

          // 成功注册一张图像后跳出内层循环，继续寻找下一张
          break;
        } else {
          // 注册失败，尝试下一个候选图像
          std::cout << "  => Could not register, trying another image."
                    << std::endl;

          // 如果连续失败次数过多且重建规模仍然很小，
          // 则放弃当前初始对，尝试不同的初始图像对
          const size_t kMinNumInitialRegTrials = 30;
          if (reg_trial >= kMinNumInitialRegTrials &&
              reconstruction.NumRegImages() <
                  static_cast<size_t>(options_->min_model_size)) {
            break;
          }
        }
      }

      // 检查模型重叠度：如果与其他模型共享的图像数过多，则停止当前模型
      // 这是为了在多模型重建时避免模型之间过度重叠
      const size_t max_model_overlap =
          static_cast<size_t>(options_->max_model_overlap);
      if (mapper.NumSharedRegImages() >= max_model_overlap) {
        break;
      }

      // 如果当前轮次注册失败但上一轮成功，
      // 则尝试一次全局优化后再次尝试注册
      // 这是一种恢复策略，可能因为累积误差导致暂时无法注册
      if (!reg_next_success && prev_reg_next_success) {
        reg_next_success = true;
        prev_reg_next_success = false;
        IterativeGlobalRefinement(*options_, &mapper);
      } else {
        prev_reg_next_success = reg_next_success;
      }
    }

    // 如果收到停止信号，保存当前重建结果后退出
    if (IsStopped()) {
      const bool kDiscardReconstruction = false;  // 保留重建结果
      mapper.EndReconstruction(kDiscardReconstruction);
      break;
    }

    // 最终全局优化：如果最后一次增量BA不是全局的，则执行一次全局BA
    // 确保最终结果是经过全局优化的
    if (reconstruction.NumRegImages() >= 2 &&
        reconstruction.NumRegImages() != ba_prev_num_reg_images &&
        reconstruction.NumPoints3D() != ba_prev_num_points) {
      IterativeGlobalRefinement(*options_, &mapper);
    }

    // 确定最小模型大小阈值
    // 如果图像总数较少，则降低阈值以支持小型图像集合的重建
    const size_t min_model_size =
        std::min(database_cache_.NumImages(),
                 static_cast<size_t>(options_->min_model_size));
    
    // 判断是否保留当前重建结果
    if ((options_->multiple_models &&
         reconstruction.NumRegImages() < min_model_size) ||
        reconstruction.NumRegImages() == 0) {
      // 重建规模太小或为空，丢弃该重建
      mapper.EndReconstruction(kDiscardReconstruction);
      reconstruction_manager_->Delete(reconstruction_idx);
    } else {
      // 保留有效的重建结果
      const bool kDiscardReconstruction = false;
      mapper.EndReconstruction(kDiscardReconstruction);
    }

    // 触发最后一张图像注册完成的回调函数
    Callback(LAST_IMAGE_REG_CALLBACK);

    // 判断是否继续尝试构建更多模型
    const size_t max_num_models = static_cast<size_t>(options_->max_num_models);
    if (initial_reconstruction_given ||     // 已有初始重建，不再创建新模型
        !options_->multiple_models ||        // 不允许多模型重建
        reconstruction_manager_->Size() >= max_num_models ||  // 已达到最大模型数
        mapper.NumTotalRegImages() >= database_cache_.NumImages() - 1) {  // 几乎所有图像都已注册
      break;
    }
  }
}

}  // namespace colmap
