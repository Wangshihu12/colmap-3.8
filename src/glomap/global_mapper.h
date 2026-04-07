// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#ifndef COLMAP_SRC_GLOMAP_GLOBAL_MAPPER_H_
#define COLMAP_SRC_GLOMAP_GLOBAL_MAPPER_H_

#include <memory>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include "base/database.h"
#include "base/database_cache.h"
#include "base/reconstruction.h"
#include "base/track.h"
#include "glomap/pose_graph.h"
#include "glomap/poselib_relpose.h"
#include "glomap/view_graph_calibration.h"
#include "optim/bundle_adjustment.h"
#include "sfm/incremental_triangulator.h"

namespace colmap {

struct GlomapOptimizationStageOptions {
  // 阶段名称，仅用于日志输出。
  std::string name;

  // 每个阶段内部 BA / 重三角化迭代轮数。
  int max_iterations = 1;

  // 三角化使用的最大重投影误差，单位为像素。
  double tri_max_project_error = 4.0;

  // 过滤三维点时使用的最大重投影误差，单位为像素。
  double filter_max_reproj_error = 4.0;

  // 阶段内是否优化内参。
  bool refine_focal_length = false;
  bool refine_principal_point = false;
  bool refine_extra_params = false;
};

struct GlomapOptions {
  // 顶层执行参数。
  int num_threads = -1;
  int random_seed = -1;
  std::string output_path;

  // 位姿图加载阈值。
  size_t min_num_matches = 15;
  bool ignore_watermarks = true;
  size_t min_num_reg_images = 3;

  // Track 建立阈值。
  double track_intra_image_consistency_threshold = 10.0;
  int track_required_tracks_per_view = std::numeric_limits<int>::max();
  size_t min_track_length = 3;

  // 旋转 / 平移平均与几何过滤阈值。
  int rotation_max_num_iterations = 100;
  int translation_max_num_iterations = 200;
  double max_reproj_error = 4.0;
  double max_normalized_reproj_error = 1e-2;
  double min_tri_angle = 1.5;

  // BA 迭代控制。
  int ba_num_iterations = 3;
  bool ba_skip_fixed_rotation_stage = false;
  bool ba_skip_joint_optimization_stage = false;
  bool use_three_stage_optimization = true;

  // 组件开关。
  bool skip_rotation_averaging = false;
  bool skip_track_establishment = false;
  bool skip_global_positioning = false;
  bool skip_bundle_adjustment = false;
  bool skip_retriangulation = false;

  // 上游新增组件兼容选项。
  bool run_view_graph_calibration = true;
  bool run_poselib_relpose = true;
  PoseLibRelPoseOptions poselib_relpose_options;
  ViewGraphCalibrationOptions view_graph_calibration_options;

  // BA 与重三角化选项。
  BundleAdjustmentOptions bundle_adjustment_options;
  IncrementalTriangulator::Options retriangulation_options;

  // 参考 test_vpgo_pipeline.cc 的三阶段由粗到细优化参数。
  GlomapOptimizationStageOptions stage1_options;
  GlomapOptimizationStageOptions stage2_options;
  GlomapOptimizationStageOptions stage3_options;

  GlomapOptions();
};

class PoseGraph;
class ObservationManager;

class GlomapMapper {
 public:
  /**
   * @brief 功能描述：使用已构建好的数据库缓存初始化全局 mapper。
   * @param database_cache 已加载图像、相机和对应图关系的数据库缓存。
   * @param database 原始数据库对象，用于读取两视图几何和 PoseLib 重估。
   * @return 返回值说明：无返回值。
   */
  explicit GlomapMapper(std::shared_ptr<const DatabaseCache> database_cache,
                        const Database* database = nullptr);

  /**
   * @brief 功能描述：为一次新的全局重建准备内部状态。
   * @param reconstruction 输出重建对象，函数会加载缓存并建立位姿图。
   * @return 返回值说明：无返回值。
   */
  void BeginReconstruction(const std::shared_ptr<Reconstruction>& reconstruction);

  /**
   * @brief 功能描述：运行完整的全局 SfM 管线。
   * @param options 当前求解轮次的全部配置。
   * @return 返回值说明：成功时返回 true，否则返回 false。
   */
  bool Solve(const GlomapOptions& options);

  /**
   * @brief 功能描述：执行旋转平均并完成活动图像集裁剪。
   * @param options 当前求解配置。
   * @return 返回值说明：成功时返回 true，否则返回 false。
   */
  bool RotationAveraging(const GlomapOptions& options);

  /**
   * @brief 功能描述：从有效位姿图边中建立候选 tracks。
   * @param options 当前求解配置。
   * @return 返回值说明：无返回值。
   */
  void EstablishTracks(const GlomapOptions& options);

  /**
   * @brief 功能描述：执行平移平均、初始三角化和几何过滤。
   * @param options 当前求解配置。
   * @return 返回值说明：成功时返回 true，否则返回 false。
   */
  bool GlobalPositioning(const GlomapOptions& options);

  /**
   * @brief 功能描述：执行多轮 BA 精化，并在首轮支持固定旋转阶段。
   * @param options 当前求解配置。
   * @return 返回值说明：成功时返回 true，否则返回 false。
   */
  bool IterativeBundleAdjustment(const GlomapOptions& options);

  /**
   * @brief 功能描述：删除当前结构后重新三角化，并做全局精化。
   * @param options 当前求解配置。
   * @return 返回值说明：成功时返回 true，否则返回 false。
   */
  bool IterativeRetriangulateAndRefine(const GlomapOptions& options);

  /**
   * @brief 功能描述：返回当前内部重建对象。
   * @return 返回值说明：返回当前重建的共享指针。
   */
  std::shared_ptr<class Reconstruction> ReconstructionPtr() const;

 private:
  bool PreparePoseGraph(const GlomapOptions& options);

  const std::shared_ptr<const DatabaseCache> database_cache_;
  const Database* database_ = nullptr;
  std::shared_ptr<class PoseGraph> pose_graph_;
  std::shared_ptr<class Reconstruction> reconstruction_;
  std::vector<Track> candidate_tracks_;
  std::unordered_set<image_t> active_image_ids_;
};

/**
 * @brief 功能描述：兼容旧接口，直接从数据库路径运行完整 glomap 管线。
 * @param database_path 数据库路径。
 * @param options 求解配置。
 * @param reconstruction 输出重建对象。
 * @return 返回值说明：成功时返回 true，否则返回 false。
 */
bool RunGlomapMapper(const std::string& database_path,
                     const GlomapOptions& options,
                     Reconstruction* reconstruction);

/**
 * @brief 功能描述：兼容旧接口，直接从数据库对象运行完整 glomap 管线。
 * @param database 已打开的数据库对象。
 * @param options 求解配置。
 * @param reconstruction 输出重建对象。
 * @return 返回值说明：成功时返回 true，否则返回 false。
 */
bool RunGlomapMapper(const Database& database,
                     const GlomapOptions& options,
                     Reconstruction* reconstruction);

}  // namespace colmap

#endif  // COLMAP_SRC_GLOMAP_GLOBAL_MAPPER_H_
