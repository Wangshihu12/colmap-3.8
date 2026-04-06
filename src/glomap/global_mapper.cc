// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "glomap/global_mapper.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "base/cost_functions.h"
#include "base/pose.h"
#include "base/projection.h"
#include "estimators/triangulation.h"
#include "glomap/observation_manager.h"
#include "sfm/incremental_mapper.h"
#include "util/logging.h"

namespace colmap {
namespace {

// 统一输出每个主阶段结束后的重建统计，方便对比哪一步开始退化。
void LogStageSummary(const std::string& stage_name,
                     const Reconstruction& reconstruction,
                     const PoseGraph* pose_graph,
                     const size_t num_tracks = 0) {
  const size_t num_edges = pose_graph != nullptr ? pose_graph->NumValidEdges() : 0;
  LOG(INFO) << "阶段[" << stage_name << "]完成："
            << "有效边=" << num_edges
            << "，注册图像=" << reconstruction.NumRegImages()
            << "，三维点=" << reconstruction.NumPoints3D()
            << "，候选轨迹=" << num_tracks;
}

// 输出位姿图在不同处理步骤后的边状态统计，用来观察 calibration / PoseLib
// 是否异常放大或缩小了有效边集合。
void LogPoseGraphStats(const std::string& stage_name, const PoseGraph& pose_graph) {
  size_t num_total_edges = pose_graph.Edges().size();
  size_t num_valid_edges = 0;
  size_t num_invalid_edges = 0;
  size_t num_nonpositive_edges = 0;
  for (const ViewEdge& edge : pose_graph.Edges()) {
    if (!edge.valid) {
      ++num_invalid_edges;
    }
    if (edge.weight <= 0.0) {
      ++num_nonpositive_edges;
    }
    if (edge.valid && edge.weight > 0.0) {
      ++num_valid_edges;
    }
  }

  LOG(INFO) << "位姿图[" << stage_name << "]统计：总边数=" << num_total_edges
            << "，有效边数=" << num_valid_edges
            << "，无效边数=" << num_invalid_edges
            << "，非正权重边数=" << num_nonpositive_edges;
}

// 根据三阶段优化配置生成一个“阶段内专用”的 mapper 选项副本。
// 这样主流程可以共享一套逻辑，而每个阶段只覆写：
// 1. 三角化阈值
// 2. 几何过滤阈值
// 3. 内参是否参与 BA
GlomapOptions BuildStageScopedOptions(
    const GlomapOptions& base_options,
    const GlomapOptimizationStageOptions& stage_options) {
  GlomapOptions stage_scoped_options = base_options;
  stage_scoped_options.ba_num_iterations = std::max(0, stage_options.max_iterations);
  stage_scoped_options.max_reproj_error = stage_options.filter_max_reproj_error;
  stage_scoped_options.min_tri_angle = 2.0;
  stage_scoped_options.retriangulation_options.min_angle = 2.0;
  stage_scoped_options.retriangulation_options.complete_max_reproj_error =
      stage_options.tri_max_project_error;
  stage_scoped_options.retriangulation_options.merge_max_reproj_error =
      stage_options.tri_max_project_error;
  stage_scoped_options.bundle_adjustment_options.refine_focal_length =
      stage_options.refine_focal_length;
  stage_scoped_options.bundle_adjustment_options.refine_principal_point =
      stage_options.refine_principal_point;
  stage_scoped_options.bundle_adjustment_options.refine_extra_params =
      stage_options.refine_extra_params;
  stage_scoped_options.bundle_adjustment_options.refine_extrinsics = true;
  stage_scoped_options.bundle_adjustment_options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::HUBER;
  stage_scoped_options.bundle_adjustment_options.loss_function_scale =
      stage_options.filter_max_reproj_error;
  stage_scoped_options.bundle_adjustment_options.solver_options.function_tolerance =
      1e-4;
  stage_scoped_options.bundle_adjustment_options.solver_options.gradient_tolerance =
      1.0;
  return stage_scoped_options;
}

struct ObservationKey {
  image_t image_id = kInvalidImageId;
  point2D_t point2D_idx = kInvalidPoint2DIdx;

  bool operator==(const ObservationKey& other) const {
    return image_id == other.image_id && point2D_idx == other.point2D_idx;
  }

  bool operator<(const ObservationKey& other) const {
    if (image_id != other.image_id) {
      return image_id < other.image_id;
    }
    return point2D_idx < other.point2D_idx;
  }
};

// 观测并查集用于把多张图上的匹配点合并成 track。
// 一个 ObservationKey 唯一标识一个 2D 观测：(image_id, point2D_idx)。
struct ObservationKeyHash {
  size_t operator()(const ObservationKey& key) const {
    return std::hash<image_t>()(key.image_id) ^
           (std::hash<point2D_t>()(key.point2D_idx) << 1);
  }
};

class ObservationUnionFind {
 public:
  // 将两个观测并到同一个 track 连通块中。
  void Union(const ObservationKey& a, const ObservationKey& b) {
    const ObservationKey root_a = Find(a);
    const ObservationKey root_b = Find(b);
    if (root_a == root_b) {
      return;
    }
    if (root_b < root_a) {
      parents_[root_a] = root_b;
    } else {
      parents_[root_b] = root_a;
    }
  }

  // 查找当前观测所在连通块的根节点，并带路径压缩。
  ObservationKey Find(const ObservationKey& key) {
    auto it = parents_.find(key);
    if (it == parents_.end()) {
      parents_.emplace(key, key);
      return key;
    }
    if (it->second == key) {
      return key;
    }
    it->second = Find(it->second);
    return it->second;
  }

  // 返回所有观测的根节点映射，供后续按根分组建立 tracks。
  std::unordered_map<ObservationKey, ObservationKey, ObservationKeyHash>
  Parents() {
    for (auto& item : parents_) {
      item.second = Find(item.first);
    }
    return parents_;
  }

 private:
  std::unordered_map<ObservationKey, ObservationKey, ObservationKeyHash>
      parents_;
};

struct RotationErrorCost {
  explicit RotationErrorCost(const Eigen::Vector4d& qvec_21)
      : qvec_21_(NormalizeQuaternion(qvec_21)) {}

  template <typename T>
  bool operator()(const T* const qvec1,
                  const T* const qvec2,
                  T* residuals) const {
    // residuals 形状为 [3]，为相对旋转误差的 angle-axis 表达。
    const T qvec1_inv[4] = {qvec1[0], -qvec1[1], -qvec1[2], -qvec1[3]};
    T predicted_qvec21[4];
    ceres::QuaternionProduct(qvec2, qvec1_inv, predicted_qvec21);

    const T qvec21_inv[4] = {T(qvec_21_(0)),
                             T(-qvec_21_(1)),
                             T(-qvec_21_(2)),
                             T(-qvec_21_(3))};
    T error_qvec[4];
    ceres::QuaternionProduct(predicted_qvec21, qvec21_inv, error_qvec);
    ceres::QuaternionToAngleAxis(error_qvec, residuals);
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec_21) {
    return new ceres::AutoDiffCostFunction<RotationErrorCost, 3, 4, 4>(
        new RotationErrorCost(qvec_21));
  }

  Eigen::Vector4d qvec_21_;
};

struct TranslationAveragingCost {
  explicit TranslationAveragingCost(const Eigen::Vector3d& direction)
      : direction_(direction) {}

  template <typename T>
  bool operator()(const T* const center1,
                  const T* const center2,
                  const T* const scale,
                  T* residuals) const {
    // residuals 形状为 [3]，约束两相机中心之差与相对平移方向共线。
    residuals[0] =
        (center1[0] - center2[0]) - scale[0] * T(direction_(0));
    residuals[1] =
        (center1[1] - center2[1]) - scale[0] * T(direction_(1));
    residuals[2] =
        (center1[2] - center2[2]) - scale[0] * T(direction_(2));
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& direction) {
    return new ceres::AutoDiffCostFunction<TranslationAveragingCost, 3, 3, 3, 1>(
        new TranslationAveragingCost(direction));
  }

  Eigen::Vector3d direction_;
};

struct MstResult {
  image_t root_image_id = kInvalidImageId;
  image_t anchor_image_id = kInvalidImageId;
  std::unordered_map<image_t, image_t> parent_image_ids;
  std::unordered_map<image_t, size_t> parent_edge_indices;
};

bool IsZeroPose(const TwoViewGeometry& geometry) {
  return geometry.qvec.squaredNorm() < 1e-12 ||
         geometry.tvec.squaredNorm() < 1e-12;
}

// 将图像中的 2D 点提取为 [N, 2] 坐标数组，供两视图位姿估计使用。
std::vector<Eigen::Vector2d> ExtractPoints2D(const Image& image) {
  std::vector<Eigen::Vector2d> points;
  points.reserve(image.NumPoints2D());
  for (const Point2D& point2D : image.Points2D()) {
    points.push_back(point2D.XY());
  }
  return points;
}

// 某些数据库条目只有基础矩阵/本质矩阵，没有直接分解出的相对位姿。
// 这里在需要时即时分解，保证后续位姿图步骤有 qvec/tvec 可用。
bool EnsureRelativePose(const Camera& camera1,
                        const Image& image1,
                        const Camera& camera2,
                        const Image& image2,
                        TwoViewGeometry* geometry) {
  if (!IsZeroPose(*geometry)) {
    return true;
  }

  const std::vector<Eigen::Vector2d> points1 = ExtractPoints2D(image1);
  const std::vector<Eigen::Vector2d> points2 = ExtractPoints2D(image2);
  return geometry->EstimateRelativePose(camera1, points1, camera2, points2);
}

std::unordered_map<image_t, std::vector<size_t>> BuildAdjacency(
    const std::vector<ViewEdge>& edges) {
  // 邻接表存的是边索引，而不是邻接图像 ID。
  // 这样后续需要访问权重或几何时可以直接回到原始边数组。
  std::unordered_map<image_t, std::vector<size_t>> adjacency;
  for (size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
    adjacency[edges[edge_idx].image_id1].push_back(edge_idx);
    adjacency[edges[edge_idx].image_id2].push_back(edge_idx);
  }
  return adjacency;
}

MstResult ComputeMaximumSpanningTree(const std::vector<ViewEdge>& edges) {
  MstResult result;
  if (edges.empty()) {
    return result;
  }

  const auto adjacency = BuildAdjacency(edges);
  // 根节点选择加权度最大的图像，能让初始化更稳定。
  double best_score = -1.0;
  for (const auto& item : adjacency) {
    double score = 0.0;
    for (const size_t edge_idx : item.second) {
      score += edges[edge_idx].weight;
    }
    if (score > best_score) {
      best_score = score;
      result.root_image_id = item.first;
    }
  }

  if (result.root_image_id == kInvalidImageId) {
    return result;
  }

  struct QueueItem {
    double weight = 0.0;
    image_t from = kInvalidImageId;
    image_t to = kInvalidImageId;
    size_t edge_idx = std::numeric_limits<size_t>::max();

    bool operator<(const QueueItem& other) const { return weight < other.weight; }
  };

  std::priority_queue<QueueItem> queue;
  std::unordered_set<image_t> visited;
  // Prim 风格最大生成树，用于给全局旋转/平移提供一个可传播的初值。
  visited.insert(result.root_image_id);
  for (const size_t edge_idx : adjacency.at(result.root_image_id)) {
    const ViewEdge& edge = edges[edge_idx];
    const image_t next_image =
        edge.image_id1 == result.root_image_id ? edge.image_id2 : edge.image_id1;
    queue.push({edge.weight, result.root_image_id, next_image, edge_idx});
  }

  while (!queue.empty()) {
    const QueueItem item = queue.top();
    queue.pop();
    if (visited.count(item.to) > 0) {
      continue;
    }

    visited.insert(item.to);
    result.parent_image_ids[item.to] = item.from;
    result.parent_edge_indices[item.to] = item.edge_idx;
    if (result.anchor_image_id == kInvalidImageId) {
      result.anchor_image_id = item.to;
    }

    for (const size_t edge_idx : adjacency.at(item.to)) {
      const ViewEdge& edge = edges[edge_idx];
      const image_t next_image =
          edge.image_id1 == item.to ? edge.image_id2 : edge.image_id1;
      if (visited.count(next_image) == 0) {
        queue.push({edge.weight, item.to, next_image, edge_idx});
      }
    }
  }

  return result;
}

Eigen::Vector4d PropagateChildRotation(const ViewEdge& edge,
                                       const image_t parent_image_id,
                                       const Eigen::Vector4d& qvec_parent) {
  // 根据边方向决定是直接乘相对旋转，还是先取逆再传播到子节点。
  const Eigen::Vector4d qvec_rel = NormalizeQuaternion(edge.geometry.qvec);
  if (edge.image_id1 == parent_image_id) {
    return NormalizeQuaternion(ConcatenateQuaternions(qvec_rel, qvec_parent));
  }

  Eigen::Vector4d qvec_rel_inv;
  Eigen::Vector3d tvec_dummy;
  InvertPose(qvec_rel, edge.geometry.tvec, &qvec_rel_inv, &tvec_dummy);
  return NormalizeQuaternion(ConcatenateQuaternions(qvec_rel_inv, qvec_parent));
}

bool SolveRotations(const std::vector<ViewEdge>& edges,
                    const MstResult& mst,
                    Reconstruction* reconstruction,
                    const GlomapOptions& options) {
  if (mst.root_image_id == kInvalidImageId) {
    return false;
  }

  reconstruction->Image(mst.root_image_id).SetQvec(Eigen::Vector4d(1, 0, 0, 0));

  // 先用 MST 传播一个全局一致的初始旋转，再交给 Ceres 做全局旋转平均。
  std::unordered_map<image_t, std::vector<image_t>> children;
  for (const auto& item : mst.parent_image_ids) {
    children[item.second].push_back(item.first);
  }

  std::queue<image_t> queue;
  queue.push(mst.root_image_id);
  while (!queue.empty()) {
    const image_t parent_image_id = queue.front();
    queue.pop();
    for (const image_t child_image_id : children[parent_image_id]) {
      const size_t edge_idx = mst.parent_edge_indices.at(child_image_id);
      const ViewEdge& edge = edges[edge_idx];
      const Eigen::Vector4d qvec_child = PropagateChildRotation(
          edge, parent_image_id, reconstruction->Image(parent_image_id).Qvec());
      reconstruction->Image(child_image_id).SetQvec(qvec_child);
      queue.push(child_image_id);
    }
  }

  ceres::Problem problem;
  for (const ViewEdge& edge : edges) {
    // 每条边都对两端图像的全局旋转施加一个相对旋转约束。
    ceres::CostFunction* cost_function =
        RotationErrorCost::Create(edge.geometry.qvec);
    problem.AddResidualBlock(cost_function,
                             new ceres::HuberLoss(0.1),
                             reconstruction->Image(edge.image_id1).Qvec().data(),
                             reconstruction->Image(edge.image_id2).Qvec().data());
  }

  for (const auto& item : reconstruction->Images()) {
    if (!item.second.IsRegistered()) {
      continue;
    }
    SetQuaternionManifold(&problem, reconstruction->Image(item.first).Qvec().data());
  }
  problem.SetParameterBlockConstant(
      reconstruction->Image(mst.root_image_id).Qvec().data());

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = options.rotation_max_num_iterations;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.minimizer_progress_to_stdout = false;
  solver_options.num_threads = options.num_threads > 0 ? options.num_threads : 1;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  return summary.IsSolutionUsable();
}

Eigen::Vector3d RelativeDirectionInWorld(const ViewEdge& edge,
                                         const Reconstruction& reconstruction) {
  // 将相对平移方向从 image2 相机坐标系转到世界坐标系，
  // 以便后续在相机中心空间里做平移平均。
  const Image& image2 = reconstruction.Image(edge.image_id2);
  const Eigen::Matrix3d rotation2 = image2.RotationMatrix();
  return (rotation2.transpose() * edge.geometry.tvec).normalized();
}

bool InitializeCentersFromTree(const std::vector<ViewEdge>& edges,
                               const MstResult& mst,
                               Reconstruction* reconstruction) {
  if (mst.root_image_id == kInvalidImageId) {
    return false;
  }

  std::unordered_map<image_t, Eigen::Vector3d> centers;
  centers[mst.root_image_id] = Eigen::Vector3d::Zero();
  // 沿 MST 逐层传播相机中心。这里先只恢复方向一致的中心布局，
  // 绝对尺度由后面的优化变量 scale 吸收。

  std::unordered_map<image_t, std::vector<image_t>> children;
  for (const auto& item : mst.parent_image_ids) {
    children[item.second].push_back(item.first);
  }

  std::queue<image_t> queue;
  queue.push(mst.root_image_id);
  while (!queue.empty()) {
    const image_t parent_image_id = queue.front();
    queue.pop();
    for (const image_t child_image_id : children[parent_image_id]) {
      const ViewEdge& edge = edges[mst.parent_edge_indices.at(child_image_id)];
      const Eigen::Vector3d direction = RelativeDirectionInWorld(edge, *reconstruction);
      if (edge.image_id1 == parent_image_id) {
        centers[child_image_id] = centers[parent_image_id] - direction;
      } else {
        centers[child_image_id] = centers[parent_image_id] + direction;
      }
      queue.push(child_image_id);
    }
  }

  for (const auto& item : centers) {
    const Eigen::Matrix3d rotation =
        reconstruction->Image(item.first).RotationMatrix();
    reconstruction->Image(item.first).SetTvec(-rotation * item.second);
  }

  return true;
}

bool SolveTranslations(const std::vector<ViewEdge>& edges,
                       const MstResult& mst,
                       Reconstruction* reconstruction,
                       const GlomapOptions& options) {
  if (!InitializeCentersFromTree(edges, mst, reconstruction)) {
    return false;
  }

  std::unordered_map<image_t, std::array<double, 3>> centers;
  for (const auto& item : reconstruction->Images()) {
    if (!item.second.IsRegistered()) {
      continue;
    }
    const Eigen::Vector3d center = item.second.ProjectionCenter();
    centers[item.first] = {center(0), center(1), center(2)};
  }

  std::vector<double> scales(edges.size(), 1.0);
  ceres::Problem problem;
  for (size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx) {
    const ViewEdge& edge = edges[edge_idx];
    const Eigen::Vector3d direction = RelativeDirectionInWorld(edge, *reconstruction);
    const Eigen::Vector3d center1 =
        reconstruction->Image(edge.image_id1).ProjectionCenter();
    const Eigen::Vector3d center2 =
        reconstruction->Image(edge.image_id2).ProjectionCenter();
    scales[edge_idx] = std::max(1e-3, (center1 - center2).dot(direction));
    // 每条边单独带一个尺度变量，这与经典全局平移平均里“已知方向、未知长度”的模型一致。

    problem.AddResidualBlock(TranslationAveragingCost::Create(direction),
                             new ceres::HuberLoss(0.1),
                             centers[edge.image_id1].data(),
                             centers[edge.image_id2].data(),
                             &scales[edge_idx]);
  }

  problem.SetParameterBlockConstant(centers[mst.root_image_id].data());
  if (mst.anchor_image_id != kInvalidImageId) {
    problem.SetParameterBlockConstant(centers[mst.anchor_image_id].data());
  }

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = options.translation_max_num_iterations;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.minimizer_progress_to_stdout = false;
  solver_options.num_threads = options.num_threads > 0 ? options.num_threads : 1;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  if (!summary.IsSolutionUsable()) {
    return false;
  }

  for (const auto& item : centers) {
    const Eigen::Vector3d center(item.second[0], item.second[1], item.second[2]);
    const Eigen::Matrix3d rotation =
        reconstruction->Image(item.first).RotationMatrix();
    reconstruction->Image(item.first).SetTvec(-rotation * center);
  }

  return true;
}

std::vector<Track> BuildTracksFromEdges(const std::vector<ViewEdge>& edges,
                                        const Reconstruction& reconstruction,
                                        const GlomapOptions& options) {
  // 第一步：把所有两视图内点匹配通过并查集合并成多视图连通块。
  ObservationUnionFind union_find;
  for (const ViewEdge& edge : edges) {
    for (const auto& match : edge.geometry.inlier_matches) {
      const ObservationKey obs1{edge.image_id1, match.point2D_idx1};
      const ObservationKey obs2{edge.image_id2, match.point2D_idx2};
      union_find.Union(obs1, obs2);
    }
  }

  std::unordered_map<ObservationKey,
                     std::vector<ObservationKey>,
                     ObservationKeyHash>
      grouped_tracks;
  for (const auto& item : union_find.Parents()) {
    grouped_tracks[item.second].push_back(item.first);
  }

  std::vector<Track> tracks;
  tracks.reserve(grouped_tracks.size());
  for (auto& item : grouped_tracks) {
    // 第二步：检查单图内几何一致性，避免一个 track 在同一张图里出现明显冲突的重复观测。
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_observations;
    Track track;
    bool valid = true;
    for (const ObservationKey& obs : item.second) {
      const Eigen::Vector2d xy =
          reconstruction.Image(obs.image_id).Point2D(obs.point2D_idx).XY();
      auto& observations = image_observations[obs.image_id];
      for (const Eigen::Vector2d& current_xy : observations) {
        if ((current_xy - xy).squaredNorm() >
            options.track_intra_image_consistency_threshold *
                options.track_intra_image_consistency_threshold) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        break;
      }
      observations.push_back(xy);
      track.AddElement(TrackElement(obs.image_id, obs.point2D_idx));
    }
    if (!valid || track.Length() < options.min_track_length) {
      continue;
    }
    tracks.push_back(std::move(track));
  }
  return tracks;
}

size_t TriangulateTracks(const GlomapOptions& options,
                         Reconstruction* reconstruction,
                         const std::vector<Track>& tracks) {
  // 使用多视图 RANSAC 三角化从候选 track 恢复初始三维点。
  EstimateTriangulationOptions tri_options;
  tri_options.min_tri_angle = DegToRad(options.min_tri_angle);
  tri_options.residual_type =
      TriangulationEstimator::ResidualType::REPROJECTION_ERROR;
  tri_options.ransac_options.max_error = options.max_reproj_error;
  tri_options.ransac_options.confidence = 0.999;
  tri_options.ransac_options.min_inlier_ratio = 0.25;
  tri_options.ransac_options.max_num_trials = 1000;

  size_t num_points3D = 0;
  for (const Track& track : tracks) {
    std::vector<TriangulationEstimator::PointData> point_data;
    std::vector<TriangulationEstimator::PoseData> pose_data;
    std::vector<TrackElement> track_elements;
    point_data.reserve(track.Length());
    pose_data.reserve(track.Length());
    track_elements.reserve(track.Length());

    for (const TrackElement& element : track.Elements()) {
      // point_data / pose_data 按同一个顺序堆叠，后续 RANSAC 结果通过 inlier_mask 对齐回 track。
      const Image& image = reconstruction->Image(element.image_id);
      if (!image.IsRegistered()) {
        continue;
      }
      const Camera& camera = reconstruction->Camera(image.CameraId());
      const Eigen::Vector2d point2D = image.Point2D(element.point2D_idx).XY();

      TriangulationEstimator::PointData point;
      point.point = point2D;
      point.point_normalized = camera.ImageToWorld(point2D);
      point_data.push_back(point);
      pose_data.emplace_back(
          image.ProjectionMatrix(), image.ProjectionCenter(), &camera);
      track_elements.push_back(element);
    }

    if (point_data.size() < options.min_track_length) {
      continue;
    }

    std::vector<char> inlier_mask;
    Eigen::Vector3d xyz;
    if (!EstimateTriangulation(
            tri_options, point_data, pose_data, &inlier_mask, &xyz)) {
      continue;
    }

    Track inlier_track;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
      if (inlier_mask[i]) {
        inlier_track.AddElement(track_elements[i]);
      }
    }
    if (inlier_track.Length() < options.min_track_length) {
      continue;
    }

    reconstruction->AddPoint3D(xyz, std::move(inlier_track));
    ++num_points3D;
  }

  LOG(INFO) << "Triangulated " << num_points3D << " tracks";
  return num_points3D;
}

bool RunBundleAdjustmentOnce(const GlomapOptions& options,
                             const MstResult& mst,
                             const bool fixed_rotation_stage,
                             Reconstruction* reconstruction) {
  if (reconstruction->NumPoints3D() == 0) {
    return false;
  }

  BundleAdjustmentConfig config;
  // 所有已注册图像和当前三维点都加入同一个全局 BA 问题。
  for (const image_t image_id : reconstruction->RegImageIds()) {
    config.AddImage(image_id);
  }
  for (const auto& point3D : reconstruction->Points3D()) {
    config.AddVariablePoint(point3D.first);
  }

  config.SetConstantPose(mst.root_image_id);
  // root 固定完整位姿，anchor 只固定平移，用来消除规约自由度。
  if (mst.anchor_image_id != kInvalidImageId &&
      mst.anchor_image_id != mst.root_image_id) {
    config.SetConstantTvec(mst.anchor_image_id, {0, 1, 2});
    if (fixed_rotation_stage) {
      config.SetConstantQvec(mst.anchor_image_id);
    }
  }
  if (fixed_rotation_stage) {
    // 第一阶段仅优化平移和结构，旋转保持不动。
    for (const image_t image_id : reconstruction->RegImageIds()) {
      if (image_id == mst.root_image_id) {
        continue;
      }
      config.SetConstantQvec(image_id);
    }
  }

  BundleAdjustmentOptions ba_options = options.bundle_adjustment_options;
  ba_options.loss_function_type = BundleAdjustmentOptions::LossFunctionType::HUBER;
  ba_options.loss_function_scale = 1.0;
  ba_options.refine_principal_point = false;
  ba_options.print_summary = false;
  ba_options.solver_options.max_num_iterations =
      fixed_rotation_stage ? 50 : ba_options.solver_options.max_num_iterations;
  ba_options.solver_options.num_threads = options.num_threads;

  BundleAdjuster adjuster(ba_options, config);
  return adjuster.Solve(reconstruction);
}

IncrementalMapper::Options MakeIncrementalMapperOptions(
    const GlomapOptions& options) {
  // 这里复用旧版 IncrementalMapper 的过滤/重三角化实现，因此需要把当前 glomap
  // 阈值映射到它的配置结构里。
  IncrementalMapper::Options mapper_options;
  mapper_options.filter_max_reproj_error = options.max_reproj_error;
  mapper_options.filter_min_tri_angle = options.min_tri_angle;
  mapper_options.num_threads = options.num_threads;
  mapper_options.max_reg_trials = 1;
  mapper_options.init_max_reg_trials = 1;
  mapper_options.fix_existing_images = false;
  mapper_options.normalize_scene = true;
  return mapper_options;
}

}  // namespace

GlomapOptions::GlomapOptions() {
  // 默认 BA 选项尽量保持稳定保守，再由三阶段配置按需覆盖。
  bundle_adjustment_options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::HUBER;
  bundle_adjustment_options.loss_function_scale = 1.0;
  bundle_adjustment_options.refine_principal_point = false;
  bundle_adjustment_options.print_summary = false;
  bundle_adjustment_options.solver_options.max_num_iterations = 100;
  retriangulation_options.complete_max_reproj_error = 15.0;
  retriangulation_options.merge_max_reproj_error = 15.0;
  retriangulation_options.min_angle = 1.0;

  stage1_options.name = "阶段1：固定内参，粗优化外参";
  stage1_options.max_iterations = 2;
  stage1_options.tri_max_project_error = 16.0;
  stage1_options.filter_max_reproj_error = 16.0;
  stage1_options.refine_focal_length = false;
  stage1_options.refine_principal_point = false;
  stage1_options.refine_extra_params = false;

  stage2_options.name = "阶段2：固定内参，细化外参";
  stage2_options.max_iterations = 2;
  stage2_options.tri_max_project_error = 4.0;
  stage2_options.filter_max_reproj_error = 4.0;
  stage2_options.refine_focal_length = false;
  stage2_options.refine_principal_point = false;
  stage2_options.refine_extra_params = false;

  stage3_options.name = "阶段3：优化内参与外参";
  stage3_options.max_iterations = 3;
  stage3_options.tri_max_project_error = 4.0;
  stage3_options.filter_max_reproj_error = 4.0;
  stage3_options.refine_focal_length = true;
  stage3_options.refine_principal_point = true;
  stage3_options.refine_extra_params = true;
}

GlomapMapper::GlomapMapper(std::shared_ptr<const DatabaseCache> database_cache,
                           const Database* database)
    : database_cache_(std::move(database_cache)), database_(database) {
  CHECK_NOTNULL(database_cache_.get());
}

void GlomapMapper::BeginReconstruction(
    const std::shared_ptr<Reconstruction>& reconstruction) {
  CHECK_NOTNULL(reconstruction.get());
  // BeginReconstruction 会把缓存中的图像、相机、对应关系全部载入到一个新的
  // Reconstruction 中，因此每次重试都应从新的对象开始，而不是复用旧状态。
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  reconstruction_->SetUp(&database_cache_->CorrespondenceGraph());
  pose_graph_ = std::make_shared<PoseGraph>();
  candidate_tracks_.clear();
  active_image_ids_.clear();
}

bool GlomapMapper::PreparePoseGraph(const GlomapOptions& options) {
  CHECK_NOTNULL(reconstruction_.get());
  CHECK_NOTNULL(pose_graph_.get());
  CHECK_NOTNULL(database_);

  if (!pose_graph_->Load(
          *database_, *reconstruction_, options.min_num_matches, options.ignore_watermarks)) {
    LOG(ERROR) << "No valid view-graph edges found";
    return false;
  }

  if (options.run_view_graph_calibration) {
    // 先做焦距校准和异常边过滤，让后面的 PoseLib 重估建立在更干净的边集合上。
    if (!CalibrateViewGraph(options.view_graph_calibration_options,
                            &pose_graph_->MutableEdges(),
                            reconstruction_.get())) {
      LOG(WARNING) << "View-graph calibration failed, continue with current graph";
    }
    LogPoseGraphStats("焦距校准后", *pose_graph_);
  }

  if (options.run_poselib_relpose) {
    // 再用 PoseLib 对剩余有效边重估相对位姿，提升边的几何质量。
    const size_t num_refined = EstimateRelativePosesWithPoseLib(
        options.poselib_relpose_options,
        &pose_graph_->MutableEdges(),
        *reconstruction_);
    LOG(INFO) << "PoseLib 相对位姿重估完成：成功重估 " << num_refined
              << " 条边";
    LogPoseGraphStats("PoseLib 重估后", *pose_graph_);
  }

  for (ViewEdge& edge : pose_graph_->MutableEdges()) {
    if (!edge.valid || edge.weight <= 0.0) {
      edge.valid = false;
      continue;
    }
    const Camera& camera1 = reconstruction_->Camera(
        reconstruction_->Image(edge.image_id1).CameraId());
    const Camera& camera2 = reconstruction_->Camera(
        reconstruction_->Image(edge.image_id2).CameraId());
    if (!EnsureRelativePose(camera1,
                            reconstruction_->Image(edge.image_id1),
                            camera2,
                            reconstruction_->Image(edge.image_id2),
                            &edge.geometry)) {
      edge.valid = false;
    }
  }

  LogPoseGraphStats("相对位姿检查后", *pose_graph_);
  return !pose_graph_->Empty();
}

bool GlomapMapper::RotationAveraging(const GlomapOptions& options) {
  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  active_image_ids_ = pose_graph_->ComputeLargestConnectedImageComponent();
  // 全局 SfM 只在主连通分量上求解，离散子图会被直接剔除。
  pose_graph_->InvalidatePairsOutsideActiveImageIds(active_image_ids_);
  if (active_image_ids_.size() < options.min_num_reg_images) {
    LOG(ERROR) << "Connected component too small for global mapping";
    return false;
  }

  for (const auto& item : reconstruction_->Images()) {
    reconstruction_->Image(item.first).SetRegistered(false);
  }
  for (const image_t image_id : active_image_ids_) {
    reconstruction_->RegisterImage(image_id);
  }

  const std::vector<ViewEdge> valid_edges = pose_graph_->ValidEdges();
  const MstResult mst = ComputeMaximumSpanningTree(valid_edges);
  if (mst.root_image_id == kInvalidImageId) {
    LOG(ERROR) << "Failed to build maximum spanning tree";
    return false;
  }

  if (!SolveRotations(valid_edges, mst, reconstruction_.get(), options)) {
    LOG(ERROR) << "Rotation averaging failed";
    return false;
  }
  LogStageSummary("旋转平均", *reconstruction_, pose_graph_.get());
  return true;
}

void GlomapMapper::EstablishTracks(const GlomapOptions& options) {
  candidate_tracks_ =
      BuildTracksFromEdges(pose_graph_->ValidEdges(), *reconstruction_, options);
  LogStageSummary("轨迹建立",
                  *reconstruction_,
                  pose_graph_.get(),
                  candidate_tracks_.size());
}

bool GlomapMapper::GlobalPositioning(const GlomapOptions& options) {
  const std::vector<ViewEdge> valid_edges = pose_graph_->ValidEdges();
  const MstResult mst = ComputeMaximumSpanningTree(valid_edges);
  if (!SolveTranslations(valid_edges, mst, reconstruction_.get(), options)) {
    LOG(ERROR) << "Translation averaging failed";
    return false;
  }

  if (candidate_tracks_.empty()) {
    LOG(ERROR) << "Track establishment produced no usable tracks";
    return false;
  }

  const size_t num_triangulated_tracks =
      TriangulateTracks(options, reconstruction_.get(), candidate_tracks_);
  LOG(INFO) << "初始三角化统计：候选轨迹=" << candidate_tracks_.size()
            << "，成功三角化轨迹=" << num_triangulated_tracks
            << "，成功率="
            << (candidate_tracks_.empty()
                    ? 0.0
                    : 100.0 * static_cast<double>(num_triangulated_tracks) /
                          static_cast<double>(candidate_tracks_.size()))
            << "%";
  if (num_triangulated_tracks == 0) {
    LOG(ERROR) << "Track triangulation failed";
    return false;
  }

  ObservationManager obs_manager(*reconstruction_);
  // 初始结构通常噪声较大，因此在第一次三角化后立刻做一轮几何清洗。
  const size_t num_filtered_points_by_geom =
      obs_manager.FilterAllPoints3D(options.max_reproj_error, options.min_tri_angle);
  const size_t num_filtered_points_by_track =
      obs_manager.FilterPoints3DWithShortTracks(options.min_track_length);
  const size_t num_filtered_points_by_depth =
      obs_manager.FilterObservationsWithNegativeDepth();
  LOG(INFO) << "初始三角化后过滤统计：几何过滤观测="
            << num_filtered_points_by_geom
            << "，短轨迹过滤观测=" << num_filtered_points_by_track
            << "，负深度过滤观测=" << num_filtered_points_by_depth;
  reconstruction_->Normalize();
  LogStageSummary("全局定位与初始三角化",
                  *reconstruction_,
                  pose_graph_.get(),
                  candidate_tracks_.size());
  return reconstruction_->NumPoints3D() > 0;
}

bool GlomapMapper::IterativeBundleAdjustment(const GlomapOptions& options) {
  const std::vector<ViewEdge> valid_edges = pose_graph_->ValidEdges();
  const MstResult mst = ComputeMaximumSpanningTree(valid_edges);

  for (int iter = 0; iter < options.ba_num_iterations; ++iter) {
    const size_t num_points_before_ba = reconstruction_->NumPoints3D();
    // 每一轮都先 BA，再过滤，再归一化；这是一个“优化-清洗”的交替过程。
    if (!options.ba_skip_fixed_rotation_stage) {
      if (!RunBundleAdjustmentOnce(
              options, mst, true, reconstruction_.get())) {
        LOG(ERROR) << "Fixed-rotation bundle adjustment failed";
        return false;
      }
    }

    if (!options.ba_skip_joint_optimization_stage) {
      if (!RunBundleAdjustmentOnce(
              options, mst, false, reconstruction_.get())) {
        LOG(ERROR) << "Joint bundle adjustment failed";
        return false;
      }
    }

    ObservationManager obs_manager(*reconstruction_);
    const size_t num_filtered_points_by_geom =
        obs_manager.FilterAllPoints3D(options.max_reproj_error, options.min_tri_angle);
    const size_t num_filtered_points_by_track =
        obs_manager.FilterPoints3DWithShortTracks(options.min_track_length);
    reconstruction_->Normalize();
    LOG(INFO) << "第 " << (iter + 1)
              << " 轮全局 BA 完成：当前注册图像="
              << reconstruction_->NumRegImages()
              << "，BA 前三维点=" << num_points_before_ba
              << "，当前三维点=" << reconstruction_->NumPoints3D()
              << "，几何过滤观测=" << num_filtered_points_by_geom
              << "，短轨迹过滤观测=" << num_filtered_points_by_track;

    if (reconstruction_->NumPoints3D() == 0) {
      LOG(ERROR) << "All 3D points were filtered after bundle adjustment";
      return false;
    }
  }

  return true;
}

bool GlomapMapper::IterativeRetriangulateAndRefine(const GlomapOptions& options) {
  // 这里不直接在当前 reconstruction 上清空再重建，而是新建一个副本：
  // 1. 保留当前更可靠的相机姿态；
  // 2. 重新走旧版 IncrementalMapper 的重三角化；
  // 3. 成功后再整体替换回主 reconstruction。
  auto refined_reconstruction = std::make_shared<Reconstruction>();
  refined_reconstruction->Load(*database_cache_);
  refined_reconstruction->SetUp(&database_cache_->CorrespondenceGraph());

  for (const auto& image_item : reconstruction_->Images()) {
    const image_t image_id = image_item.first;
    if (!refined_reconstruction->ExistsImage(image_id)) {
      continue;
    }
    Image& refined_image = refined_reconstruction->Image(image_id);
    const Image& current_image = reconstruction_->Image(image_id);
    refined_image.SetQvec(current_image.Qvec());
    refined_image.SetTvec(current_image.Tvec());
    if (current_image.IsRegistered()) {
      refined_reconstruction->RegisterImage(image_id);
    }
  }
  for (const auto& camera_item : reconstruction_->Cameras()) {
    if (refined_reconstruction->ExistsCamera(camera_item.first)) {
      refined_reconstruction->Camera(camera_item.first) = camera_item.second;
    }
  }

  IncrementalMapper mapper(database_cache_.get());
  mapper.BeginReconstruction(refined_reconstruction.get());

  std::vector<RelativePoseConstraint> constraints;
  // 将位姿图边转为旧版 BA 可识别的相对位姿约束，帮助重三角化后的全局精化稳定收敛。
  for (const ViewEdge& edge : pose_graph_->ValidEdges()) {
    RelativePoseConstraint constraint;
    constraint.image_id1 = edge.image_id1;
    constraint.image_id2 = edge.image_id2;
    constraint.qvec12 = edge.geometry.qvec;
    constraint.tvec12 = edge.geometry.tvec;
    constraint.trans_weight = std::max(1.0, std::sqrt(edge.weight));
    constraints.push_back(constraint);
  }
  mapper.SetRelativePoseConstraints(constraints);

  for (const image_t image_id : refined_reconstruction->RegImageIds()) {
    mapper.TriangulateImage(options.retriangulation_options, image_id);
  }

  IncrementalMapper::Options mapper_options = MakeIncrementalMapperOptions(options);
  for (int iter = 0; iter < 3; ++iter) {
    // 一个简化的“补全-合并-重三角化-全局 BA”循环，用来逼近新版全局管线里的 refine 行为。
    mapper.CompleteTracks(options.retriangulation_options);
    mapper.MergeTracks(options.retriangulation_options);
    mapper.Retriangulate(options.retriangulation_options);
    if (!mapper.AdjustGlobalBundle(mapper_options,
                                   options.bundle_adjustment_options)) {
      LOG(ERROR) << "Incremental-style global refinement failed during retriangulation";
      mapper.EndReconstruction(false);
      return false;
    }
    mapper.FilterPoints(mapper_options);
  }

  mapper.EndReconstruction(false);
  ObservationManager refined_obs_manager(*refined_reconstruction);
  refined_obs_manager.FilterAllPoints3D(options.max_reproj_error,
                                        options.min_tri_angle);
  refined_reconstruction->Normalize();

  *reconstruction_ = *refined_reconstruction;
  LogStageSummary("重三角化与全局精化", *reconstruction_, pose_graph_.get());
  return reconstruction_->NumPoints3D() > 0;
}

bool GlomapMapper::Solve(const GlomapOptions& options) {
  CHECK_NOTNULL(reconstruction_.get());
  const std::shared_ptr<Reconstruction> output_reconstruction = reconstruction_;

  auto run_attempt = [&](const GlomapOptions& attempt_options) -> bool {
    std::shared_ptr<Reconstruction> working_reconstruction =
        std::make_shared<Reconstruction>();
    // 每次尝试都从干净状态开始，这样 PoseLib 失败回退时不会污染第二次求解。
    reconstruction_ = working_reconstruction;
    BeginReconstruction(reconstruction_);
    if (!PreparePoseGraph(attempt_options)) {
      return false;
    }
    LOG(INFO) << "开始全局 SfM：有效边="
              << pose_graph_->NumValidEdges()
              << "，图像总数="
              << reconstruction_->NumImages();

    if (!attempt_options.skip_rotation_averaging) {
      LOG(INFO) << "开始执行旋转平均";
      if (!RotationAveraging(attempt_options)) {
        return false;
      }
    }

    if (!attempt_options.skip_track_establishment) {
      LOG(INFO) << "开始执行轨迹建立";
      EstablishTracks(attempt_options);
    }

    if (!attempt_options.skip_global_positioning) {
      LOG(INFO) << "开始执行全局定位";
      if (!GlobalPositioning(attempt_options)) {
        return false;
      }
    }

    if (!attempt_options.skip_bundle_adjustment) {
      if (attempt_options.use_three_stage_optimization) {
        // 三阶段优化与 test_vpgo_pipeline.cc 的思想一致：
        // 阶段1 宽松阈值粗优化，阶段2 收紧阈值细化，阶段3 再放开内参。
        const std::vector<GlomapOptimizationStageOptions> stage_sequence = {
            attempt_options.stage1_options,
            attempt_options.stage2_options,
            attempt_options.stage3_options};
        for (const GlomapOptimizationStageOptions& stage_options :
             stage_sequence) {
          if (stage_options.max_iterations <= 0) {
            continue;
          }

          const GlomapOptions stage_scoped_options =
              BuildStageScopedOptions(attempt_options, stage_options);
          LOG(INFO) << "开始执行" << stage_options.name
                    << "：阶段迭代次数=" << stage_options.max_iterations
                    << "，三角化阈值=" << stage_options.tri_max_project_error
                    << "，过滤阈值=" << stage_options.filter_max_reproj_error
                    << "，优化焦距="
                    << (stage_options.refine_focal_length ? "是" : "否")
                    << "，优化主点="
                    << (stage_options.refine_principal_point ? "是" : "否")
                    << "，优化畸变="
                    << (stage_options.refine_extra_params ? "是" : "否");

          if (!IterativeBundleAdjustment(stage_scoped_options)) {
            LOG(ERROR) << stage_options.name << " 的 BA 阶段失败";
            return false;
          }

          if (!stage_scoped_options.skip_retriangulation &&
              !IterativeRetriangulateAndRefine(stage_scoped_options)) {
            LOG(ERROR) << stage_options.name << " 的重三角化阶段失败";
            return false;
          }
        }
      } else {
        LOG(INFO) << "开始执行迭代全局 BA";
        if (!IterativeBundleAdjustment(attempt_options)) {
          LOG(ERROR) << "Iterative bundle adjustment stage failed";
          return false;
        }
      }
    }

    if (!attempt_options.use_three_stage_optimization &&
        !attempt_options.skip_retriangulation) {
      LOG(INFO) << "开始执行重三角化与全局精化";
      if (!IterativeRetriangulateAndRefine(attempt_options)) {
        LOG(ERROR) << "Iterative retriangulation and refinement stage failed";
        return false;
      }
    }

    reconstruction_->TearDown();
    if (reconstruction_->NumRegImages() < attempt_options.min_num_reg_images ||
        reconstruction_->NumPoints3D() == 0) {
      return false;
    }

    *output_reconstruction = *reconstruction_;
    reconstruction_ = output_reconstruction;
    return true;
  };

  if (run_attempt(options)) {
    return true;
  }

  if (options.run_poselib_relpose) {
    GlomapOptions fallback_options = options;
    fallback_options.run_poselib_relpose = false;
    // 如果 PoseLib 重估后的边集让全局链路失败，则自动回退到数据库原始相对位姿。
    LOG(WARNING) << "Retrying global mapping with original database relative poses";
    return run_attempt(fallback_options);
  }

  return false;
}

std::shared_ptr<class Reconstruction> GlomapMapper::ReconstructionPtr() const {
  return reconstruction_;
}

bool RunGlomapMapper(const std::string& database_path,
                     const GlomapOptions& options,
                     Reconstruction* reconstruction) {
  Database database(database_path);
  return RunGlomapMapper(database, options, reconstruction);
}

bool RunGlomapMapper(const Database& database,
                     const GlomapOptions& options,
                     Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  auto database_cache = std::make_shared<DatabaseCache>();
  database_cache->Load(
      database, options.min_num_matches, options.ignore_watermarks, {});

  std::shared_ptr<Reconstruction> owned_reconstruction =
      std::make_shared<Reconstruction>();
  GlomapMapper mapper(database_cache, &database);
  mapper.BeginReconstruction(owned_reconstruction);
  if (!mapper.Solve(options)) {
    return false;
  }

  *reconstruction = *owned_reconstruction;
  return true;
}

}  // namespace colmap
