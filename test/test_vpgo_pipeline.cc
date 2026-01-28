#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ceres/ceres.h>

#include "base/database.h"
#include "base/pose.h"
#include "base/reconstruction.h"
#include "controllers/bundle_adjustment.h"
#include "controllers/incremental_mapper.h"
#include "estimators/similarity_transform.h"
#include "estimators/homography_matrix.h"
#include "estimators/pose.h"
#include "estimators/two_view_geometry.h"
#include "estimators/utils.h"
#include "feature/utils.h"
#include "optim/ransac.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/string.h"

using namespace colmap;

namespace {

struct SE3RelativePoseCost {
  SE3RelativePoseCost(const Eigen::Quaterniond& q_ij,
                      const Eigen::Vector3d& t_ij_unit, double rot_weight,
                      double trans_weight)
      : q_ij_(q_ij),
        t_ij_unit_(t_ij_unit),
        rot_weight_(rot_weight),
        trans_weight_(trans_weight) {}

  template <typename T>
  bool operator()(const T* const pose_i, const T* const pose_j,
                  T* residuals) const {
    Eigen::Quaternion<T> q_i(pose_i[0], pose_i[1], pose_i[2], pose_i[3]);
    Eigen::Matrix<T, 3, 1> t_i(pose_i[4], pose_i[5], pose_i[6]);

    Eigen::Quaternion<T> q_j(pose_j[0], pose_j[1], pose_j[2], pose_j[3]);
    Eigen::Matrix<T, 3, 1> t_j(pose_j[4], pose_j[5], pose_j[6]);

    const Eigen::Quaternion<T> q_ij = q_ij_.cast<T>();
    const Eigen::Matrix<T, 3, 1> t_ij_obs = t_ij_unit_.cast<T>();

    Eigen::Quaternion<T> q_ij_pred = q_j * q_i.conjugate();
    Eigen::Quaternion<T> dq = q_ij.conjugate() * q_ij_pred;

    Eigen::Matrix<T, 3, 1> r_rot;
    r_rot << T(2) * dq.x(), T(2) * dq.y(), T(2) * dq.z();

    Eigen::Matrix<T, 3, 1> t_pred = t_j - q_j * (q_i.conjugate() * t_i);
    t_pred.normalize();

    Eigen::Matrix<T, 3, 1> r_trans = t_pred.cross(t_ij_obs);

    residuals[0] = T(rot_weight_) * r_rot(0);
    residuals[1] = T(rot_weight_) * r_rot(1);
    residuals[2] = T(rot_weight_) * r_rot(2);
    residuals[3] = T(trans_weight_) * r_trans(0);
    residuals[4] = T(trans_weight_) * r_trans(1);
    residuals[5] = T(trans_weight_) * r_trans(2);

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Quaterniond& q_ij,
                                     const Eigen::Vector3d& t_ij_unit,
                                     double rot_weight, double trans_weight) {
    return new ceres::AutoDiffCostFunction<SE3RelativePoseCost, 6, 7, 7>(
        new SE3RelativePoseCost(q_ij, t_ij_unit, rot_weight, trans_weight));
  }

 private:
  Eigen::Quaterniond q_ij_;
  Eigen::Vector3d t_ij_unit_;
  double rot_weight_;
  double trans_weight_;
};

struct SE3RelativePoseCostFull {
  SE3RelativePoseCostFull(const Eigen::Quaterniond& q_ij,
                          const Eigen::Vector3d& t_ij, double rot_weight,
                          double trans_weight)
      : q_ij_(q_ij),
        t_ij_(t_ij),
        rot_weight_(rot_weight),
        trans_weight_(trans_weight) {}

  template <typename T>
  bool operator()(const T* const pose_i, const T* const pose_j,
                  T* residuals) const {
    Eigen::Quaternion<T> q_i(pose_i[0], pose_i[1], pose_i[2], pose_i[3]);
    Eigen::Matrix<T, 3, 1> t_i(pose_i[4], pose_i[5], pose_i[6]);

    Eigen::Quaternion<T> q_j(pose_j[0], pose_j[1], pose_j[2], pose_j[3]);
    Eigen::Matrix<T, 3, 1> t_j(pose_j[4], pose_j[5], pose_j[6]);

    const Eigen::Quaternion<T> q_ij = q_ij_.cast<T>();
    const Eigen::Matrix<T, 3, 1> t_ij_obs = t_ij_.cast<T>();

    Eigen::Quaternion<T> q_ij_pred = q_j * q_i.conjugate();
    Eigen::Quaternion<T> dq = q_ij.conjugate() * q_ij_pred;

    Eigen::Matrix<T, 3, 1> r_rot;
    r_rot << T(2) * dq.x(), T(2) * dq.y(), T(2) * dq.z();

    Eigen::Matrix<T, 3, 1> t_pred = t_j - q_j * (q_i.conjugate() * t_i);
    Eigen::Matrix<T, 3, 1> r_trans = t_pred - t_ij_obs;

    residuals[0] = T(rot_weight_) * r_rot(0);
    residuals[1] = T(rot_weight_) * r_rot(1);
    residuals[2] = T(rot_weight_) * r_rot(2);
    residuals[3] = T(trans_weight_) * r_trans(0);
    residuals[4] = T(trans_weight_) * r_trans(1);
    residuals[5] = T(trans_weight_) * r_trans(2);

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Quaterniond& q_ij,
                                     const Eigen::Vector3d& t_ij,
                                     double rot_weight, double trans_weight) {
    return new ceres::AutoDiffCostFunction<SE3RelativePoseCostFull, 6, 7, 7>(
        new SE3RelativePoseCostFull(q_ij, t_ij, rot_weight, trans_weight));
  }

 private:
  Eigen::Quaterniond q_ij_;
  Eigen::Vector3d t_ij_;
  double rot_weight_;
  double trans_weight_;
};

class SE3Manifold : public ceres::Manifold {
 public:
  int AmbientSize() const override { return 7; }
  int TangentSize() const override { return 6; }

  bool Plus(const double* x, const double* delta,
            double* x_plus_delta) const override {
    Eigen::Quaterniond q(x[0], x[1], x[2], x[3]);
    Eigen::Vector3d t(x[4], x[5], x[6]);

    Eigen::Vector3d omega(delta[0], delta[1], delta[2]);
    Eigen::Vector3d upsilon(delta[3], delta[4], delta[5]);

    const double theta = omega.norm();
    Eigen::Quaterniond dq = Eigen::Quaterniond::Identity();
    if (theta > 1e-12) {
      dq = Eigen::Quaterniond(Eigen::AngleAxisd(theta, omega / theta));
    }

    const Eigen::Quaterniond q_new = (dq * q).normalized();
    const Eigen::Vector3d t_new = t + upsilon;

    x_plus_delta[0] = q_new.w();
    x_plus_delta[1] = q_new.x();
    x_plus_delta[2] = q_new.y();
    x_plus_delta[3] = q_new.z();
    x_plus_delta[4] = t_new.x();
    x_plus_delta[5] = t_new.y();
    x_plus_delta[6] = t_new.z();

    return true;
  }

  bool PlusJacobian(const double*, double* jacobian) const override {
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
    J.setZero();
    J.block<6, 6>(1, 0).setIdentity();
    return true;
  }

  bool Minus(const double* y, const double* x,
             double* y_minus_x) const override {
    Eigen::Quaterniond qx(x[0], x[1], x[2], x[3]);
    Eigen::Quaterniond qy(y[0], y[1], y[2], y[3]);
    Eigen::Vector3d tx(x[4], x[5], x[6]);
    Eigen::Vector3d ty(y[4], y[5], y[6]);

    Eigen::Quaterniond dq = qy * qx.conjugate();
    Eigen::AngleAxisd aa(dq);

    Eigen::Vector3d omega = aa.axis() * aa.angle();
    Eigen::Vector3d upsilon = ty - tx;

    y_minus_x[0] = omega.x();
    y_minus_x[1] = omega.y();
    y_minus_x[2] = omega.z();
    y_minus_x[3] = upsilon.x();
    y_minus_x[4] = upsilon.y();
    y_minus_x[5] = upsilon.z();

    return true;
  }

  bool MinusJacobian(const double*, double* jacobian) const override {
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobian);
    J.setZero();
    J.block<6, 6>(0, 1).setIdentity();
    return true;
  }
};

enum class EdgeType { kOdom, kLoop };

struct EdgeDefaults {
  double odom_rot_weight = 5.0;
  double odom_trans_weight = 5.0;
  double loop_rot_weight = 20.0;
  double loop_trans_weight = 20.0;
  bool odom_translation_is_unit = false;
  bool loop_translation_is_unit = true;
};

struct PoseGraphEdge {
  EdgeType type = EdgeType::kLoop;
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  Eigen::Quaterniond q_ij = Eigen::Quaterniond::Identity();
  Eigen::Vector3d t_ij = Eigen::Vector3d::Zero();
  bool translation_is_unit = true;
  double rot_weight = 1.0;
  double trans_weight = 1.0;
};

struct PipelineOptions {
  std::string sparse_path;
  std::string database_path;
  std::string output_path;
  std::string edge_input_path;
  std::string edge_output_path;
  std::string manual_loop_path;
  bool build_odom_edges = true;
  bool build_loop_edges = true;
  bool run_self_test = false;
  double eval_threshold_px = -1.0;
  int min_loop_id_gap = 100;
  size_t min_loop_inliers = 200;
  double min_loop_inlier_ratio = 0.2;
  double max_loop_geom_error_px = 4.0;
  double min_loop_tri_angle_deg = 1.0;
  size_t max_loop_edges_per_image = 30;
  EdgeDefaults defaults;
};

bool IsNumericToken(const std::string& token) {
  char* end_ptr = nullptr;
  std::strtod(token.c_str(), &end_ptr);
  return end_ptr != token.c_str() && *end_ptr == '\0';
}

std::string EdgeTypeToString(const EdgeType type) {
  return type == EdgeType::kOdom ? "odom" : "loop";
}

/**
 * [功能描述]：解析一行文本，提取位姿图边(Edge)的信息
 * 支持两种格式：
 *   1. 带类型: type id1 id2 qw qx qy qz tx ty tz [is_unit rot_weight trans_weight]
 *   2. 无类型: id1 id2 qw qx qy qz tx ty tz [is_unit rot_weight trans_weight] (默认为loop)
 * @param line：待解析的文本行
 * @param defaults：边的默认参数配置
 * @param edge：输出参数，存储解析后的边信息
 * @return 解析成功返回true，失败返回false
 */
bool ParseEdgeLine(const std::string& line, const EdgeDefaults& defaults,
                   PoseGraphEdge* edge) {
  std::string content = line;
  
  // 移除注释部分（'#'之后的内容）
  const auto comment_pos = content.find('#');
  if (comment_pos != std::string::npos) {
    content = content.substr(0, comment_pos);
  }
  
  // 去除首尾空白字符，若为空行则返回失败
  StringTrim(&content);
  if (content.empty()) {
    return false;
  }

  // 创建字符串流用于逐个解析字段
  std::istringstream iss(content);
  std::string first;
  if (!(iss >> first)) {
    return false;
  }

  // 判断第一个token是否为类型标识（非数字则为类型）
  bool has_type = !IsNumericToken(first);
  std::string type_token;
  int64_t id1 = -1;   // 边的起始节点ID
  int64_t id2 = -1;   // 边的终止节点ID
  // 四元数表示的相对旋转 (qw, qx, qy, qz)
  double qw = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  // 相对平移向量 (tx, ty, tz)
  double tx = 0.0;
  double ty = 0.0;
  double tz = 0.0;

  if (has_type) {
    // 格式：type id1 id2 qw qx qy qz tx ty tz
    type_token = first;
    StringToLower(&type_token);  // 转为小写以便比较
    if (!(iss >> id1 >> id2 >> qw >> qx >> qy >> qz >> tx >> ty >> tz)) {
      return false;
    }
  } else {
    // 格式：id1 id2 qw qx qy qz tx ty tz（无类型，默认为loop）
    id1 = std::stoll(first);
    if (!(iss >> id2 >> qw >> qx >> qy >> qz >> tx >> ty >> tz)) {
      return false;
    }
    type_token = "loop";
  }

  // 填充解析结果
  PoseGraphEdge parsed;
  parsed.type = (type_token == "odom") ? EdgeType::kOdom : EdgeType::kLoop;
  parsed.image_id1 = static_cast<image_t>(id1);
  parsed.image_id2 = static_cast<image_t>(id2);
  parsed.q_ij = Eigen::Quaterniond(qw, qx, qy, qz);
  // 对四元数进行归一化（避免数值误差）
  if (parsed.q_ij.norm() > 1e-12) {
    parsed.q_ij.normalize();
  }
  Eigen::Vector3d t_ij = Eigen::Vector3d(tx, ty, tz);
  parsed.t_ij = t_ij.normalized();  // 需要归一化，因为 loop_edges_gt.txt 中记录的平移是原始向量，没有归一化

  // 根据边类型设置默认权重参数
  if (parsed.type == EdgeType::kOdom) {
    parsed.translation_is_unit = defaults.odom_translation_is_unit;
    parsed.rot_weight = defaults.odom_rot_weight;
    parsed.trans_weight = defaults.odom_trans_weight;
  } else {
    parsed.translation_is_unit = defaults.loop_translation_is_unit;
    parsed.rot_weight = defaults.loop_rot_weight;
    parsed.trans_weight = defaults.loop_trans_weight;
  }

  // 解析可选的额外参数（is_unit, rot_weight, trans_weight）
  std::vector<double> extra;
  double extra_value = 0.0;
  while (iss >> extra_value) {
    extra.push_back(extra_value);
  }

  // 如果有额外参数，则覆盖默认值
  if (!extra.empty()) {
    parsed.translation_is_unit = (std::abs(extra[0]) > 0.5);  // >0.5视为true
  }
  if (extra.size() >= 2) {
    parsed.rot_weight = extra[1];
  }
  if (extra.size() >= 3) {
    parsed.trans_weight = extra[2];
  }

  // 若平移为单位向量模式，则归一化平移向量
  if (parsed.translation_is_unit) {
    const double t_norm = parsed.t_ij.norm();
    if (t_norm > 1e-12) {
      parsed.t_ij /= t_norm;
    }
  }

  *edge = parsed;
  return true;
}

/**
 * [功能描述]：从输入流中逐行读取并解析位姿图边信息
 * @param in：输入流（如文件流）
 * @param defaults：边的默认参数配置
 * @param edges：输出参数，存储所有解析成功的边
 * @param error：输出参数，存储错误信息（可为nullptr）
 * @return 至少解析出一条有效边返回true，否则返回false
 */
bool ReadPoseGraphEdges(std::istream& in, const EdgeDefaults& defaults,
                        std::vector<PoseGraphEdge>* edges,
                        std::string* error) {
  std::string line;
  // 逐行读取输入流
  while (std::getline(in, line)) {
    PoseGraphEdge edge;
    // 解析失败则跳过该行
    if (!ParseEdgeLine(line, defaults, &edge)) {
      continue;
    }
    // 跳过包含无效图像ID的边
    if (edge.image_id1 == kInvalidImageId ||
        edge.image_id2 == kInvalidImageId) {
      continue;
    }
    edges->push_back(edge);
  }

  // 若未解析到任何有效边，返回错误
  if (edges->empty()) {
    if (error != nullptr) {
      *error = "no valid edges parsed";
    }
    return false;
  }
  return true;
}

/**
 * [功能描述]：从文件中读取位姿图边信息
 * @param path：文件路径，若为"-"则从标准输入读取
 * @param defaults：边的默认参数配置
 * @param edges：输出参数，存储所有解析成功的边
 * @param error：输出参数，存储错误信息（可为nullptr）
 * @return 读取并解析成功返回true，否则返回false
 */
bool ReadPoseGraphEdgesFromFile(const std::string& path,
                                const EdgeDefaults& defaults,
                                std::vector<PoseGraphEdge>* edges,
                                std::string* error) {
  // 支持从标准输入读取（路径为"-"）
  if (path == "-") {
    return ReadPoseGraphEdges(std::cin, defaults, edges, error);
  }
  // 打开文件
  std::ifstream file(path);
  if (!file.is_open()) {
    if (error != nullptr) {
      *error = "could not open edge file";
    }
    return false;
  }
  // 从文件流读取边信息
  return ReadPoseGraphEdges(file, defaults, edges, error);
}

/**
 * [功能描述]：将位姿图边信息写入输出流
 * @param out：输出流（如文件流或标准输出）
 * @param edges：待写入的边列表
 */
void WritePoseGraphEdges(std::ostream& out,
                         const std::vector<PoseGraphEdge>& edges,
                         const bool only_loop_edges = false) {
  // 写入表头注释行，说明各字段含义
  out << "# type id1 id2 qw qx qy qz tx ty tz t_is_unit rot_weight trans_weight\n";
  // 设置高精度输出（17位有效数字）
  out << std::setprecision(17);
  // 遍历每条边，按格式输出
  for (const auto& edge : edges) {
    if (only_loop_edges && edge.type != EdgeType::kLoop) {
      continue;
    }
    out << EdgeTypeToString(edge.type) << " " << edge.image_id1 << " "
        << edge.image_id2 << " " << edge.q_ij.w() << " " << edge.q_ij.x()
        << " " << edge.q_ij.y() << " " << edge.q_ij.z() << " " << edge.t_ij.x()
        << " " << edge.t_ij.y() << " " << edge.t_ij.z() << " "
        << (edge.translation_is_unit ? 1 : 0) << " " << edge.rot_weight << " "
        << edge.trans_weight << "\n";
  }
}

/**
 * [功能描述]：将位姿图边信息写入文件
 * @param path：文件路径，若为"-"则输出到标准输出
 * @param edges：待写入的边列表
 * @param error：输出参数，存储错误信息（可为nullptr）
 * @return 写入成功返回true，否则返回false
 */
bool WritePoseGraphEdgesToFile(const std::string& path,
                               const std::vector<PoseGraphEdge>& edges,
                               std::string* error) {
  // 支持输出到标准输出（路径为"-"）
  if (path == "-") {
    WritePoseGraphEdges(std::cout, edges, false);
    return true;
  }
  // 打开文件
  std::ofstream file(path);
  if (!file.is_open()) {
    if (error != nullptr) {
      *error = "could not write edge file";
    }
    return false;
  }
  // 写入文件
  WritePoseGraphEdges(file, edges, false);
  return true;
}

/**
 * [功能描述]：获取图像的2D特征点坐标（带缓存机制）
 * @param image_id：图像ID
 * @param database：数据库指针，用于读取特征点
 * @param points_cache：特征点缓存，避免重复读取数据库
 * @return 图像特征点的2D坐标向量引用
 */
const std::vector<Eigen::Vector2d>& GetOrLoadImagePoints(
    const image_t image_id, Database* database,
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>>* points_cache) {
  // 先从缓存中查找
  auto it = points_cache->find(image_id);
  if (it != points_cache->end()) {
    return it->second;  // 缓存命中，直接返回
  }

  // 缓存未命中，从数据库读取特征点并转换为2D坐标
  FeatureKeypoints keypoints = database->ReadKeypoints(image_id);
  auto points = FeatureKeypointsToPointsVector(keypoints);
  // 存入缓存并返回
  auto emplace_result = points_cache->emplace(image_id, std::move(points));
  return emplace_result.first->second;
}

/**
 * [功能描述]：检验两视图几何是否通过几何误差过滤
 *            根据不同的几何配置（本质矩阵E/基础矩阵F/单应矩阵H）计算Sampson误差
 * @param geom：两视图几何信息
 * @param image_i：第一幅图像
 * @param image_j：第二幅图像
 * @param camera_i：第一幅图像的相机
 * @param camera_j：第二幅图像的相机
 * @param database：数据库指针
 * @param points_cache：特征点缓存
 * @param max_error_px：最大允许误差（像素）
 * @return 中值误差小于阈值返回true，否则返回false
 */
bool PassesGeometricErrorFilter(
    const TwoViewGeometry& geom, const Image& image_i, const Image& image_j,
    const Camera& camera_i, const Camera& camera_j, Database* database,
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>>* points_cache,
    const double max_error_px) {
  // 无内点匹配则直接返回失败
  if (geom.inlier_matches.empty()) {
    return false;
  }

  // 获取两幅图像的特征点坐标（像素坐标）
  const auto& points1_px =
      GetOrLoadImagePoints(image_i.ImageId(), database, points_cache);
  const auto& points2_px =
      GetOrLoadImagePoints(image_j.ImageId(), database, points_cache);
  if (points1_px.empty() || points2_px.empty()) {
    return false;
  }

  // 提取内点匹配对应的2D点坐标
  std::vector<Eigen::Vector2d> inlier_points1_px;
  std::vector<Eigen::Vector2d> inlier_points2_px;
  inlier_points1_px.reserve(geom.inlier_matches.size());
  inlier_points2_px.reserve(geom.inlier_matches.size());
  for (const auto& match : geom.inlier_matches) {
    // 检查索引有效性
    if (match.point2D_idx1 >= points1_px.size() ||
        match.point2D_idx2 >= points2_px.size()) {
      continue;
    }
    inlier_points1_px.push_back(points1_px[match.point2D_idx1]);
    inlier_points2_px.push_back(points2_px[match.point2D_idx2]);
  }

  if (inlier_points1_px.empty()) {
    return false;
  }

  std::vector<double> residuals;
  residuals.reserve(inlier_points1_px.size());

  // 情况1：已标定相机，使用本质矩阵E计算误差
  if (geom.config == TwoViewGeometry::CALIBRATED &&
      geom.E.squaredNorm() > 1e-12) {
    // 将像素坐标转换为归一化坐标
    std::vector<Eigen::Vector2d> inlier_points1_norm;
    std::vector<Eigen::Vector2d> inlier_points2_norm;
    inlier_points1_norm.reserve(inlier_points1_px.size());
    inlier_points2_norm.reserve(inlier_points1_px.size());
    for (size_t k = 0; k < inlier_points1_px.size(); ++k) {
      inlier_points1_norm.push_back(
          camera_i.ImageToWorld(inlier_points1_px[k]));
      inlier_points2_norm.push_back(
          camera_j.ImageToWorld(inlier_points2_px[k]));
    }
    // 计算Sampson误差
    ComputeSquaredSampsonError(inlier_points1_norm, inlier_points2_norm, geom.E,
                               &residuals);
    // 将像素误差阈值转换为归一化坐标下的阈值
    const double max_error_norm =
        (camera_i.ImageToWorldThreshold(max_error_px) +
         camera_j.ImageToWorldThreshold(max_error_px)) /
        2.0;
    return Median(residuals) <= max_error_norm * max_error_norm;
  }

  // 情况2：使用基础矩阵F计算误差（未标定或已标定）
  if (geom.F.squaredNorm() > 1e-12 &&
      (geom.config == TwoViewGeometry::UNCALIBRATED ||
       geom.config == TwoViewGeometry::CALIBRATED)) {
    ComputeSquaredSampsonError(inlier_points1_px, inlier_points2_px, geom.F,
                               &residuals);
    return Median(residuals) <= max_error_px * max_error_px;
  }

  // 情况3：使用单应矩阵H计算误差（平面/全景场景）
  if (geom.H.squaredNorm() > 1e-12 &&
      (geom.config == TwoViewGeometry::PLANAR ||
       geom.config == TwoViewGeometry::PANORAMIC ||
       geom.config == TwoViewGeometry::PLANAR_OR_PANORAMIC)) {
    HomographyMatrixEstimator::Residuals(inlier_points1_px, inlier_points2_px,
                                         geom.H, &residuals);
    return Median(residuals) <= max_error_px * max_error_px;
  }

  return false;
}

/**
 * [功能描述]：从两视图几何的内点匹配估计相对位姿（旋转和平移方向）
 * @param geom：两视图几何信息（包含内点匹配）
 * @param image_i：第一幅图像
 * @param image_j：第二幅图像
 * @param camera_i：第一幅图像的相机
 * @param camera_j：第二幅图像的相机
 * @param database：数据库指针
 * @param points_cache：特征点缓存
 * @param q_ij：输出参数，相对旋转四元数
 * @param t_ij_dir：输出参数，相对平移方向（单位向量）
 * @return 估计成功返回true，否则返回false
 */
bool EstimateRelativePoseFromInliers(
    const TwoViewGeometry& geom, const Image& image_i, const Image& image_j,
    const Camera& camera_i, const Camera& camera_j, Database* database,
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>>* points_cache,
    Eigen::Quaterniond* q_ij, Eigen::Vector3d* t_ij_dir) {
  // 无内点匹配则返回失败
  if (geom.inlier_matches.empty()) {
    return false;
  }

  // 检查几何配置类型是否支持
  if (geom.config != TwoViewGeometry::CALIBRATED &&
      geom.config != TwoViewGeometry::UNCALIBRATED &&
      geom.config != TwoViewGeometry::PLANAR &&
      geom.config != TwoViewGeometry::PANORAMIC &&
      geom.config != TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    return false;
  }

  // 获取两幅图像的特征点
  const auto& points1 =
      GetOrLoadImagePoints(image_i.ImageId(), database, points_cache);
  const auto& points2 =
      GetOrLoadImagePoints(image_j.ImageId(), database, points_cache);
  if (points1.empty() || points2.empty()) {
    return false;
  }

  // 提取内点匹配对应的2D点坐标
  std::vector<Eigen::Vector2d> inlier_points1;
  std::vector<Eigen::Vector2d> inlier_points2;
  inlier_points1.reserve(geom.inlier_matches.size());
  inlier_points2.reserve(geom.inlier_matches.size());
  for (const auto& match : geom.inlier_matches) {
    if (match.point2D_idx1 >= points1.size() ||
        match.point2D_idx2 >= points2.size()) {
      continue;
    }
    inlier_points1.push_back(points1[match.point2D_idx1]);
    inlier_points2.push_back(points2[match.point2D_idx2]);
  }

  if (inlier_points1.empty()) {
    return false;
  }

  // 估计相对位姿
  TwoViewGeometry geom_copy = geom;
  if (!geom_copy.EstimateRelativePose(camera_i, inlier_points1, camera_j,
                                      inlier_points2)) {
    return false;
  }

  // 再次检查估计后的几何配置类型
  if (geom_copy.config != TwoViewGeometry::CALIBRATED &&
      geom_copy.config != TwoViewGeometry::UNCALIBRATED &&
      geom_copy.config != TwoViewGeometry::PLANAR &&
      geom_copy.config != TwoViewGeometry::PANORAMIC &&
      geom_copy.config != TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    return false;
  }

  // 检查结果有效性（四元数和平移向量不能为零）
  if (geom_copy.qvec.squaredNorm() < 1e-12 ||
      geom_copy.tvec.squaredNorm() < 1e-12) {
    return false;
  }

  // 输出归一化的四元数和单位平移方向
  const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(geom_copy.qvec);
  *q_ij = Eigen::Quaterniond(normalized_qvec(0), normalized_qvec(1),
                             normalized_qvec(2), normalized_qvec(3));
  *t_ij_dir = geom_copy.tvec.normalized();
  return true;
}

/**
 * [功能描述]：使用2D-2D内点匹配对应的不同3D点估计回环平移尺度
 *            将匹配提升为3D-3D对应，通过Sim3(Umeyama+RANSAC)估计
 *            i->j 的相似变换，取平移向量模长作为尺度
 * @param reconstruction：3D重建结果
 * @param geom：两视图几何信息（包含内点匹配）
 * @param image_i：第一幅图像
 * @param image_j：第二幅图像
 * @param q_ij：相对旋转（从i到j）
 * @param t_ij_dir：相对平移方向（单位向量）
 * @param scale：输出参数，估计得到的尺度
 * @param q_ij_out：输出参数，估计得到的相对旋转四元数
 * @param t_ij_out：输出参数，估计得到的相对平移
 * @return 估计成功返回true，否则返回false
 */
bool EstimateLoopScaleFromPointPairsSim3(
    const Reconstruction& reconstruction, const TwoViewGeometry& geom,
    const Image& image_i, const Image& image_j,
    const Eigen::Quaterniond& q_ij, const Eigen::Vector3d& t_ij_dir,
    double* scale, Eigen::Quaterniond& q_ij_out, Eigen::Vector3d& t_ij_out) {
  if (scale == nullptr) {
    return false;
  }
  if (t_ij_dir.squaredNorm() < 1e-12) {
    return false;
  }

  const size_t kMinPointPairs = 12;  // 最少匹配点对数
  const double kMinDepth = 1e-6;  // 最小深度阈值
  const double kMinAbsTransCos = 0.2;  // 最小绝对平移余弦值
  const double kMaxRotDiffRad = DegToRad(30.0);  // 最大旋转差异阈值

  // 提取匹配点对应的3D点
  std::vector<Eigen::Vector3d> points_i;
  std::vector<Eigen::Vector3d> points_j;
  points_i.reserve(geom.inlier_matches.size());
  points_j.reserve(geom.inlier_matches.size());

  // 获取两幅图像的旋转和平移矩阵
  const Eigen::Matrix3d R_i = image_i.RotationMatrix();
  const Eigen::Vector3d t_i = image_i.Tvec();
  const Eigen::Matrix3d R_j = image_j.RotationMatrix();
  const Eigen::Vector3d t_j = image_j.Tvec();

  // 遍历匹配点，计算对应3D点在两视图相机坐标系下的坐标
  for (const auto& match : geom.inlier_matches) {
    if (match.point2D_idx1 >= image_i.NumPoints2D() ||
        match.point2D_idx2 >= image_j.NumPoints2D()) {
      continue;
    }

    // 获取匹配点对应的2D点
    const Point2D& point2D_i = image_i.Point2D(match.point2D_idx1);
    const Point2D& point2D_j = image_j.Point2D(match.point2D_idx2);
    if (!point2D_i.HasPoint3D() || !point2D_j.HasPoint3D()) {
      continue;
    }

    // 获取匹配点对应的3D点
    const point3D_t point3D_id_i = point2D_i.Point3DId();
    const point3D_t point3D_id_j = point2D_j.Point3DId();
    if (!reconstruction.ExistsPoint3D(point3D_id_i) ||
        !reconstruction.ExistsPoint3D(point3D_id_j)) {
      continue;
    }

    // 计算3D点在两视图相机坐标系下的坐标
    const Point3D& point3D_i = reconstruction.Point3D(point3D_id_i);
    const Point3D& point3D_j = reconstruction.Point3D(point3D_id_j);
    const Eigen::Vector3d xyz_i = R_i * point3D_i.XYZ() + t_i;
    const Eigen::Vector3d xyz_j = R_j * point3D_j.XYZ() + t_j;
    if (!xyz_i.allFinite() || !xyz_j.allFinite()) {
      continue;
    }
    if (xyz_i.z() <= kMinDepth || xyz_j.z() <= kMinDepth) {
      continue;
    }

    points_i.push_back(xyz_i);
    points_j.push_back(xyz_j);
  }

  // 检查是否满足最少匹配点数要求
  if (points_i.size() < kMinPointPairs) {
    return false;
  }

  // 计算匹配点对应的3D点在两视图相机坐标系下的尺度
  std::vector<double> norms;
  norms.reserve(points_i.size() + points_j.size());
  for (const auto& p : points_i) {
    norms.push_back(p.norm());
  }
  for (const auto& p : points_j) {
    norms.push_back(p.norm());
  }
  const double median_norm = Median(norms);
  if (!std::isfinite(median_norm) || median_norm <= kMinDepth) {
    return false;
  }

  // 设置RANSAC参数
  RANSACOptions ransac_options;
  ransac_options.max_error =
      std::max(1e-12, std::pow(0.05 * median_norm, 2));
  ransac_options.min_inlier_ratio = 0.2;
  ransac_options.confidence = 0.999;
  ransac_options.min_num_trials = 50;
  ransac_options.max_num_trials = 2000;

  // 使用相似变换估计器
  using Sim3Estimator = SimilarityTransformEstimator<3, true>;
  RANSAC<Sim3Estimator> ransac(ransac_options);
  const auto report = ransac.Estimate(points_i, points_j);
  if (!report.success) {
    return false;
  }
  if (report.support.num_inliers < kMinPointPairs) {
    return false;
  }

  // 提取内点
  std::vector<Eigen::Vector3d> inlier_i;
  std::vector<Eigen::Vector3d> inlier_j;
  inlier_i.reserve(report.support.num_inliers);
  inlier_j.reserve(report.support.num_inliers);
  for (size_t idx = 0; idx < report.inlier_mask.size(); ++idx) {
    if (report.inlier_mask[idx]) {
      inlier_i.push_back(points_i[idx]);
      inlier_j.push_back(points_j[idx]);
    }
  }

  // 检查内点数量是否满足最少匹配点数要求
  if (inlier_i.size() < kMinPointPairs) {
    return false;
  }

  const auto models = Sim3Estimator::Estimate(inlier_i, inlier_j);
  if (models.empty()) {
    return false;
  }

  // 提取相似变换模型
  const auto& model = models[0];
  const Eigen::Matrix3d SR = model.leftCols<3>();
  const Eigen::Vector3d t = model.col(3);
  // 计算尺度因子
  const double s =
      (SR.col(0).norm() + SR.col(1).norm() + SR.col(2).norm()) / 3.0;
  if (!std::isfinite(s) || s <= 0.0) {
    return false;
  }
  if (!t.allFinite()) {
    return false;
  }

  const Eigen::Matrix3d R_sim3 = SR / s;
  if (!R_sim3.allFinite()) {
    return false;
  }
  // 计算相对旋转
  const Eigen::Matrix3d R_ij = q_ij.normalized().toRotationMatrix();
  // 计算相对旋转的余弦值
  double cos_angle = (R_sim3 * R_ij.transpose()).trace();
  cos_angle = (cos_angle - 1.0) * 0.5;
  cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
  // 计算相对旋转的差值
  const double rot_diff = std::acos(cos_angle);
  if (!std::isfinite(rot_diff) || rot_diff > kMaxRotDiffRad) {
    return false;
  }

  // 计算相对平移的模长
  const double t_norm = t.norm();
  // 检查相对平移的模长是否满足最小深度阈值
  if (!std::isfinite(t_norm) || t_norm <= kMinDepth) {
    return false;
  }

  // 检查相对平移的余弦值是否满足最小余弦值阈值
  const double trans_cos =
      std::abs(t.normalized().dot(t_ij_dir.normalized()));
  if (trans_cos < kMinAbsTransCos) {
    return false;
  }
  // TODO: 是否可以直接用估算出来的相对位姿来进行回环约束???
  // std::cout << "scale: " << t_norm << ", inliers: " << report.support.num_inliers << std::endl;

  q_ij_out = Eigen::Quaterniond(R_sim3);
  t_ij_out = t;

  // 设置尺度因子
  *scale = t_norm;
  return true;
}

/**
 * [功能描述]：从重建结果构建里程计边（相邻帧之间的相对位姿）
 * @param reconstruction：3D重建结果
 * @param defaults：边的默认参数配置
 * @return 里程计边列表
 */
std::vector<PoseGraphEdge> BuildOdometryEdges(
    const Reconstruction& reconstruction, const EdgeDefaults& defaults) {
  // 按相机ID分组图像
  std::unordered_map<camera_t, std::vector<image_t>> camera_image_ids;
  camera_image_ids.reserve(reconstruction.Cameras().size());

  for (const auto image_id : reconstruction.RegImageIds()) {
    const Image& image = reconstruction.Image(image_id);
    camera_image_ids[image.CameraId()].push_back(image_id);
  }

  std::vector<PoseGraphEdge> edges;
  // 遍历每个相机的图像序列
  for (auto& pair : camera_image_ids) {
    auto& ids = pair.second;
    std::sort(ids.begin(), ids.end());  // 按ID排序保证时序
    if (ids.size() < 2) {
      continue;  // 至少需要2帧才能构建边
    }

    // 构建相邻帧之间的里程计边
    for (size_t idx = 0; idx + 1 < ids.size(); ++idx) {
      const image_t image_id_i = ids[idx];
      const image_t image_id_j = ids[idx + 1];
      const Image& image_i = reconstruction.Image(image_id_i);
      const Image& image_j = reconstruction.Image(image_id_j);

      // 计算相对位姿（从i到j的变换）
      Eigen::Vector4d qvec_ij;
      Eigen::Vector3d tvec_ij;
      ComputeRelativePose(image_i.Qvec(), image_i.Tvec(), image_j.Qvec(),
                          image_j.Tvec(), &qvec_ij, &tvec_ij);
      
      Eigen::Vector3d tvec_ij_dir = tvec_ij.normalized();

      // 构建里程计边
      PoseGraphEdge edge;
      edge.type = EdgeType::kOdom;
      edge.image_id1 = image_id_i;
      edge.image_id2 = image_id_j;
      edge.q_ij = Eigen::Quaterniond(qvec_ij(0), qvec_ij(1), qvec_ij(2),
                                     qvec_ij(3));
      if (defaults.odom_translation_is_unit) {
        edge.t_ij = tvec_ij_dir;
      } else {
        edge.t_ij = tvec_ij;
      }
      edge.translation_is_unit = defaults.odom_translation_is_unit;
      edge.rot_weight = defaults.odom_rot_weight;
      edge.trans_weight = defaults.odom_trans_weight;
      edges.push_back(edge);
    }
  }

  return edges;
}

std::vector<PoseGraphEdge> BuildOdometryEdges(
    const Reconstruction& reconstruction, Database* database,
    const EdgeDefaults& defaults) {
  // 从数据库读取所有两视图几何
  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  database->ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);

  std::vector<PoseGraphEdge> edges;

  // 遍历所有图像对
  for (size_t idx = 0; idx < image_pair_ids.size(); ++idx) {
    image_t i;
    image_t j;
    Database::PairIdToImagePair(image_pair_ids[idx], &i, &j);
    (void)two_view_geometries[idx];

    // 跳过id间隔大于10的图像对
    if (std::abs(static_cast<int>(i) - static_cast<int>(j)) > 10) {
      continue;
    }

    // 图像必须存在于重建中
    if (!reconstruction.ExistsImage(i) || !reconstruction.ExistsImage(j)) {
      continue;
    }

    // 构建里程计边
    const Image& image_i = reconstruction.Image(i);
    const Image& image_j = reconstruction.Image(j);

    // 计算相对位姿（从i到j的变换）
    Eigen::Vector4d qvec_ij;
    Eigen::Vector3d tvec_ij;
    ComputeRelativePose(image_i.Qvec(), image_i.Tvec(), image_j.Qvec(),
                        image_j.Tvec(), &qvec_ij, &tvec_ij);
    
    Eigen::Vector3d tvec_ij_dir = tvec_ij;
    const double t_norm = tvec_ij_dir.norm();
    if (t_norm > 1e-12) {
      tvec_ij_dir /= t_norm;
    }

    // 构建里程计边
    PoseGraphEdge edge;
    edge.type = EdgeType::kOdom;
    edge.image_id1 = i;
    edge.image_id2 = j;
    edge.q_ij = Eigen::Quaterniond(qvec_ij(0), qvec_ij(1), qvec_ij(2),
                                    qvec_ij(3));
    if (defaults.odom_translation_is_unit) {
      edge.t_ij = tvec_ij_dir;
    } else {
      edge.t_ij = tvec_ij;
    }
    edge.translation_is_unit = defaults.odom_translation_is_unit;
    edge.rot_weight = defaults.odom_rot_weight;
    edge.trans_weight = defaults.odom_trans_weight;
    edges.push_back(edge);
  }
  return edges;
}

/**
 * [功能描述]：从数据库中的两视图几何构建回环边
 *            应用多种过滤条件筛选高质量的回环约束
 * @param reconstruction：3D重建结果
 * @param database：数据库指针
 * @param options：管线配置参数
 * @param occupied_pairs：已占用的图像对集合（输入输出，避免重复）
 * @return 回环边列表
 */
std::vector<PoseGraphEdge> BuildLoopEdges(
    const Reconstruction& reconstruction, Database* database,
    const PipelineOptions& options,
    std::unordered_set<image_pair_t>* occupied_pairs) {
  // 从数据库读取所有两视图几何
  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  database->ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);

  // 特征点缓存，避免重复读取
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>> points_cache;
  points_cache.reserve(reconstruction.RegImageIds().size());

  // 按内点数量降序排序，优先处理高质量匹配
  std::vector<size_t> loop_indices(image_pair_ids.size());
  std::iota(loop_indices.begin(), loop_indices.end(), 0);
  std::sort(loop_indices.begin(), loop_indices.end(),
            [&](size_t a, size_t b) {
              return two_view_geometries[a].inlier_matches.size() >
                     two_view_geometries[b].inlier_matches.size();
            });

  // 记录每个图像的回环边数量（用于限制每图最大回环数）
  std::unordered_map<image_t, size_t> loop_degree;
  loop_degree.reserve(reconstruction.RegImageIds().size());

  const double min_tri_angle = DegToRad(options.min_loop_tri_angle_deg);
  std::vector<PoseGraphEdge> edges;

  for (const size_t idx : loop_indices) {
    image_t i;
    image_t j;
    Database::PairIdToImagePair(image_pair_ids[idx], &i, &j);
    const TwoViewGeometry& geom = two_view_geometries[idx];

    // 过滤1：图像必须存在于重建中
    if (!reconstruction.ExistsImage(i) || !reconstruction.ExistsImage(j)) {
      continue;
    }

    // 过滤2：图像对未被占用
    const image_pair_t pair_id = Database::ImagePairToPairId(i, j);
    if (occupied_pairs->count(pair_id) > 0) {
      continue;
    }

    // 过滤3：图像ID间隔足够大（避免相邻帧）
    if (std::abs(static_cast<int>(i) - static_cast<int>(j)) <
        options.min_loop_id_gap) {
      continue;
    }

    // 过滤4：几何配置类型有效
    if (geom.config == TwoViewGeometry::UNDEFINED ||
        geom.config == TwoViewGeometry::DEGENERATE ||
        geom.config == TwoViewGeometry::WATERMARK ||
        geom.config == TwoViewGeometry::MULTIPLE) {
      continue;
    }

    // 过滤5：内点数量足够
    const size_t num_inliers = geom.inlier_matches.size();
    if (num_inliers < options.min_loop_inliers) {
      continue;
    }

    // 过滤6：内点比例足够高
    const FeatureMatches matches = database->ReadMatches(i, j);
    if (matches.empty()) {
      continue;
    }
    const double inlier_ratio =
        static_cast<double>(num_inliers) /
        static_cast<double>(matches.size());
    if (inlier_ratio < options.min_loop_inlier_ratio) {
      continue;
    }

    // 过滤7：三角化角度足够大
    if (geom.tri_angle > 0 && geom.tri_angle < min_tri_angle) {
      continue;
    }

    // 过滤8：通过几何误差检验
    const Image& image_i = reconstruction.Image(i);
    const Image& image_j = reconstruction.Image(j);
    const Camera& camera_i = reconstruction.Camera(image_i.CameraId());
    const Camera& camera_j = reconstruction.Camera(image_j.CameraId());
    if (!PassesGeometricErrorFilter(geom, image_i, image_j, camera_i, camera_j,
                                    database, &points_cache,
                                    options.max_loop_geom_error_px)) {
      continue;
    }

    // 获取相对位姿：优先使用已有位姿，否则重新估计
    Eigen::Quaterniond q_ij;
    Eigen::Vector3d t_ij_dir;
    if (geom.qvec.squaredNorm() > 1e-12 && geom.tvec.squaredNorm() > 1e-12) {
      const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(geom.qvec);
      q_ij = Eigen::Quaterniond(normalized_qvec(0), normalized_qvec(1),
                                normalized_qvec(2), normalized_qvec(3));
      t_ij_dir = geom.tvec.normalized();
    } else {
      if (!EstimateRelativePoseFromInliers(geom, image_i, image_j, camera_i,
                                           camera_j, database, &points_cache,
                                           &q_ij, &t_ij_dir)) {
        continue;
      }
    }

    // 过滤9：每个图像的回环边数量不超过上限
    if (loop_degree[i] >= options.max_loop_edges_per_image ||
        loop_degree[j] >= options.max_loop_edges_per_image) {
      continue;
    }

    // 构建回环边
    PoseGraphEdge edge;
    edge.type = EdgeType::kLoop;
    edge.image_id1 = i;
    edge.image_id2 = j;
    edge.q_ij = q_ij;
    double loop_scale = 0.0;
    Eigen::Quaterniond q_ij_sim3;
    Eigen::Vector3d t_ij_sim3;
    if (EstimateLoopScaleFromPointPairsSim3(reconstruction, geom, image_i,
                                            image_j, q_ij, t_ij_dir,
                                            &loop_scale, q_ij_sim3, t_ij_sim3)) {
      // 只使用尺度信息调整平移向量
      // edge.t_ij = t_ij_dir * loop_scale;
      // edge.translation_is_unit = false;
      // 直接使用sim3估计出来的旋转和平移
      edge.q_ij = q_ij_sim3;
      edge.t_ij = t_ij_sim3;
      edge.translation_is_unit = false;
    } else {
      edge.t_ij = t_ij_dir;
      edge.translation_is_unit = true;
    }
    edge.rot_weight = options.defaults.loop_rot_weight;
    edge.trans_weight = options.defaults.loop_trans_weight;

    edges.push_back(edge);
    occupied_pairs->insert(pair_id);  // 标记该对已使用
    loop_degree[i] += 1;
    loop_degree[j] += 1;
  }

  return edges;
}

/**
 * [功能描述]：将位姿图边作为残差块添加到Ceres优化问题中
 * @param edges：位姿图边列表
 * @param pose_params：图像ID到位姿参数指针的映射
 * @param problem：Ceres优化问题
 * @param num_loop_edges：输出参数，添加的回环边数量（可为nullptr）
 * @param num_odom_edges：输出参数，添加的里程计边数量（可为nullptr）
 */
void AddEdgesToProblem(const std::vector<PoseGraphEdge>& edges,
                       const std::unordered_map<image_t, double*>& pose_params,
                       ceres::Problem* problem,
                       size_t* num_loop_edges,
                       size_t* num_odom_edges) {
  // 初始化计数器
  if (num_loop_edges) {
    *num_loop_edges = 0;
  }
  if (num_odom_edges) {
    *num_odom_edges = 0;
  }

  for (const auto& edge : edges) {
    // 查找两端图像的位姿参数
    auto it1 = pose_params.find(edge.image_id1);
    auto it2 = pose_params.find(edge.image_id2);
    if (it1 == pose_params.end() || it2 == pose_params.end()) {
      continue;  // 位姿参数不存在则跳过
    }

    const double rot_weight = edge.rot_weight;
    const double trans_weight = edge.trans_weight;

    // 根据平移类型选择代价函数
    ceres::CostFunction* cost = nullptr;
    if (edge.translation_is_unit) {
      // 单位平移向量（仅约束方向）
      cost = SE3RelativePoseCost::Create(edge.q_ij, edge.t_ij, rot_weight,
                                         trans_weight);
    } else {
      // 完整平移向量（约束方向和尺度）
      cost = SE3RelativePoseCostFull::Create(edge.q_ij, edge.t_ij, rot_weight,
                                             trans_weight);
    }

    // 回环边使用Huber鲁棒核函数，里程计边不使用
    ceres::LossFunction* loss = nullptr;
    if (edge.type == EdgeType::kLoop) {
      loss = new ceres::HuberLoss(1.0);
    }

    // 添加残差块到优化问题
    problem->AddResidualBlock(cost, loss, it1->second, it2->second);

    // 统计边数量
    if (edge.type == EdgeType::kLoop) {
      if (num_loop_edges) {
        *num_loop_edges += 1;
      }
    } else {
      if (num_odom_edges) {
        *num_odom_edges += 1;
      }
    }
  }
}

bool NearlyEqual(const double a, const double b, const double tol) {
  return std::abs(a - b) <= tol;
}

/**
 * [功能描述]：IO自测试函数，验证边的读写功能是否正确
 * @return 测试通过返回true，否则返回false
 */
bool RunIoSelfTest() {
  EdgeDefaults defaults;

  // 构建测试用例1：里程计边（完整平移）
  PoseGraphEdge e1;
  e1.type = EdgeType::kOdom;
  e1.image_id1 = 1;
  e1.image_id2 = 2;
  e1.q_ij = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);  // 单位四元数
  e1.t_ij = Eigen::Vector3d(1.0, 2.0, 3.0);
  e1.translation_is_unit = false;
  e1.rot_weight = 1.5;
  e1.trans_weight = 2.5;

  // 构建测试用例2：回环边（单位平移方向）
  PoseGraphEdge e2;
  e2.type = EdgeType::kLoop;
  e2.image_id1 = 10;
  e2.image_id2 = 42;
  e2.q_ij = Eigen::Quaterniond(0.9238795, 0.0, 0.3826834, 0.0);  // 绕Y轴旋转45度
  e2.t_ij = Eigen::Vector3d(0.0, 1.0, 0.0);
  e2.translation_is_unit = true;
  e2.rot_weight = 5.0;
  e2.trans_weight = 2.0;

  // 写入到字符串流
  std::vector<PoseGraphEdge> edges{e1, e2};
  std::stringstream ss;
  WritePoseGraphEdges(ss, edges);

  // 从字符串流读回
  const std::string serialized = ss.str();
  std::istringstream iss(serialized);
  std::vector<PoseGraphEdge> parsed;
  std::string error;
  if (!ReadPoseGraphEdges(iss, defaults, &parsed, &error)) {
    std::cout << "Self-test failed: read error: " << error << "\n";
    return false;
  }

  // 检查数量一致
  if (parsed.size() != edges.size()) {
    std::cout << "Self-test failed: size mismatch\n";
    return false;
  }

  // 逐条检查边的各项属性
  for (size_t i = 0; i < edges.size(); ++i) {
    const auto& a = edges[i];
    const auto& b = parsed[i];
    // 检查类型和ID
    if (a.type != b.type || a.image_id1 != b.image_id1 ||
        a.image_id2 != b.image_id2) {
      std::cout << "Self-test failed: id/type mismatch\n";
      return false;
    }
    // 检查四元数
    if (!NearlyEqual(a.q_ij.w(), b.q_ij.w(), 1e-6) ||
        !NearlyEqual(a.q_ij.x(), b.q_ij.x(), 1e-6) ||
        !NearlyEqual(a.q_ij.y(), b.q_ij.y(), 1e-6) ||
        !NearlyEqual(a.q_ij.z(), b.q_ij.z(), 1e-6)) {
      std::cout << "Self-test failed: quaternion mismatch\n";
      return false;
    }
    // 检查平移向量
    if (!NearlyEqual(a.t_ij.x(), b.t_ij.x(), 1e-9) ||
        !NearlyEqual(a.t_ij.y(), b.t_ij.y(), 1e-9) ||
        !NearlyEqual(a.t_ij.z(), b.t_ij.z(), 1e-9)) {
      std::cout << "Self-test failed: translation mismatch\n";
      return false;
    }
    // 检查权重参数
    if (a.translation_is_unit != b.translation_is_unit ||
        !NearlyEqual(a.rot_weight, b.rot_weight, 1e-9) ||
        !NearlyEqual(a.trans_weight, b.trans_weight, 1e-9)) {
      std::cout << "Self-test failed: weights mismatch\n";
      return false;
    }
  }

  // 测试旧格式（无类型前缀，默认为loop）的解析
  const std::string legacy_line =
      "1 9 1 0 0 0 0.1 0 0";
  PoseGraphEdge legacy_edge;
  if (!ParseEdgeLine(legacy_line, defaults, &legacy_edge)) {
    std::cout << "Self-test failed: legacy parse\n";
    return false;
  }
  if (legacy_edge.type != EdgeType::kLoop ||
      legacy_edge.image_id1 != 1 || legacy_edge.image_id2 != 9) {
    std::cout << "Self-test failed: legacy edge content\n";
    return false;
  }

  std::cout << "Self-test passed.\n";
  return true;
}

void PrintUsage() {
  std::cout
      << "Usage: test_vpgo_pipeline [--self-test] "
         "sparse_path database_path output_path\n"
         "Options:\n"
         "  --edge-input <path>   Read pose-graph edges (manual recall).\n"
         "  --edge-output <path>  Write pose-graph edges.\n"
         "  --manual-loop <path>  Add manual loop edges file.\n"
         "  --skip-odom           Disable odometry edges.\n"
         "  --skip-loop           Disable auto loop edges.\n"
         "  --eval-threshold <px> Fail if mean reprojection error exceeds.\n"
         "  --self-test           Run IO self-test and exit.\n";
}

/**
 * [功能描述]：解析命令行参数
 * @param argc：命令行参数个数
 * @param argv：命令行参数数组
 * @param options：输出参数，存储解析后的配置选项
 * @return 解析成功返回true，否则返回false
 */
bool ParseArgs(int argc, char** argv, PipelineOptions* options) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    // 可选参数：运行自测试
    if (arg == "--self-test") {
      options->run_self_test = true;
      continue;
    }
    // 可选参数：跳过里程计边构建
    if (arg == "--skip-odom") {
      options->build_odom_edges = false;
      continue;
    }
    // 可选参数：跳过回环边构建
    if (arg == "--skip-loop") {
      options->build_loop_edges = false;
      continue;
    }
    // 可选参数：边输入文件路径
    if (arg == "--edge-input" && i + 1 < argc) {
      options->edge_input_path = argv[++i];
      continue;
    }
    // 可选参数：边输出文件路径
    if (arg == "--edge-output" && i + 1 < argc) {
      options->edge_output_path = argv[++i];
      continue;
    }
    // 可选参数：手动回环边文件路径
    if (arg == "--manual-loop" && i + 1 < argc) {
      options->manual_loop_path = argv[++i];
      continue;
    }
    // 可选参数：评估阈值（像素）
    if (arg == "--eval-threshold" && i + 1 < argc) {
      options->eval_threshold_px = std::stod(argv[++i]);
      continue;
    }
    // 未知选项报错
    if (arg.rfind("--", 0) == 0) {
      std::cout << "Unknown option: " << arg << "\n";
      return false;
    }
    // 位置参数：依次为 sparse_path, database_path, output_path
    if (options->sparse_path.empty()) {
      options->sparse_path = arg;
    } else if (options->database_path.empty()) {
      options->database_path = arg;
    } else if (options->output_path.empty()) {
      options->output_path = arg;
    } else {
      std::cout << "Unexpected argument: " << arg << "\n";
      return false;
    }
  }

  // 自测试模式不需要其他参数
  if (options->run_self_test) {
    return true;
  }

  // 检查必需的位置参数
  if (options->sparse_path.empty() || options->database_path.empty() ||
      options->output_path.empty()) {
    return false;
  }

  // 设置默认的边输出路径
  if (options->edge_output_path.empty()) {
    options->edge_output_path =
        JoinPaths(options->output_path, "pose_graph_edges.txt");
  }

  // 自动检测默认的手动回环边文件
  // if (options->manual_loop_path.empty()) {
  //   const std::string default_manual =
  //       JoinPaths(options->output_path, "loop_edges_gt.txt");
  //   if (ExistsFile(default_manual)) {
  //     options->manual_loop_path = default_manual;
  //   }
  // }

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  // ========== 1. 解析命令行参数 ==========
  PipelineOptions options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintUsage();
    return -1;
  }

  // 自测试模式
  if (options.run_self_test) {
    return RunIoSelfTest() ? 0 : 1;
  }

  // ========== 2. 加载重建数据 ==========
  CreateDirIfNotExists(options.output_path);

  Reconstruction reconstruction;
  reconstruction.Read(options.sparse_path);

  if (reconstruction.RegImageIds().empty()) {
    std::cout << "No registered images found.\n";
    return -1;
  }

  // 记录优化前的重投影误差
  const double pre_error = reconstruction.ComputeMeanReprojectionError();
  std::cout << "Mean reprojection error before PGO: " << pre_error << " px\n";

  Database database(options.database_path);

  // ========== 3. 初始化Ceres优化问题 ==========
  ceres::Problem problem;
  std::unordered_map<image_t, double*> pose_params;
  pose_params.reserve(reconstruction.RegImageIds().size());

  // 为每个图像创建位姿参数块 [qw, qx, qy, qz, tx, ty, tz]
  for (const auto image_id : reconstruction.RegImageIds()) {
    const Image& image = reconstruction.Image(image_id);
    double* pose = new double[7];
    const auto Q = image.Qvec();
    const auto T = image.Tvec();
    pose[0] = Q[0];  // qw
    pose[1] = Q[1];  // qx
    pose[2] = Q[2];  // qy
    pose[3] = Q[3];  // qz
    pose[4] = T[0];  // tx
    pose[5] = T[1];  // ty
    pose[6] = T[2];  // tz

    pose_params[image.ImageId()] = pose;
    problem.AddParameterBlock(pose, 7, new SE3Manifold());
  }

  // ========== 4. 构建位姿图边 ==========
  std::unordered_set<image_pair_t> occupied_pairs;  // 记录已使用的图像对
  occupied_pairs.reserve(reconstruction.RegImageIds().size());

  std::vector<PoseGraphEdge> edges;
  if (!options.edge_input_path.empty()) {
    // 从文件读取边
    std::string error;
    if (!ReadPoseGraphEdgesFromFile(options.edge_input_path, options.defaults,
                                    &edges, &error)) {
      std::cout << "Failed to read edge input: " << error << "\n";
      return -1;
    }
  } else {
    // 自动构建里程计边
    if (options.build_odom_edges) {
      // auto odom_edges = BuildOdometryEdges(reconstruction, options.defaults);
      auto odom_edges = BuildOdometryEdges(reconstruction, &database, options.defaults);
      for (const auto& edge : odom_edges) {
        const image_pair_t pair_id =
            Database::ImagePairToPairId(edge.image_id1, edge.image_id2);
        occupied_pairs.insert(pair_id);
      }
      edges.insert(edges.end(), odom_edges.begin(), odom_edges.end());
    }

    // 自动构建回环边
    if (options.build_loop_edges) {
      auto loop_edges =
          BuildLoopEdges(reconstruction, &database, options, &occupied_pairs);
      edges.insert(edges.end(), loop_edges.begin(), loop_edges.end());
    }

    // 添加手动指定的回环边
    if (!options.manual_loop_path.empty()) {
      std::vector<PoseGraphEdge> manual_edges;
      std::string error;
      if (ReadPoseGraphEdgesFromFile(options.manual_loop_path, options.defaults,
                                     &manual_edges, &error)) {
        // 点的缓存,避免重复读取
        std::unordered_map<image_t, std::vector<Eigen::Vector2d>> points_cache;
        points_cache.reserve(reconstruction.RegImageIds().size());

        size_t added_manual = 0;
        for (auto& edge : manual_edges) {
          if (edge.type == EdgeType::kOdom) {
            continue;  // 跳过里程计类型
          }
          edge.type = EdgeType::kLoop;
          if (!reconstruction.ExistsImage(edge.image_id1) ||
              !reconstruction.ExistsImage(edge.image_id2)) {
            continue;
          }

          const image_pair_t pair_id =
              Database::ImagePairToPairId(edge.image_id1, edge.image_id2);
          if (occupied_pairs.count(pair_id) > 0) {
            continue;  // 跳过已存在的边
          }

          // 获取对应的图像和相机
          const Image& image_i = reconstruction.Image(edge.image_id1);
          const Image& image_j = reconstruction.Image(edge.image_id2);
          const Camera& camera_i = reconstruction.Camera(image_i.CameraId());
          const Camera& camera_j = reconstruction.Camera(image_j.CameraId());

          // 获取图像对的双视图几何
          TwoViewGeometry geom =
              database.ReadTwoViewGeometry(edge.image_id1, edge.image_id2);

          // 获取图像对的相对位姿
          Eigen::Quaterniond q_ij = edge.q_ij;
          Eigen::Vector3d t_ij_dir = edge.t_ij;

          // 如果边中没有位姿信息，则尝试从几何中恢复
          if (q_ij.squaredNorm() < 1e-12 ||
              t_ij_dir.squaredNorm() < 1e-12) {
            if (geom.qvec.squaredNorm() > 1e-12 &&
                geom.tvec.squaredNorm() > 1e-12) {
              const Eigen::Vector4d normalized_qvec =
                  NormalizeQuaternion(geom.qvec);
              q_ij = Eigen::Quaterniond(normalized_qvec(0), normalized_qvec(1),
                                        normalized_qvec(2), normalized_qvec(3));
              t_ij_dir = geom.tvec.normalized();
            } else {
              Eigen::Quaterniond q_ij_est;
              Eigen::Vector3d t_ij_dir_est;
              if (EstimateRelativePoseFromInliers(
                      geom, image_i, image_j, camera_i, camera_j, &database,
                      &points_cache, &q_ij_est, &t_ij_dir_est)) {
                q_ij = q_ij_est;
                t_ij_dir = t_ij_dir_est;
              }
            }
          }

          // 如果是归一化的边,则尝试估计尺度
          if (edge.translation_is_unit &&
              t_ij_dir.squaredNorm() > 1e-12 &&
              q_ij.squaredNorm() > 1e-12) {
            double loop_scale = 0.0;
            Eigen::Quaterniond q_ij_sim3;
            Eigen::Vector3d t_ij_sim3;
            if (EstimateLoopScaleFromPointPairsSim3(
                    reconstruction, geom, image_i, image_j, q_ij, t_ij_dir,
                    &loop_scale, q_ij_sim3, t_ij_sim3)) {
              // 只使用尺度信息调整平移向量
              // edge.t_ij = t_ij_dir * loop_scale;
              // edge.translation_is_unit = false;
              // 直接使用sim3估计出来的旋转和平移
              edge.q_ij = q_ij_sim3;
              edge.t_ij = t_ij_sim3;
              edge.translation_is_unit = false;
            } else if (edge.t_ij.squaredNorm() > 1e-12) {
              edge.t_ij.normalize();
              edge.translation_is_unit = true;
            }
          }

          if (q_ij.squaredNorm() > 1e-12) {
            edge.q_ij = q_ij;
          }
          occupied_pairs.insert(pair_id);
          edges.push_back(edge);
          added_manual += 1;
        }
        std::cout << "Added manual loop edges: " << added_manual
                  << "\n";
      }
    }

    // 保存构建的边到文件
    if (!options.edge_output_path.empty()) {
      std::string error;
      if (!WritePoseGraphEdgesToFile(options.edge_output_path, edges, &error)) {
        std::cout << "Failed to write edge output: " << error << "\n";
      } else {
        std::cout << "Wrote edge file: " << options.edge_output_path << "\n";
      }
    }
  }

  // ========== 5. 添加边到优化问题 ==========
  size_t num_loop = 0;
  size_t num_odom = 0;
  AddEdgesToProblem(edges, pose_params, &problem, &num_loop, &num_odom);

  // 固定第一个图像的位姿（消除规范自由度）
  const image_t root_id = reconstruction.RegImageIds().front();
  problem.SetParameterBlockConstant(pose_params[root_id]);

  std::cout << "PGO edges: odom=" << num_odom << " loop=" << num_loop << "\n";

  // ========== 6. 运行Ceres求解器 ==========
  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  solver_options.max_num_iterations = 200;
  solver_options.function_tolerance = 1e-6;
  solver_options.gradient_tolerance = 1e-10;
  solver_options.parameter_tolerance = 1e-8;
  solver_options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";

  // ========== 7. 更新重建中的位姿 ==========
  for (const auto image_id : reconstruction.RegImageIds()) {
    Image& image = reconstruction.Image(image_id);
    double* pose = pose_params[image_id];
    Eigen::Quaterniond q(pose[0], pose[1], pose[2], pose[3]);
    Eigen::Vector3d t(pose[4], pose[5], pose[6]);
    image.SetQvec(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
    image.SetTvec(t);
  }

  // // 全局BA优化
  // BundleAdjustmentOptions ba_options;
  // ba_options.refine_focal_length = false;
  // ba_options.refine_principal_point = false;

  // colmap::BundleAdjustmentConfig ba_config;
  // // 把所有已注册的 image 加进去
  // for (const auto image_id : reconstruction.RegImageIds()) {
  //   ba_config.AddImage(image_id);
  // }

  // colmap::BundleAdjuster bundle_adjuster(ba_options, ba_config);
  // bundle_adjuster.Solve(&reconstruction);

  // ========== 8. 评估并保存结果 ==========
  const double post_error = reconstruction.ComputeMeanReprojectionError();
  std::cout << "Mean reprojection error after PGO: " << post_error << " px\n";

  // 检查是否超过误差阈值
  if (options.eval_threshold_px > 0.0 && post_error > options.eval_threshold_px) {
    std::cout << "ERROR: reprojection error exceeds threshold ("
              << options.eval_threshold_px << " px)\n";
    return -1;
  }

  // 保存优化后的重建结果
  reconstruction.Write(options.output_path);
  std::cout << "PGO done. Output: " << options.output_path << "\n";
  return 0;
}
