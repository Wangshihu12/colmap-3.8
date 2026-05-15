#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#ifdef BOOL
#undef BOOL
#endif
#ifdef DWORD
#undef DWORD
#endif
#include <psapi.h>
#endif
#include "base/pose.h"
#include "base/reconstruction.h"
#include "controllers/bundle_adjustment.h"
#include "controllers/incremental_mapper.h"

#include "estimators/homography_matrix.h"
#include "estimators/pose.h"
#include "estimators/utils.h"
#include "feature/utils.h"
#include "util/misc.h"
#include "util/math.h"
#include "util/string.h"
// #include "xgbase/utils.h"

using namespace colmap;

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#ifdef _WIN32
// #include <json/json.h>
#else
// #include "json.h"
#endif

// #include "lixel/lidar_post_process.h"

#ifdef AUTHENTICATE_KEY
#include <md5/include/XGrids/Base/MD5.h>
#endif

#include "base/cost_functions.h"
// #include "base/lcc_errocde.h"
// #include "base/memory_instruction.hpp"
// #include "xgbase/lcc_config.h"
// #include "xgbase/lcc_config_device.h"

#include <ceres/ceres.h>
#if __has_include(<ceres/local_parameterization.h>)
#include <ceres/local_parameterization.h>
#endif

// using namespace xgrids3d;
using namespace colmap;

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
    // 位姿参数： [qw qx qy qz tx ty tz]

    Eigen::Quaternion<T> q_i(pose_i[0], pose_i[1], pose_i[2], pose_i[3]);
    Eigen::Matrix<T, 3, 1> t_i(pose_i[4], pose_i[5], pose_i[6]);

    Eigen::Quaternion<T> q_j(pose_j[0], pose_j[1], pose_j[2], pose_j[3]);
    Eigen::Matrix<T, 3, 1> t_j(pose_j[4], pose_j[5], pose_j[6]);

    Eigen::Quaternion<T> q_ij = q_ij_.cast<T>();
    Eigen::Matrix<T, 3, 1> t_ij_obs = t_ij_unit_.cast<T>();

    // 旋转残差：预测相对旋转与观测相对旋转的差
    Eigen::Quaternion<T> q_ij_pred = q_j * q_i.conjugate();
    Eigen::Quaternion<T> dq = q_ij.conjugate() * q_ij_pred;

    Eigen::Matrix<T, 3, 1> r_rot;
    r_rot << T(2) * dq.x(), T(2) * dq.y(), T(2) * dq.z();

    // 平移方向残差：只约束方向，不约束尺度
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
    // 位姿参数： [qw qx qy qz tx ty tz]

    Eigen::Quaternion<T> q_i(pose_i[0], pose_i[1], pose_i[2], pose_i[3]);
    Eigen::Matrix<T, 3, 1> t_i(pose_i[4], pose_i[5], pose_i[6]);

    Eigen::Quaternion<T> q_j(pose_j[0], pose_j[1], pose_j[2], pose_j[3]);
    Eigen::Matrix<T, 3, 1> t_j(pose_j[4], pose_j[5], pose_j[6]);

    Eigen::Quaternion<T> q_ij = q_ij_.cast<T>();
    Eigen::Matrix<T, 3, 1> t_ij_obs = t_ij_.cast<T>();

    // 旋转残差：预测相对旋转与观测相对旋转的差
    Eigen::Quaternion<T> q_ij_pred = q_j * q_i.conjugate();
    Eigen::Quaternion<T> dq = q_ij.conjugate() * q_ij_pred;

    Eigen::Matrix<T, 3, 1> r_rot;
    r_rot << T(2) * dq.x(), T(2) * dq.y(), T(2) * dq.z();

    // 平移残差：直接约束完整平移，保留尺度
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

struct Sim3RelativePoseCost {
  Sim3RelativePoseCost(const Eigen::Quaterniond& q_ij,
                       const Eigen::Vector3d& t_ij_dir,
                       const double t_ij_scale, const double rot_weight,
                       const double trans_weight)
      : q_ij_(q_ij),
        t_ij_dir_(t_ij_dir),
        t_ij_scale_(t_ij_scale),
        rot_weight_(rot_weight),
        trans_weight_(trans_weight) {}

  template <typename T>
  bool operator()(const T* const pose_i, const T* const pose_j,
                  T* residuals) const {
    // 位姿参数： [qw qx qy qz tx ty tz s]
    Eigen::Quaternion<T> q_i(pose_i[0], pose_i[1], pose_i[2], pose_i[3]);
    Eigen::Matrix<T, 3, 1> t_i(pose_i[4], pose_i[5], pose_i[6]);
    const T s_i = pose_i[7];

    Eigen::Quaternion<T> q_j(pose_j[0], pose_j[1], pose_j[2], pose_j[3]);
    Eigen::Matrix<T, 3, 1> t_j(pose_j[4], pose_j[5], pose_j[6]);
    const T s_j = pose_j[7];

    const Eigen::Quaternion<T> q_ij = q_ij_.cast<T>();
    const Eigen::Matrix<T, 3, 1> t_ij_dir = t_ij_dir_.cast<T>();
    const T t_ij_scale = T(t_ij_scale_);
    const Eigen::Matrix<T, 3, 1> t_ij_obs = t_ij_dir * t_ij_scale;

    // 预测相对旋转
    Eigen::Quaternion<T> q_ij_pred = q_j * q_i.conjugate();
    Eigen::Quaternion<T> dq = q_ij.conjugate() * q_ij_pred;

    Eigen::Matrix<T, 3, 1> r_rot;
    r_rot << T(2) * dq.x(), T(2) * dq.y(), T(2) * dq.z();

    // 预测相对平移（包含尺度比例）
    const T s_ij = s_j / s_i;
    Eigen::Matrix<T, 3, 1> t_pred = t_j - s_ij * (q_ij_pred * t_i);
    // 直接约束完整平移向量（由方向 + 尺度构成）
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
                                     const Eigen::Vector3d& t_ij_dir,
                                     const double t_ij_scale,
                                     const double rot_weight,
                                     const double trans_weight) {
    return new ceres::AutoDiffCostFunction<Sim3RelativePoseCost, 6, 8, 8>(
        new Sim3RelativePoseCost(q_ij, t_ij_dir, t_ij_scale, rot_weight,
                                 trans_weight));
  }

 private:
  Eigen::Quaterniond q_ij_;
  Eigen::Vector3d t_ij_dir_;
  double t_ij_scale_;
  double rot_weight_;
  double trans_weight_;
};

class Sim3Manifold : public ceres::Manifold {
 public:
  int AmbientSize() const override { return 8; }
  int TangentSize() const override { return 7; }

  // x_plus_delta = x ⊕ delta（李代数增量更新）
  bool Plus(const double* x, const double* delta,
            double* x_plus_delta) const override {
    // 位姿参数： [qw qx qy qz tx ty tz s]
    Eigen::Quaterniond q(x[0], x[1], x[2], x[3]);
    Eigen::Vector3d t(x[4], x[5], x[6]);
    const double s = x[7];

    Eigen::Vector3d omega(delta[0], delta[1], delta[2]);
    Eigen::Vector3d upsilon(delta[3], delta[4], delta[5]);
    const double sigma = delta[6];

    double theta = omega.norm();
    Eigen::Quaterniond dq = Eigen::Quaterniond::Identity();
    if (theta > 1e-12) {
      dq = Eigen::Quaterniond(Eigen::AngleAxisd(theta, omega / theta));
    }

    Eigen::Quaterniond q_new = (dq * q).normalized();
    Eigen::Vector3d t_new = t + upsilon;
    const double s_new = s * std::exp(sigma);

    x_plus_delta[0] = q_new.w();
    x_plus_delta[1] = q_new.x();
    x_plus_delta[2] = q_new.y();
    x_plus_delta[3] = q_new.z();
    x_plus_delta[4] = t_new.x();
    x_plus_delta[5] = t_new.y();
    x_plus_delta[6] = t_new.z();
    x_plus_delta[7] = s_new;

    return true;
  }

  bool PlusJacobian(const double*, double* jacobian) const override {
    Eigen::Map<Eigen::Matrix<double, 8, 7, Eigen::RowMajor>> J(jacobian);
    J.setZero();
    J.block<6, 6>(1, 0).setIdentity();
    J(7, 6) = 1.0;
    return true;
  }

  // y_minus_x = y ⊖ x（局部坐标差）
  bool Minus(const double* y, const double* x,
             double* y_minus_x) const override {
    Eigen::Quaterniond qx(x[0], x[1], x[2], x[3]);
    Eigen::Quaterniond qy(y[0], y[1], y[2], y[3]);
    Eigen::Vector3d tx(x[4], x[5], x[6]);
    Eigen::Vector3d ty(y[4], y[5], y[6]);
    const double sx = x[7];
    const double sy = y[7];

    Eigen::Quaterniond dq = qy * qx.conjugate();
    Eigen::AngleAxisd aa(dq);

    Eigen::Vector3d omega = aa.axis() * aa.angle();
    Eigen::Vector3d upsilon = ty - tx;
    const double sigma = std::log(sy / sx);

    y_minus_x[0] = omega.x();
    y_minus_x[1] = omega.y();
    y_minus_x[2] = omega.z();
    y_minus_x[3] = upsilon.x();
    y_minus_x[4] = upsilon.y();
    y_minus_x[5] = upsilon.z();
    y_minus_x[6] = sigma;

    return true;
  }

  bool MinusJacobian(const double*, double* jacobian) const override {
    Eigen::Map<Eigen::Matrix<double, 7, 8, Eigen::RowMajor>> J(jacobian);
    J.setZero();
    J.block<6, 6>(0, 1).setIdentity();
    J(6, 7) = 1.0;
    return true;
  }
};

bool ParseLoopEdgeMeasurement(const std::string& line, image_t* image_id1,
                              image_t* image_id2, Eigen::Quaterniond* q_ij,
                              Eigen::Vector3d* t_ij) {
  std::string content = line;
  const auto comment_pos = content.find('#');
  if (comment_pos != std::string::npos) {
    content = content.substr(0, comment_pos);
  }
  StringTrim(&content);
  if (content.empty()) {
    return false;
  }

  std::istringstream iss(content);
  int64_t id1 = -1;
  int64_t id2 = -1;
  double qw = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  double tx = 0.0;
  double ty = 0.0;
  double tz = 0.0;
  if (!(iss >> id1 >> id2 >> qw >> qx >> qy >> qz >> tx >> ty >> tz)) {
    return false;
  }

  *image_id1 = static_cast<image_t>(id1);
  *image_id2 = static_cast<image_t>(id2);
  *q_ij = Eigen::Quaterniond(qw, qx, qy, qz);
  q_ij->normalize();
  *t_ij = Eigen::Vector3d(tx, ty, tz);
  return true;
}

// 从数据库读取关键点，并缓存为二维点坐标
const std::vector<Eigen::Vector2d>& GetOrLoadImagePoints(
    const image_t image_id, Database* database,
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>>* points_cache) {
  auto it = points_cache->find(image_id);
  if (it != points_cache->end()) {
    return it->second;
  }

  // 从数据库读取关键点并转换为点坐标（像素坐标）
  FeatureKeypoints keypoints = database->ReadKeypoints(image_id);
  auto points = FeatureKeypointsToPointsVector(keypoints);
  auto emplace_result =
      points_cache->emplace(image_id, std::move(points));
  return emplace_result.first->second;
}

// 基于当前两视图模型计算内点的几何误差中位数，做质量过滤
bool PassesGeometricErrorFilter(
    const TwoViewGeometry& geom, const Image& image_i, const Image& image_j,
    const Camera& camera_i, const Camera& camera_j, Database* database,
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>>* points_cache,
    const double max_error_px) {
  if (geom.inlier_matches.empty()) {
    return false;
  }

  const auto& points1_px =
      GetOrLoadImagePoints(image_i.ImageId(), database, points_cache);
  const auto& points2_px =
      GetOrLoadImagePoints(image_j.ImageId(), database, points_cache);
  if (points1_px.empty() || points2_px.empty()) {
    return false;
  }

  std::vector<Eigen::Vector2d> inlier_points1_px;
  std::vector<Eigen::Vector2d> inlier_points2_px;
  inlier_points1_px.reserve(geom.inlier_matches.size());
  inlier_points2_px.reserve(geom.inlier_matches.size());
  for (const auto& match : geom.inlier_matches) {
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

  if (geom.config == TwoViewGeometry::CALIBRATED &&
      geom.E.squaredNorm() > 1e-12) {
    // 标定情形：使用归一化坐标与E计算Sampson误差
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
    ComputeSquaredSampsonError(inlier_points1_norm, inlier_points2_norm, geom.E,
                               &residuals);
    const double max_error_norm =
        (camera_i.ImageToWorldThreshold(max_error_px) +
         camera_j.ImageToWorldThreshold(max_error_px)) /
        2.0;
    return Median(residuals) <= max_error_norm * max_error_norm;
  }

  if (geom.F.squaredNorm() > 1e-12 &&
      (geom.config == TwoViewGeometry::UNCALIBRATED ||
       geom.config == TwoViewGeometry::CALIBRATED)) {
    // 非标定/退化时退回使用F与像素坐标
    ComputeSquaredSampsonError(inlier_points1_px, inlier_points2_px, geom.F,
                               &residuals);
    return Median(residuals) <= max_error_px * max_error_px;
  }

  if (geom.H.squaredNorm() > 1e-12 &&
      (geom.config == TwoViewGeometry::PLANAR ||
       geom.config == TwoViewGeometry::PANORAMIC ||
       geom.config == TwoViewGeometry::PLANAR_OR_PANORAMIC)) {
    // 平面/纯旋转情形：使用H的重投影误差
    HomographyMatrixEstimator::Residuals(inlier_points1_px, inlier_points2_px,
                                         geom.H, &residuals);
    return Median(residuals) <= max_error_px * max_error_px;
  }

  return false;
}

// 当数据库中未存储qvec/tvec时，基于匹配内点估计相对位姿
bool EstimateRelativePoseFromInliers(
    const TwoViewGeometry& geom, const Image& image_i,
    const Image& image_j, const Camera& camera_i, const Camera& camera_j,
    Database* database,
    std::unordered_map<image_t, std::vector<Eigen::Vector2d>>* points_cache,
    Eigen::Quaterniond* q_ij, Eigen::Vector3d* t_ij_dir) {
  if (geom.inlier_matches.empty()) {
    return false;
  }

  // EstimateRelativePose 只对特定配置有效，避免退化配置带来错误解
  if (geom.config != TwoViewGeometry::CALIBRATED &&
      geom.config != TwoViewGeometry::UNCALIBRATED &&
      geom.config != TwoViewGeometry::PLANAR &&
      geom.config != TwoViewGeometry::PANORAMIC &&
      geom.config != TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    return false;
  }

  // 获取图像关键点（像素坐标）
  const auto& points1 =
      GetOrLoadImagePoints(image_i.ImageId(), database, points_cache);
  const auto& points2 =
      GetOrLoadImagePoints(image_j.ImageId(), database, points_cache);
  if (points1.empty() || points2.empty()) {
    return false;
  }

  // 直接复用TwoViewGeometry中的模型，用内点匹配估计相对位姿
  TwoViewGeometry geom_copy = geom;
  if (!geom_copy.EstimateRelativePose(camera_i, points1, camera_j, points2)) {
    return false;
  }

  // 再次过滤
  if (geom_copy.config != TwoViewGeometry::CALIBRATED &&
      geom_copy.config != TwoViewGeometry::UNCALIBRATED &&
      geom_copy.config != TwoViewGeometry::PLANAR &&
      geom_copy.config != TwoViewGeometry::PANORAMIC &&
      geom_copy.config != TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    return false;
  }

  if (geom_copy.qvec.squaredNorm() < 1e-12 ||
      geom_copy.tvec.squaredNorm() < 1e-12) {
    return false;
  }

  const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(geom_copy.qvec);
  *q_ij = Eigen::Quaterniond(normalized_qvec(0), normalized_qvec(1),
                             normalized_qvec(2), normalized_qvec(3));

  *t_ij_dir = geom_copy.tvec.normalized();
  return true;
}

// 0. read reconstruction and database
// 1. build pose graph from sequential edges and loop edges from two view
// geometries
// 2. optimize pose graph
// 3. update reconstruction with optimized poses
// 4. run full bundle adjustment
/**
 * [功能描述]：SE3位姿图优化（Pose Graph Optimization, PGO）测试程序主函数。
 *            该程序读取稀疏重建结果，构建位姿图，通过Ceres优化器进行位姿图优化，
 *            利用序列边（里程计约束）和回环边（回环检测约束）优化所有相机位姿。
 * @param argc：命令行参数数量
 * @param argv：命令行参数数组
 *              argv[1]: sparse_path - 稀疏重建结果路径
 *              argv[2]: database_path - 数据库路径
 *              argv[3]: output_path - 优化结果输出路径
 * @return 程序退出状态码
 */
int main(int argc, char** argv) {
  // 检查命令行参数数量
  if (argc < 4) {
    std::cout << "Usage: run_se3_pgo_then_ba "
              << "sparse_path database_path output_path\n";
    return -1;
  }

  // 解析命令行参数
  std::string sparse_path = argv[1];    // 稀疏重建结果路径
  std::string database_path = argv[2];  // COLMAP数据库路径
  std::string output_path = argv[3];    // 优化后的输出路径

  // 读取稀疏重建结果
  Reconstruction reconstruction;
  reconstruction.Read(sparse_path);

  // 打开数据库（用于读取双视图几何信息）
  Database database(database_path);

  // ----------------------------
  // 构建位姿参数块
  // ----------------------------
  // 创建Ceres优化问题
  ceres::Problem problem;
  // 存储每张图像的位姿参数指针，key为图像ID，value为7维位姿参数
  std::unordered_map<image_t, double*> pose_params;

  // 遍历所有图像，初始化位姿参数
  for (const auto image_id : reconstruction.RegImageIds()) {
    const Image& image = reconstruction.Image(image_id);
    // 分配8维位姿参数：4维四元数 + 3维平移 + 1维尺度
    double* pose = new double[8];

    // 获取图像的旋转四元数和平移向量
    const auto Q = image.Qvec();  // 四元数，形状为(4,)
    const auto T = image.Tvec();  // 平移向量，形状为(3,)

    // 复制四元数参数（Eigen四元数顺序：w, x, y, z）
    pose[0] = Q[0];  // qw
    pose[1] = Q[1];  // qx
    pose[2] = Q[2];  // qy
    pose[3] = Q[3];  // qz
    // 复制平移参数
    pose[4] = T[0];  // tx
    pose[5] = T[1];  // ty
    pose[6] = T[2];  // tz
    // 初始化尺度为1（保持与原重建尺度一致）
    pose[7] = 1.0;

    // 存储位姿参数指针
    pose_params[image.ImageId()] = pose;

    // 将位姿参数块添加到优化问题中
    // 使用Sim3流形保证四元数单位性与尺度正值
    problem.AddParameterBlock(pose, 8, new Sim3Manifold());
  }

  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  database.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);

  // 权重
  double odom_rot_weight = 1.0;
  double odom_trans_weight = 1.0;
  double loop_rot_weight = 5.0;
  double loop_trans_weight = 2.0;

  // ----------------------------
  // 构建序列边
  // ----------------------------
  // 统计边的数量
  int num_edge = 0;  // 序列边（里程计边）数量
  int num_loop = 0;  // 回环边数量

  for (size_t k = 0; k < image_pair_ids.size(); ++k) {
    image_t i;
    image_t j;
    Database::PairIdToImagePair(image_pair_ids[k], &i, &j);
    const TwoViewGeometry& geom = two_view_geometries[k];
    // std::cout << "q_ij: " << geom.qvec.transpose() << "\n";

    if (!reconstruction.ExistsImage(i) || !reconstruction.ExistsImage(j)) {
      continue;
    }

    if (std::abs(static_cast<int>(i) - static_cast<int>(j)) <= 10) {
      // sequential edges   odom edge
      const Image& image_i = reconstruction.Image(i);
      const Image& image_j = reconstruction.Image(j);

      Eigen::Quaterniond q_i(image_i.Qvec()[0], image_i.Qvec()[1],
                             image_i.Qvec()[2], image_i.Qvec()[3]);
      Eigen::Vector3d t_i = image_i.Tvec();

      Eigen::Quaterniond q_j(image_j.Qvec()[0], image_j.Qvec()[1],
                             image_j.Qvec()[2], image_j.Qvec()[3]);
      Eigen::Vector3d t_j = image_j.Tvec();

      Eigen::Quaterniond q_ij = q_j * q_i.conjugate();
      Eigen::Vector3d t_ij = t_j - (q_ij * t_i);
      double t_ij_norm = t_ij.norm();
      Eigen::Vector3d t_ij_dir = t_ij.normalized();
      ceres::CostFunction* cost =
          Sim3RelativePoseCost::Create(q_ij, t_ij_dir, t_ij_norm,
                                       odom_rot_weight,   // rot
                                       odom_trans_weight);  // trans
      problem.AddResidualBlock(cost, nullptr, pose_params[i], pose_params[j]);
      num_edge++;
    }
  }

  // ----------------------------
  // 构建回环边（手动添加）
  // ----------------------------
  const std::string loop_edges_gt_path =
      JoinPaths(output_path, "loop_edges_gt.txt");
  if (ExistsFile(loop_edges_gt_path)) {
    std::ifstream loop_edges_gt_file(loop_edges_gt_path);
    if (!loop_edges_gt_file.is_open()) {
      std::cout << "WARNING: Could not open loop edges file at "
                << loop_edges_gt_path << std::endl;
    } else {
      size_t num_manual_loops = 0;
      std::string line;
      while (std::getline(loop_edges_gt_file, line)) {
        image_t i = kInvalidImageId;
        image_t j = kInvalidImageId;
        Eigen::Quaterniond q_ij;
        Eigen::Vector3d t_ij;
        if (!ParseLoopEdgeMeasurement(line, &i, &j, &q_ij, &t_ij)) {
          continue;
        }

        if (pose_params.find(i) == pose_params.end() ||
            pose_params.find(j) == pose_params.end()) {
          continue;
        }

        Eigen::Vector3d t_ij_dir = t_ij.normalized();

        double t_ij_scale = t_ij.norm();

        ceres::CostFunction* cost = Sim3RelativePoseCost::Create(
            q_ij, t_ij_dir, t_ij_scale, loop_rot_weight, loop_trans_weight);
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                                 pose_params[i], pose_params[j]);

        num_loop++;
      }
    }
  }

  // ----------------------------
  // 固定锚点（gauge fixing）
  // ----------------------------
  // 固定第一张图像的位姿以消除位姿图的全局自由度（6 DoF）
  // 否则整个位姿图可以任意平移和旋转
  image_t root_id = reconstruction.Images().begin()->first;
  problem.SetParameterBlockConstant(pose_params[root_id]);

  std::cout << "num_edge: " << num_edge << ", num_loop: " << num_loop << "\n";

  // ----------------------------
  // 求解位姿图优化问题
  // ----------------------------
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;  // 使用稀疏Schur求解器
  options.max_num_iterations = 100;                   // 最大迭代次数
  options.minimizer_progress_to_stdout = true;        // 输出优化进度

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // 输出完整的优化报告
  std::cout << summary.FullReport() << std::endl;

  // ----------------------------
  // 将优化后的位姿写回重建对象
  // ----------------------------
  for (auto id : reconstruction.RegImageIds()) {
    Image& image = reconstruction.Image(id);
    double* pose = pose_params[image.ImageId()];

    // 从优化结果中提取四元数和平移
    Eigen::Quaterniond q(pose[0], pose[1], pose[2], pose[3]);
    Eigen::Vector3d t(pose[4], pose[5], pose[6]);
    const double s = pose[7];

    // Sim3转换为SE3：保持相机中心一致，将平移按尺度归一化
    Eigen::Vector3d t_se3 = t;
    if (s > 1e-12) {
      t_se3 = t / s;
    }

    // 将优化后的位姿写回图像对象
    image.SetQvec(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
    image.SetTvec(t_se3);
  }

  // // ----------------------------
  // // 运行光束法平差（可选，当前已注释）
  // // ----------------------------
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

  // 将优化后的重建结果写入输出路径
  reconstruction.Write(output_path);
  std::cout << "PGO + BA done.\n";

  return 0;
}
