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
#include <ceres/local_parameterization.h>

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
                       const double trans_weight, const double scale_weight)
      : q_ij_(q_ij),
        t_ij_dir_(t_ij_dir),
        t_ij_scale_(t_ij_scale),
        rot_weight_(rot_weight),
        trans_weight_(trans_weight),
        scale_weight_(scale_weight) {}

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
    // 额外尺度残差：约束平移长度
    const T r_scale = t_pred.norm() - t_ij_scale;

    residuals[0] = T(rot_weight_) * r_rot(0);
    residuals[1] = T(rot_weight_) * r_rot(1);
    residuals[2] = T(rot_weight_) * r_rot(2);
    residuals[3] = T(trans_weight_) * r_trans(0);
    residuals[4] = T(trans_weight_) * r_trans(1);
    residuals[5] = T(trans_weight_) * r_trans(2);
    residuals[6] = T(scale_weight_) * r_scale;

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Quaterniond& q_ij,
                                     const Eigen::Vector3d& t_ij_dir,
                                     const double t_ij_scale,
                                     const double rot_weight,
                                     const double trans_weight,
                                     const double scale_weight) {
    return new ceres::AutoDiffCostFunction<Sim3RelativePoseCost, 7, 8, 8>(
        new Sim3RelativePoseCost(q_ij, t_ij_dir, t_ij_scale, rot_weight,
                                 trans_weight, scale_weight));
  }

 private:
  Eigen::Quaterniond q_ij_;
  Eigen::Vector3d t_ij_dir_;
  double t_ij_scale_;
  double rot_weight_;
  double trans_weight_;
  double scale_weight_;
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

  // ----------------------------
  // 构建序列边（按相机分组，按image_id递增）
  // ----------------------------
  // 统计边的数量
  int num_edge = 0;  // 序列边（里程计边）数量
  int num_loop = 0;  // 回环边数量

  // 按相机分组并按image_id升序排序
  std::unordered_map<camera_t, std::vector<image_t>> camera_image_ids;
  camera_image_ids.reserve(reconstruction.Cameras().size());
  std::vector<image_t> all_image_ids;
  all_image_ids.reserve(reconstruction.RegImageIds().size());
  for (const auto image_id : reconstruction.RegImageIds()) {
    const Image& image = reconstruction.Image(image_id);
    camera_image_ids[image.CameraId()].push_back(image_id);
    all_image_ids.push_back(image_id);
  }

  if (all_image_ids.empty()) {
    std::cout << "no registered images found in reconstruction.\n";
    return -1;
  }

  for (auto& pair : camera_image_ids) {
    auto& ids = pair.second;
    std::sort(ids.begin(), ids.end());
  }

  std::sort(all_image_ids.begin(), all_image_ids.end());

  const double min_tri_angle = DegToRad(1.0);

  // 记录已添加的序列边，避免后续回环边重复使用同一对
  std::unordered_set<image_pair_t> sequential_pairs;
  sequential_pairs.reserve(all_image_ids.size());

  // 构建序列边：对每个相机按image_id递增连接相邻帧
  for (const auto& pair : camera_image_ids) {
    const auto& ids = pair.second;
    if (ids.size() < 2) {
      continue;
    }
    for (size_t idx = 0; idx + 1 < ids.size(); ++idx) {
      const image_t image_id_i = ids[idx];
      const image_t image_id_j = ids[idx + 1];
      const Image& image_i = reconstruction.Image(image_id_i);
      const Image& image_j = reconstruction.Image(image_id_j);

      // 用当前重建位姿计算相对位姿，作为序列边约束
      Eigen::Vector4d qvec_ij;
      Eigen::Vector3d tvec_ij;
      ComputeRelativePose(image_i.Qvec(), image_i.Tvec(), image_j.Qvec(),
                          image_j.Tvec(), &qvec_ij, &tvec_ij);

      const Eigen::Quaterniond q_ij(qvec_ij(0), qvec_ij(1), qvec_ij(2),
                                    qvec_ij(3));
      const Eigen::Vector3d t_ij = tvec_ij;
      const double t_ij_scale = t_ij.norm();
      if (t_ij_scale <= 1e-12) {
        continue;
      }
      const Eigen::Vector3d t_ij_dir = t_ij / t_ij_scale;

      // 序列边使用Sim3约束：方向 + 尺度构成完整平移，并强化尺度残差
      const double scale_weight = 1.0;
      ceres::CostFunction* cost = Sim3RelativePoseCost::Create(
          q_ij, t_ij_dir, t_ij_scale, 1.0, 1.0, scale_weight);
      problem.AddResidualBlock(cost, nullptr, pose_params[image_id_i],
                               pose_params[image_id_j]);
      sequential_pairs.insert(
          Database::ImagePairToPairId(image_id_i, image_id_j));
      num_edge++;
    }
  }

  // ----------------------------
  // 构建回环边（来自双视图几何）
  // ----------------------------
  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  database.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);
  // 缓存关键点坐标，避免重复访问数据库
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>> points_cache;
  points_cache.reserve(reconstruction.RegImageIds().size());

  // 按内点数量从高到低处理候选，有助于后续稀疏化保留高质量边
  std::vector<size_t> loop_indices(image_pair_ids.size());
  std::iota(loop_indices.begin(), loop_indices.end(), 0);
  std::sort(loop_indices.begin(), loop_indices.end(),
            [&](size_t a, size_t b) {
              return two_view_geometries[a].inlier_matches.size() >
                     two_view_geometries[b].inlier_matches.size();
            });

  // 控制回环边稀疏度：限制每张图像参与的回环边数量
  const size_t kMaxLoopEdgesPerImage = 30;
  std::unordered_map<image_t, size_t> loop_degree;
  loop_degree.reserve(reconstruction.RegImageIds().size());

  const double max_error_px = 4.0;
  const double min_inlier_ratio = 0.2;

  for (const size_t idx : loop_indices) {
    image_t i;
    image_t j;
    Database::PairIdToImagePair(image_pair_ids[idx], &i, &j);
    const TwoViewGeometry& geom = two_view_geometries[idx];

    // 只处理参与优化的图像对
    if (pose_params.find(i) == pose_params.end() ||
        pose_params.find(j) == pose_params.end()) {
      continue;
    }

    const image_pair_t pair_id = Database::ImagePairToPairId(i, j);
    // 已作为序列边的对不再作为回环边
    if (sequential_pairs.count(pair_id) > 0) {
      continue;
    }

    // 过滤掉id差小于100的图像对，避免近邻帧被误认为回环
    if (std::abs(static_cast<int>(i) - static_cast<int>(j)) < 100) {
      continue;
    }

    // 仅保留标定情形，避免UNCALIBRATED时E为空导致相对位姿不可靠
    if (geom.config == colmap::TwoViewGeometry::UNDEFINED ||
        geom.config == colmap::TwoViewGeometry::DEGENERATE ||
        geom.config == colmap::TwoViewGeometry::WATERMARK ||
        geom.config == colmap::TwoViewGeometry::MULTIPLE) {
      continue;
    }

    const size_t num_inliers = geom.inlier_matches.size();
    // 过滤内点过少的回环
    if (num_inliers < 200) {
      continue;
    }

    // 基于内点比例的过滤：内点比例过低的回环容易误检
    const FeatureMatches matches = database.ReadMatches(i, j);
    if (matches.empty()) {
      continue;
    }
    const double inlier_ratio =
        static_cast<double>(num_inliers) /
        static_cast<double>(matches.size());
    if (inlier_ratio < min_inlier_ratio) {
      continue;
    }

    // 过滤三角化角过小的回环（几何退化）
    if (geom.tri_angle > 0 && geom.tri_angle < min_tri_angle) {
      continue;
    }

    // 基于几何误差的过滤：误差过大说明模型不可靠
    const Image& image_i = reconstruction.Image(i);
    const Image& image_j = reconstruction.Image(j);
    const Camera& camera_i = reconstruction.Camera(image_i.CameraId());
    const Camera& camera_j = reconstruction.Camera(image_j.CameraId());
    if (!PassesGeometricErrorFilter(geom, image_i, image_j, camera_i, camera_j,
                                    &database, &points_cache, max_error_px)) {
      continue;
    }

    Eigen::Quaterniond q_ij;
    Eigen::Vector3d t_ij_dir;
    double t_ij_scale = 0.0;
    if (geom.qvec.squaredNorm() > 1e-12 && geom.tvec.squaredNorm() > 1e-12) {
      // 数据库中已有相对位姿，直接使用
      const Eigen::Vector4d normalized_qvec =
          colmap::NormalizeQuaternion(geom.qvec);
      q_ij = Eigen::Quaterniond(normalized_qvec(0), normalized_qvec(1),
                                normalized_qvec(2), normalized_qvec(3));
      t_ij_dir = geom.tvec.normalized();
    } else {
      // 数据库未存qvec/tvec时，基于匹配内点估计相对位姿
      if (!EstimateRelativePoseFromInliers(
              geom, image_i, image_j, camera_i, camera_j, &database,
              &points_cache, &q_ij, &t_ij_dir)) {
        continue;
      }
    }

    // 使用重建中两相机中心的距离作为回环尺度
    const Eigen::Vector3d center_i =
        ProjectionCenterFromPose(image_i.Qvec(), image_i.Tvec());
    const Eigen::Vector3d center_j =
        ProjectionCenterFromPose(image_j.Qvec(), image_j.Tvec());
    t_ij_scale = (center_j - center_i).norm();
    if (t_ij_scale <= 1e-12) {
      continue;
    }
    // std::cout << "q_ij: " << q_ij.w() << " " << q_ij.x() << " " << q_ij.y() << " " << q_ij.z() << std::endl;
    // std::cout << "t_ij_dir: " << t_ij_dir.transpose() << std::endl;

    // 回环边权重：根据内点数和三角化角调节
    const double inlier_scale =
        std::min(3.0, std::sqrt(static_cast<double>(num_inliers) / 200.0));
    // tri_scale 用于抑制小三角化角（近退化）的边
    const double tri_scale =
        (geom.tri_angle > 0) ? std::min(2.0, geom.tri_angle / min_tri_angle)
                             : 1.0;
    // const double rot_weight = 5.0 * inlier_scale * tri_scale;
    const double rot_weight = 5.0;
    // const double trans_weight = 2.0 * inlier_scale * tri_scale;
    const double trans_weight = 1.0;
    // 回环边尺度不确定，尺度残差权重设置为较小值
    const double scale_weight = 0.1;

    // 稀疏度控制：限制每张图像参与的回环边数量
    if (loop_degree[i] >= kMaxLoopEdgesPerImage ||
        loop_degree[j] >= kMaxLoopEdgesPerImage) {
      continue;
    }

    ceres::CostFunction* cost = Sim3RelativePoseCost::Create(
        q_ij, t_ij_dir, t_ij_scale, rot_weight, trans_weight, scale_weight);
    // 回环边加入Huber鲁棒核，降低误匹配的影响
    problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), pose_params[i],
                             pose_params[j]);
    loop_degree[i] += 1;
    loop_degree[j] += 1;
    num_loop++;
  }

  // ----------------------------
  // 固定锚点（gauge fixing）
  // ----------------------------
  // 固定第一张图像的位姿以消除位姿图的全局自由度（6 DoF）
  // 否则整个位姿图可以任意平移和旋转
  image_t root_id = all_image_ids.front();
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
