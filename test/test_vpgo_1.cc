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
#include <fstream>
#include <numeric>
#include <thread>
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
    // pose = [qw qx qy qz tx ty tz]

    Eigen::Quaternion<T> q_i(pose_i[0], pose_i[1], pose_i[2], pose_i[3]);
    Eigen::Matrix<T, 3, 1> t_i(pose_i[4], pose_i[5], pose_i[6]);

    Eigen::Quaternion<T> q_j(pose_j[0], pose_j[1], pose_j[2], pose_j[3]);
    Eigen::Matrix<T, 3, 1> t_j(pose_j[4], pose_j[5], pose_j[6]);

    Eigen::Quaternion<T> q_ij = q_ij_.cast<T>();
    Eigen::Matrix<T, 3, 1> t_ij_obs = t_ij_unit_.cast<T>();

    // rotation residual
    Eigen::Quaternion<T> q_ij_pred = q_j * q_i.conjugate();
    Eigen::Quaternion<T> dq = q_ij.conjugate() * q_ij_pred;

    Eigen::Matrix<T, 3, 1> r_rot;
    r_rot << T(2) * dq.x(), T(2) * dq.y(), T(2) * dq.z();

    // translation direction residual
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

class SE3Manifold : public ceres::Manifold {
 public:
  int AmbientSize() const override { return 7; }
  int TangentSize() const override { return 6; }

  // x_plus_delta = x ⊕ delta
  bool Plus(const double* x, const double* delta,
            double* x_plus_delta) const override {
    // x = [qw qx qy qz tx ty tz]
    Eigen::Quaterniond q(x[0], x[1], x[2], x[3]);
    Eigen::Vector3d t(x[4], x[5], x[6]);

    Eigen::Vector3d omega(delta[0], delta[1], delta[2]);
    Eigen::Vector3d upsilon(delta[3], delta[4], delta[5]);

    double theta = omega.norm();
    Eigen::Quaterniond dq = Eigen::Quaterniond::Identity();
    if (theta > 1e-12) {
      dq = Eigen::Quaterniond(Eigen::AngleAxisd(theta, omega / theta));
    }

    Eigen::Quaterniond q_new = (dq * q).normalized();
    Eigen::Vector3d t_new = t + upsilon;

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

  // y_minus_x = y ⊖ x
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

struct LoopEdge {
  image_t image_id1;
  image_t image_id2;
  Eigen::Quaterniond q_ij;
  Eigen::Vector3d t_ij_dir;
};

// 0. read reconstruction and database
// 1. build pose graph from sequential edges and loop edges from two view
// geometries
// 2. optimize pose graph
// 3. update reconstruction with optimized poses
// 4. run full bundle adjustment
int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "Usage: run_se3_pgo_then_ba "
              << "sparse_path database_path output_path\n";
    return -1;
  }

  std::string sparse_path = argv[1];
  std::string database_path = argv[2];
  std::string output_path = argv[3];

  Reconstruction reconstruction;
  reconstruction.Read(sparse_path);

  Database database(database_path);

  const std::string loop_edges_path = JoinPaths(output_path, "loop_edges.txt");
  std::ofstream loop_edges_file(loop_edges_path);
  if (!loop_edges_file.is_open()) {
    std::cout << "WARNING: Could not open loop edge output file at "
              << loop_edges_path << std::endl;
  }

  // ----------------------------
  // Build pose parameters
  // ----------------------------
  ceres::Problem problem;
  std::unordered_map<image_t, double*> pose_params;

  for (auto& image_pair : reconstruction.Images()) {
    const Image& image = image_pair.second;
    double* pose = new double[7];

    const auto Q = image.Qvec();
    const auto T = image.Tvec();

    pose[0] = Q[0];
    pose[1] = Q[1];
    pose[2] = Q[2];
    pose[3] = Q[3];
    pose[4] = T[0];
    pose[5] = T[1];
    pose[6] = T[2];

    pose_params[image.ImageId()] = pose;
    ceres::Manifold* pose_manifold = new ceres::EigenQuaternionManifold();

    // problem.AddParameterBlock(pose, 7);
    problem.AddParameterBlock(pose, 7, new SE3Manifold());
  }

  // ----------------------------
  //
  // ----------------------------
  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  database.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);
  // using optimized camera to calculate relative poses

  double odom_rot_weight = 10.0;
  double odom_trans_weight = 2.0;

  int num_edge = 0;
  int num_loop = 0;
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
      Eigen::Vector3d t_ij_dir = t_ij.normalized();
      ceres::CostFunction* cost = SE3RelativePoseCostFull::Create(q_ij, t_ij,
                                                              odom_rot_weight,   // rot
                                                              odom_trans_weight);  // trans
      problem.AddResidualBlock(cost, nullptr, pose_params[i], pose_params[j]);
      num_edge++;

    }
    // else if (std::abs(static_cast<int>(i) - static_cast<int>(j)) >= 100) {
    //   // continue;
    //   //  loop edges from two view geometries
    //   if (geom.config == colmap::TwoViewGeometry::UNDEFINED ||
    //       geom.config == colmap::TwoViewGeometry::DEGENERATE ||
    //       geom.config == colmap::TwoViewGeometry::WATERMARK ||
    //       geom.config == colmap::TwoViewGeometry::MULTIPLE) {
    //     std::cout << "pair is degenerate: " << i << " " << j << "\n";
    //     continue;
    //   }

    //   if (geom.inlier_matches.size() < 200) {
    //     std::cout << "pair has too less inliers: " << i << " " << j << " "
    //               << geom.inlier_matches.size() << "\n";
    //     continue;
    //   }

    //   // Eigen::Quaterniond
    //   // q_ij(geom.cam2_from_cam1.rotation.toRotationMatrix()); Eigen::Vector3d
    //   // t_ij = geom.cam2_from_cam1.translation;

    //   const Eigen::Vector4d normalized_qvec =
    //       colmap::NormalizeQuaternion(geom.qvec);
    //   const Eigen::Quaterniond q_ij(normalized_qvec(0), normalized_qvec(1),
    //                                 normalized_qvec(2), normalized_qvec(3));

    //   Eigen::Vector3d t_ij = geom.tvec;
    //   Eigen::Vector3d t_ij_dir = t_ij.normalized();
    //   // Eigen::Vector3d c_ij = (q_ij.conjugate() * t_ij);
    //   // Eigen::Vector3d t_ij_dir = c_ij.normalized();
    //   ceres::CostFunction* cost = SE3RelativePoseCost::Create(q_ij, t_ij_dir,
    //                                                           5.0,   // rot
    //                                                           2.0);  // trans
    //   problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), pose_params[i],
    //                            pose_params[j]);
    //   num_loop++;

    //   if (loop_edges_file.is_open()) {
    //     loop_edges_file << i << " " << j << " " << q_ij.w() << " " << q_ij.x()
    //                     << " " << q_ij.y() << " " << q_ij.z() << " "
    //                     << t_ij_dir(0) << " " << t_ij_dir(1) << " "
    //                     << t_ij_dir(2) << "\n";
    //   }
    // }
  }

  double loop_rot_weight = 50.0;
  double loop_trans_weight = 50.0;

  // ----------------------------
  // 手动添加回环边（从 loop_edges_gt.txt 读取）
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

        ceres::CostFunction* cost = SE3RelativePoseCost::Create(
            q_ij, t_ij_dir, loop_rot_weight, loop_trans_weight);
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0),
                                 pose_params[i], pose_params[j]);

        num_loop++;
      }
    }
  }

  // ----------------------------
  // Fix anchor
  // ----------------------------
  image_t root_id = reconstruction.Images().begin()->first;
  problem.SetParameterBlockConstant(pose_params[root_id]);

  std::cout << "num_edge: " << num_edge << ", num_loop: " << num_loop << "\n";
  // ----------------------------
  // Solve PGO
  // ----------------------------
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  // ----------------------------
  // Write back poses
  // ----------------------------
  // for (auto& image_pair : reconstruction.Images())
  for (auto id : reconstruction.RegImageIds()) {
    Image& image = reconstruction.Image(id);
    double* pose = pose_params[image.ImageId()];

    Eigen::Quaterniond q(pose[0], pose[1], pose[2], pose[3]);
    Eigen::Vector3d t(pose[4], pose[5], pose[6]);

    // image.SetCamFromWorld(Rigid3d(q.toRotationMatrix(), t));
    image.SetQvec(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
    image.SetTvec(t);
  }

  // // ----------------------------
  // // Run BA
  // // ----------------------------
  // BundleAdjustmentOptions ba_options;
  // ba_options.refine_focal_length = false;
  // ba_options.refine_principal_point = false;

  // BundleAdjuster bundle_adjuster(ba_options, &reconstruction);
  // bundle_adjuster.Solve();

  reconstruction.Write(output_path);
  std::cout << "PGO + BA done.\n";

  return 0;
}
