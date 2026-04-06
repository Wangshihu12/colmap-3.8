// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "glomap/view_graph_calibration.h"

#include "glomap/global_mapper.h"

#include <array>
#include <unordered_map>
#include <vector>

#include <Eigen/SVD>
#include <ceres/ceres.h>

#include "base/reconstruction.h"

namespace colmap {
namespace {

// Fetzer 约束中的四元多项式系数构造。
// 这里输入的是分解得到的向量块，输出 4 维系数向量，用于后续焦距残差计算。
Eigen::Vector4d FetzerD(const Eigen::Vector3d& ai,
                        const Eigen::Vector3d& bi,
                        const Eigen::Vector3d& aj,
                        const Eigen::Vector3d& bj,
                        const int u,
                        const int v) {
  Eigen::Vector4d d = Eigen::Vector4d::Zero();
  d(0) = ai(u) * aj(v) - ai(v) * aj(u);
  d(1) = ai(u) * bj(v) - ai(v) * bj(u);
  d(2) = bi(u) * aj(v) - bi(v) * aj(u);
  d(3) = bi(u) * bj(v) - bi(v) * bj(u);
  return d;
}

// 根据基础矩阵构造三组 Fetzer 系数。
// 返回数组长度固定为 3，对应同一对图像上的三条独立约束。
std::array<Eigen::Vector4d, 3> FetzerDs(const Eigen::Matrix3d& G) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      G, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Vector3d s = svd.singularValues();
  const Eigen::Vector3d v0 = svd.matrixV().col(0);
  const Eigen::Vector3d v1 = svd.matrixV().col(1);
  const Eigen::Vector3d u0 = svd.matrixU().col(0);
  const Eigen::Vector3d u1 = svd.matrixU().col(1);

  const Eigen::Vector3d ai(
      s(0) * s(0) * (v0(0) * v0(0) + v0(1) * v0(1)),
      s(0) * s(1) * (v0(0) * v1(0) + v0(1) * v1(1)),
      s(1) * s(1) * (v1(0) * v1(0) + v1(1) * v1(1)));
  const Eigen::Vector3d aj(
      u1(0) * u1(0) + u1(1) * u1(1),
      -(u0(0) * u1(0) + u0(1) * u1(1)),
      u0(0) * u0(0) + u0(1) * u0(1));
  const Eigen::Vector3d bi(
      s(0) * s(0) * v0(2) * v0(2),
      s(0) * s(1) * v0(2) * v1(2),
      s(1) * s(1) * v1(2) * v1(2));
  const Eigen::Vector3d bj(
      u1(2) * u1(2), -(u0(2) * u1(2)), u0(2) * u0(2));

  return {FetzerD(ai, bi, aj, bj, 1, 0),
          FetzerD(ai, bi, aj, bj, 0, 2),
          FetzerD(ai, bi, aj, bj, 2, 1)};
}

struct FetzerFocalLengthCost {
  FetzerFocalLengthCost(const Eigen::Matrix3d& F,
                        const Eigen::Vector2d& pp0,
                        const Eigen::Vector2d& pp1) {
    Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity();
    K0(0, 2) = pp0(0);
    K0(1, 2) = pp0(1);

    Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity();
    K1(0, 2) = pp1(0);
    K1(1, 2) = pp1(1);

    const std::array<Eigen::Vector4d, 3> ds =
        FetzerDs(K1.transpose() * F * K0);
    d01 = ds[0];
    d12 = ds[2];
  }

  template <typename T>
  bool operator()(const T* const f0, const T* const f1, T* residuals) const {
    // residuals 形状为 [2]，分别约束两台相机的焦距平方一致性。
    T di = f1[0] * f1[0] * T(d01(0)) + T(d01(1));
    T dj = f0[0] * f0[0] * T(d12(0)) + T(d12(2));
    if (ceres::abs(di) < T(1e-9)) di = T(1e-9);
    if (ceres::abs(dj) < T(1e-9)) dj = T(1e-9);

    const T k0 = -(f1[0] * f1[0] * T(d01(2)) + T(d01(3))) / di;
    const T k1 = -(f0[0] * f0[0] * T(d12(1)) + T(d12(3))) / dj;

    residuals[0] = (f0[0] * f0[0] - k0) / (f0[0] * f0[0]);
    residuals[1] = (f1[0] * f1[0] - k1) / (f1[0] * f1[0]);
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d& F,
                                     const Eigen::Vector2d& pp0,
                                     const Eigen::Vector2d& pp1) {
    return new ceres::AutoDiffCostFunction<FetzerFocalLengthCost, 2, 1, 1>(
        new FetzerFocalLengthCost(F, pp0, pp1));
  }

  Eigen::Vector4d d01;
  Eigen::Vector4d d12;
};

struct FetzerFocalLengthSameCameraCost {
  FetzerFocalLengthSameCameraCost(const Eigen::Matrix3d& F,
                                  const Eigen::Vector2d& pp) {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 2) = pp(0);
    K(1, 2) = pp(1);
    const std::array<Eigen::Vector4d, 3> ds = FetzerDs(K.transpose() * F * K);
    d01 = ds[0];
    d02 = ds[1];
    d12 = ds[2];
  }

  template <typename T>
  bool operator()(const T* const f, T* residuals) const {
    // residuals 形状为 [3]，同相机内参时使用三条约束联合估计同一个焦距。
    const T f2 = f[0] * f[0];
    residuals[0] = f2 * f2 * T(d01(0)) +
                   f2 * (T(d01(1)) + T(d01(2))) + T(d01(3));
    residuals[1] = f2 * f2 * T(d02(0)) +
                   f2 * (T(d02(1)) + T(d02(2))) + T(d02(3));
    residuals[2] = f2 * f2 * T(d12(0)) +
                   f2 * (T(d12(1)) + T(d12(2))) + T(d12(3));
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d& F,
                                     const Eigen::Vector2d& pp) {
    return new ceres::AutoDiffCostFunction<FetzerFocalLengthSameCameraCost, 3, 1>(
        new FetzerFocalLengthSameCameraCost(F, pp));
  }

  Eigen::Vector4d d01;
  Eigen::Vector4d d02;
  Eigen::Vector4d d12;
};

}  // namespace

bool CalibrateViewGraph(const ViewGraphCalibrationOptions& options,
                        std::vector<ViewEdge>* edges,
                        Reconstruction* reconstruction) {
  CHECK_NOTNULL(edges);
  CHECK_NOTNULL(reconstruction);
  if (edges->empty()) {
    return true;
  }

  std::unordered_map<camera_t, double> focals;
  std::unordered_map<camera_t, double> original_focals;
  // 保存每个相机当前的平均焦距，后面优化的是这个标量而不是整组内参。
  for (const auto& item : reconstruction->Cameras()) {
    focals[item.first] = item.second.MeanFocalLength();
    original_focals[item.first] = item.second.MeanFocalLength();
  }

  ceres::Problem problem;
  for (const ViewEdge& edge : *edges) {
    // 只对可用于几何约束的配置类型建立残差。
    if (edge.geometry.config != TwoViewGeometry::CALIBRATED &&
        edge.geometry.config != TwoViewGeometry::UNCALIBRATED) {
      continue;
    }

    const Camera& camera1 = reconstruction->Camera(
        reconstruction->Image(edge.image_id1).CameraId());
    const Camera& camera2 = reconstruction->Camera(
        reconstruction->Image(edge.image_id2).CameraId());
    if (camera1.CameraId() == camera2.CameraId()) {
      problem.AddResidualBlock(
          FetzerFocalLengthSameCameraCost::Create(
              edge.geometry.F,
              Eigen::Vector2d(camera1.PrincipalPointX(),
                              camera1.PrincipalPointY())),
          new ceres::CauchyLoss(options.loss_scale),
          &focals[camera1.CameraId()]);
    } else {
      problem.AddResidualBlock(
          FetzerFocalLengthCost::Create(
              edge.geometry.F,
              Eigen::Vector2d(camera1.PrincipalPointX(),
                              camera1.PrincipalPointY()),
              Eigen::Vector2d(camera2.PrincipalPointX(),
                              camera2.PrincipalPointY())),
          new ceres::CauchyLoss(options.loss_scale),
          &focals[camera1.CameraId()],
          &focals[camera2.CameraId()]);
    }
  }

  for (const auto& item : reconstruction->Cameras()) {
    const camera_t camera_id = item.first;
    const Camera& camera = item.second;
    if (!problem.HasParameterBlock(&focals[camera_id])) {
      continue;
    }

    problem.SetParameterLowerBound(&focals[camera_id], 0, 1e-3);
    // 若相机已有可信焦距先验，则保持不动，让其它相机向其对齐。
    if (camera.HasPriorFocalLength()) {
      problem.SetParameterBlockConstant(&focals[camera_id]);
    }
  }

  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = options.max_num_iterations;
  solver_options.linear_solver_type =
      reconstruction->NumCameras() < 50 ? ceres::DENSE_NORMAL_CHOLESKY
                                        : ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.minimizer_progress_to_stdout = false;
  solver_options.num_threads = 1;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  if (!summary.IsSolutionUsable()) {
    return false;
  }

  for (auto& item : reconstruction->Cameras()) {
    const camera_t camera_id = item.first;
    Camera& camera = reconstruction->Camera(camera_id);
    if (!problem.HasParameterBlock(&focals[camera_id])) {
      continue;
    }

    const double ratio = focals[camera_id] / original_focals[camera_id];
    // 只接受落在合理比例范围内的焦距更新，避免异常边把内参拉飞。
    if (ratio < options.min_focal_ratio || ratio > options.max_focal_ratio) {
      continue;
    }

    for (const size_t idx : camera.FocalLengthIdxs()) {
      camera.Params(idx) = focals[camera_id];
    }
    camera.SetPriorFocalLength(true);
  }

  ceres::Problem::EvaluateOptions eval_options;
  eval_options.apply_loss_function = false;
  eval_options.num_threads = 1;
  std::vector<double> residuals;
  problem.Evaluate(eval_options, nullptr, &residuals, nullptr, nullptr);

  size_t residual_idx = 0;
  const double max_error_sq = options.max_two_view_error *
                              options.max_two_view_error;
  for (ViewEdge& edge : *edges) {
    if (edge.geometry.config != TwoViewGeometry::CALIBRATED &&
        edge.geometry.config != TwoViewGeometry::UNCALIBRATED) {
      continue;
    }

    const Camera& camera1 = reconstruction->Camera(
        reconstruction->Image(edge.image_id1).CameraId());
    const Camera& camera2 = reconstruction->Camera(
        reconstruction->Image(edge.image_id2).CameraId());
    if (camera1.CameraId() == camera2.CameraId()) {
      const double error_sq =
          residuals[residual_idx] * residuals[residual_idx] +
          residuals[residual_idx + 1] * residuals[residual_idx + 1] +
          residuals[residual_idx + 2] * residuals[residual_idx + 2];
      if (error_sq > max_error_sq) {
        // 这里不直接删边，而是把 weight 设为负数，由上层统一判断其失效。
        edge.weight = -1.0;
      }
      residual_idx += 3;
    } else {
      const double error_sq =
          residuals[residual_idx] * residuals[residual_idx] +
          residuals[residual_idx + 1] * residuals[residual_idx + 1];
      if (error_sq > max_error_sq) {
        edge.weight = -1.0;
      }
      residual_idx += 2;
    }
  }

  return true;
}

}  // namespace colmap
