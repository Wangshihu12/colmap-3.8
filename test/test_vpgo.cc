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
#include "base/reconstruction.h"
#include "controllers/bundle_adjustment.h"
#include "controllers/incremental_mapper.h"

#include "util/misc.h"
#include "util/string.h"
// #include "xgbase/utils.h"

using namespace colmap;

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
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
  for (auto& image_pair : reconstruction.Images()) {
    const Image& image = image_pair.second;
    // 分配7维位姿参数：4维四元数（qw, qx, qy, qz）+ 3维平移（tx, ty, tz）
    double* pose = new double[7];

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

    // 存储位姿参数指针
    pose_params[image.ImageId()] = pose;
    // 注：这里创建了流形但未使用，仅作为参考
    ceres::Manifold* pose_manifold = new ceres::EigenQuaternionManifold();

    // 将位姿参数块添加到优化问题中
    // 使用SE3流形确保四元数的单位性约束和正确的切空间参数化
    problem.AddParameterBlock(pose, 7, new SE3Manifold());
  }

  // ----------------------------
  // 构建位姿图的边（约束）
  // ----------------------------
  // 从数据库读取所有图像对的双视图几何信息
  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  database.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);

  // 统计边的数量
  int num_edge = 0;  // 序列边（里程计边）数量
  int num_loop = 0;  // 回环边数量

  // 遍历所有图像对，构建位姿图的边
  for (size_t k = 0; k < image_pair_ids.size(); ++k) {
    image_t i;
    image_t j;
    // 将图像对ID解析为两个图像ID
    Database::PairIdToImagePair(image_pair_ids[k], &i, &j);
    const TwoViewGeometry& geom = two_view_geometries[k];
    std::cout << "q_ij: " << geom.qvec.transpose() << "\n";

    // 跳过不存在的图像
    if (!reconstruction.ExistsImage(i) || !reconstruction.ExistsImage(j)) {
      continue;
    }

    // 根据图像ID差异判断边的类型
    if (std::abs(static_cast<int>(i) - static_cast<int>(j)) <= 10) {
      // 序列边（里程计边）：ID差异小于等于10的相邻帧
      // 使用优化后的相机位姿计算相对位姿作为约束
      const Image& image_i = reconstruction.Image(i);
      const Image& image_j = reconstruction.Image(j);

      // 构造图像i的旋转四元数，形状为(4,)：(w, x, y, z)
      Eigen::Quaterniond q_i(image_i.Qvec()[0], image_i.Qvec()[1],
                             image_i.Qvec()[2], image_i.Qvec()[3]);
      // 图像i的平移向量，形状为(3,)
      Eigen::Vector3d t_i = image_i.Tvec();

      // 构造图像j的旋转四元数
      Eigen::Quaterniond q_j(image_j.Qvec()[0], image_j.Qvec()[1],
                             image_j.Qvec()[2], image_j.Qvec()[3]);
      // 图像j的平移向量
      Eigen::Vector3d t_j = image_j.Tvec();

      // 计算从i到j的相对旋转：q_ij = q_j * q_i^{-1}
      Eigen::Quaterniond q_ij = q_j * q_i.conjugate();
      // 计算从i到j的相对平移：t_ij = t_j - q_ij * t_i
      Eigen::Vector3d t_ij = t_j - (q_ij * t_i);
      // 归一化平移方向（仅使用方向信息，不使用尺度）
      Eigen::Vector3d t_ij_dir = t_ij.normalized();

      // 创建SE3相对位姿代价函数
      // 参数：相对旋转、相对平移方向、旋转权重、平移权重
      ceres::CostFunction* cost = SE3RelativePoseCost::Create(q_ij, t_ij_dir,
                                                              1.0,   // 旋转权重
                                                              1.0);  // 平移权重
      // 添加残差块，无鲁棒损失函数（序列边通常可靠）
      problem.AddResidualBlock(cost, nullptr, pose_params[i], pose_params[j]);
      num_edge++;

    } else if (std::abs(static_cast<int>(i) - static_cast<int>(j)) >= 100) {
      // 回环边：ID差异大于等于100的非相邻帧
      // 使用双视图几何估计的相对位姿作为约束

      // 过滤无效的双视图几何配置
      if (geom.config == colmap::TwoViewGeometry::UNDEFINED ||
          geom.config == colmap::TwoViewGeometry::DEGENERATE ||
          geom.config == colmap::TwoViewGeometry::WATERMARK ||
          geom.config == colmap::TwoViewGeometry::MULTIPLE) {
        std::cout << "pair is degenerate: " << i << " " << j << "\n";
        continue;
      }

      // 过滤内点数过少的匹配（阈值200）
      if (geom.inlier_matches.size() < 200) {
        std::cout << "pair has too less inliers: " << i << " " << j << " "
                  << geom.inlier_matches.size() << "\n";
        continue;
      }

      // 从双视图几何中提取相对位姿
      // 归一化四元数以确保单位性
      const Eigen::Vector4d normalized_qvec =
          colmap::NormalizeQuaternion(geom.qvec);
      const Eigen::Quaterniond q_ij(normalized_qvec(0), normalized_qvec(1),
                                    normalized_qvec(2), normalized_qvec(3));

      // 获取相对平移并归一化
      Eigen::Vector3d t_ij = geom.tvec;
      Eigen::Vector3d t_ij_dir = t_ij.normalized();

      // 创建SE3相对位姿代价函数
      // 回环边使用更大的权重以强调回环约束的重要性
      ceres::CostFunction* cost = SE3RelativePoseCost::Create(q_ij, t_ij_dir,
                                                              5.0,   // 旋转权重（较大）
                                                              2.0);  // 平移权重（较大）
      // 添加残差块，使用Huber鲁棒损失函数抑制回环误检
      problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), pose_params[i],
                               pose_params[j]);
      num_loop++;
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

    // 将优化后的位姿写回图像对象
    image.SetQvec(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
    image.SetTvec(t);
  }

  // // ----------------------------
  // // 运行光束法平差（可选，当前已注释）
  // // ----------------------------
  // BundleAdjustmentOptions ba_options;
  // ba_options.refine_focal_length = false;
  // ba_options.refine_principal_point = false;

  // BundleAdjuster bundle_adjuster(ba_options, &reconstruction);
  // bundle_adjuster.Solve();

  // 将优化后的重建结果写入输出路径
  reconstruction.Write(output_path);
  std::cout << "PGO + BA done.\n";

  return 0;
}