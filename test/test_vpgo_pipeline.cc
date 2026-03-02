#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <ceres/ceres.h>

#include "base/camera_rig.h"
#include "base/database.h"
#include "base/pose.h"
#include "base/reconstruction.h"
#include "base/scene_clustering.h"
#include "controllers/bundle_adjustment.h"
#include "controllers/incremental_mapper.h"
#include "estimators/similarity_transform.h"
#include "estimators/homography_matrix.h"
#include "estimators/pose.h"
#include "estimators/triangulation.h"
#include "estimators/two_view_geometry.h"
#include "estimators/utils.h"
#include "feature/utils.h"
#include "optim/loransac.h"
#include "optim/ransac.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/string.h"
#include "util/threading.h"

using namespace colmap;

enum DistributionType { DISTRIBUTE_BY_NUM = 0, DISTRIBUTE_BY_ERROR = 1 };
inline std::vector<std::vector<Point2D>> Distribute(
    const std::vector<Point2D>& lfs) {
  std::vector<std::vector<Point2D>> lists(4);
  if (!lfs.empty()) {
    double minx, miny, maxx, maxy;
    minx = miny = std::numeric_limits<double>::max();
    maxx = maxy = -std::numeric_limits<double>::max();
    for (const auto& f : lfs) {
      if (f.X() < minx) {
        minx = f.X();
      }
      if (f.X() > maxx) {
        maxx = f.X();
      }
      if (f.Y() < miny) {
        miny = f.Y();
      }
      if (f.Y() > maxy) {
        maxy = f.Y();
      }
    }
    double midx = (maxx + minx) / 2, midy = (maxy + miny) / 2;
    for (const auto& f : lfs) {
      double x = f.X(), y = f.Y();
      if (x >= minx && x < midx && y >= miny && y < midy) {
        lists[0].push_back(f);
      }
      if (x >= midx && x <= maxx && y >= miny && y < midy) {
        lists[1].push_back(f);
      }
      if (x >= midx && x <= maxx && y >= midy && y <= maxy) {
        lists[2].push_back(f);
      }
      if (x >= minx && x < midx && y >= midy && y <= maxy) {
        lists[3].push_back(f);
      }
    }
  }
  return lists;
}


inline std::vector<point3D_t> DistributeLandmarksImpl(
    Reconstruction& reconstruction, std::vector<Point2D> keyPoints,
    DistributionType type, int minNodeCount, int maxSplitCount = 16) {
  std::vector<std::vector<Point2D>> lists;
  lists.push_back(keyPoints);
  for (size_t iter = 0;
       iter < size_t(maxSplitCount) && lists.size() < size_t(minNodeCount);
       ++iter) {
    std::vector<std::vector<Point2D>> tmpLists;
    for (size_t i = 0; i < lists.size(); ++i) {
      auto tmp = Distribute(lists[i]);
      for (const auto& lfs : tmp) {
        if (!lfs.empty()) {
          tmpLists.push_back(lfs);
        }
      }
    }
    lists.swap(tmpLists);
  }
  std::vector<point3D_t> validLandmarkIdx;
  for (const auto& lfs : lists) {
    size_t maxOb = 0;
    float min_err = std::numeric_limits<float>::max();
    point3D_t max_idx = lfs[0].Point3DId();
    for (const auto& lf : lfs) {
      size_t track_length =
          reconstruction.Point3D(lf.Point3DId()).Track().Length();
      float err = std::numeric_limits<float>::max();
      if (reconstruction.Point3D(lf.Point3DId()).HasError())
        err = reconstruction.Point3D(lf.Point3DId()).Error();
      if (type == DistributionType::DISTRIBUTE_BY_NUM) {
        if (track_length > maxOb) {
          max_idx = lf.Point3DId();
          maxOb = track_length;
        }
      } else if (type == DistributionType::DISTRIBUTE_BY_ERROR) {
        if (err < min_err) {
          max_idx = lf.Point3DId();
          min_err = err;
        }
      }
    }
    validLandmarkIdx.push_back(max_idx);
  }
  return validLandmarkIdx;
}


inline bool DistributeLandmarks(
    Reconstruction& reconstruction, int average_num,
    std::map<std::string, int> map_map_featunum = {},
    DistributionType type = DISTRIBUTE_BY_NUM, bool remove_tracks = true) {
  uint32_t remove_num = 0;
  for (point3D_t point3D_id : reconstruction.Point3DIds()) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
  }
  std::set<point3D_t> validLandmarkIds;
  for (const image_t image_id : reconstruction.RegImageIds()) {
    Image& image = reconstruction.Image(image_id);
    std::vector<Point2D> keyPoints;
    for (const Point2D& point2D : image.Points2D()) {
      if (!point2D.HasPoint3D()) continue;
      keyPoints.push_back(point2D);
    }

    auto image_name = image.Name();
    //auto image_prefix_name = GetPathPrefixName(image_name);
    // if (!map_map_featunum.empty() &&
    //     map_map_featunum.count(image_prefix_name) > 0) {
    //   average_num = map_map_featunum.at(image_prefix_name);
    // }
    auto validIdx =
        DistributeLandmarksImpl(reconstruction, keyPoints, type, average_num);
    for (auto id : validIdx) {
      validLandmarkIds.insert(id);
    }
  }
  std::cout << "MMMMMMMMMMMMMMMMMM[Point3D before]"
            << reconstruction.Point3DIds().size() << std::endl;
  for (point3D_t point3D_id : reconstruction.Point3DIds()) {
    if (validLandmarkIds.find(point3D_id) == validLandmarkIds.end()) {
      if (remove_tracks)
        reconstruction.DeletePoint3D(point3D_id);
      else {
        Point3D& point3D = reconstruction.Point3D(point3D_id);
        remove_num++;
      }
    }
  }
  std::cout << "MMMMMMMMMMMMMMMMMM[Point3D  after]"
            << reconstruction.Point3DIds().size() << " - " << remove_num
            << std::endl;
  return true;
}



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

// 不同相机，不同时间快照
struct SE3RelativePoseCostRig {
  SE3RelativePoseCostRig(const Eigen::Quaterniond& q_ij,
                         const Eigen::Vector3d& t_ij_unit, double rot_weight,
                         double trans_weight)
      : q_ij_(q_ij),
        t_ij_unit_(t_ij_unit),
        rot_weight_(rot_weight),
        trans_weight_(trans_weight) {}

  template <typename T>
  bool operator()(const T* const base_i, const T* const rel_i,
                  const T* const base_j, const T* const rel_j,
                  T* residuals) const {
    Eigen::Quaternion<T> q_base_i(base_i[0], base_i[1], base_i[2], base_i[3]);
    Eigen::Matrix<T, 3, 1> t_base_i(base_i[4], base_i[5], base_i[6]);
    Eigen::Quaternion<T> q_rel_i(rel_i[0], rel_i[1], rel_i[2], rel_i[3]);
    Eigen::Matrix<T, 3, 1> t_rel_i(rel_i[4], rel_i[5], rel_i[6]);

    Eigen::Quaternion<T> q_base_j(base_j[0], base_j[1], base_j[2], base_j[3]);
    Eigen::Matrix<T, 3, 1> t_base_j(base_j[4], base_j[5], base_j[6]);
    Eigen::Quaternion<T> q_rel_j(rel_j[0], rel_j[1], rel_j[2], rel_j[3]);
    Eigen::Matrix<T, 3, 1> t_rel_j(rel_j[4], rel_j[5], rel_j[6]);

    const Eigen::Quaternion<T> q_ij = q_ij_.cast<T>();
    const Eigen::Matrix<T, 3, 1> t_ij_obs = t_ij_unit_.cast<T>();

    // 计算相机i和相机j的绝对位姿: rel * base
    const Eigen::Quaternion<T> q_i = q_rel_i * q_base_i;
    const Eigen::Matrix<T, 3, 1> t_i = t_rel_i + q_rel_i * t_base_i;
    const Eigen::Quaternion<T> q_j = q_rel_j * q_base_j;
    const Eigen::Matrix<T, 3, 1> t_j = t_rel_j + q_rel_j * t_base_j;

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
    return new ceres::AutoDiffCostFunction<SE3RelativePoseCostRig, 6, 7, 7, 7,
                                           7>(
        new SE3RelativePoseCostRig(q_ij, t_ij_unit, rot_weight, trans_weight));
  }

 private:
  Eigen::Quaterniond q_ij_;
  Eigen::Vector3d t_ij_unit_;
  double rot_weight_;
  double trans_weight_;
};

// 不同相机，不同时间快照
struct SE3RelativePoseCostRigFull {
  SE3RelativePoseCostRigFull(const Eigen::Quaterniond& q_ij,
                             const Eigen::Vector3d& t_ij, double rot_weight,
                             double trans_weight)
      : q_ij_(q_ij),
        t_ij_(t_ij),
        rot_weight_(rot_weight),
        trans_weight_(trans_weight) {}

  template <typename T>
  bool operator()(const T* const base_i, const T* const rel_i,
                  const T* const base_j, const T* const rel_j,
                  T* residuals) const {
    Eigen::Quaternion<T> q_base_i(base_i[0], base_i[1], base_i[2], base_i[3]);
    Eigen::Matrix<T, 3, 1> t_base_i(base_i[4], base_i[5], base_i[6]);
    Eigen::Quaternion<T> q_rel_i(rel_i[0], rel_i[1], rel_i[2], rel_i[3]);
    Eigen::Matrix<T, 3, 1> t_rel_i(rel_i[4], rel_i[5], rel_i[6]);

    Eigen::Quaternion<T> q_base_j(base_j[0], base_j[1], base_j[2], base_j[3]);
    Eigen::Matrix<T, 3, 1> t_base_j(base_j[4], base_j[5], base_j[6]);
    Eigen::Quaternion<T> q_rel_j(rel_j[0], rel_j[1], rel_j[2], rel_j[3]);
    Eigen::Matrix<T, 3, 1> t_rel_j(rel_j[4], rel_j[5], rel_j[6]);

    const Eigen::Quaternion<T> q_ij = q_ij_.cast<T>();
    const Eigen::Matrix<T, 3, 1> t_ij_obs = t_ij_.cast<T>();

    // 计算相机i和相机j的绝对位姿: rel * base
    const Eigen::Quaternion<T> q_i = q_rel_i * q_base_i;
    const Eigen::Matrix<T, 3, 1> t_i = t_rel_i + q_rel_i * t_base_i;
    const Eigen::Quaternion<T> q_j = q_rel_j * q_base_j;
    const Eigen::Matrix<T, 3, 1> t_j = t_rel_j + q_rel_j * t_base_j;

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
    return new ceres::AutoDiffCostFunction<SE3RelativePoseCostRigFull, 6, 7, 7,
                                           7, 7>(
        new SE3RelativePoseCostRigFull(q_ij, t_ij, rot_weight, trans_weight));
  }

 private:
  Eigen::Quaterniond q_ij_;
  Eigen::Vector3d t_ij_;
  double rot_weight_;
  double trans_weight_;
};

// 同一相机，不同时间快照，共享 rel_pose
struct SE3RelativePoseCostRigSharedRel {
  SE3RelativePoseCostRigSharedRel(const Eigen::Quaterniond& q_ij,
                                  const Eigen::Vector3d& t_ij_unit,
                                  double rot_weight, double trans_weight)
      : q_ij_(q_ij),
        t_ij_unit_(t_ij_unit),
        rot_weight_(rot_weight),
        trans_weight_(trans_weight) {}

  template <typename T>
  bool operator()(const T* const base_i, const T* const base_j,
                  const T* const rel, T* residuals) const {
    Eigen::Quaternion<T> q_base_i(base_i[0], base_i[1], base_i[2], base_i[3]);
    Eigen::Matrix<T, 3, 1> t_base_i(base_i[4], base_i[5], base_i[6]);
    Eigen::Quaternion<T> q_base_j(base_j[0], base_j[1], base_j[2], base_j[3]);
    Eigen::Matrix<T, 3, 1> t_base_j(base_j[4], base_j[5], base_j[6]);
    Eigen::Quaternion<T> q_rel(rel[0], rel[1], rel[2], rel[3]);
    Eigen::Matrix<T, 3, 1> t_rel(rel[4], rel[5], rel[6]);

    const Eigen::Quaternion<T> q_ij = q_ij_.cast<T>();
    const Eigen::Matrix<T, 3, 1> t_ij_obs = t_ij_unit_.cast<T>();

    const Eigen::Quaternion<T> q_i = q_rel * q_base_i;
    const Eigen::Matrix<T, 3, 1> t_i = t_rel + q_rel * t_base_i;
    const Eigen::Quaternion<T> q_j = q_rel * q_base_j;
    const Eigen::Matrix<T, 3, 1> t_j = t_rel + q_rel * t_base_j;

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
    return new ceres::AutoDiffCostFunction<SE3RelativePoseCostRigSharedRel, 6,
                                           7, 7, 7>(
        new SE3RelativePoseCostRigSharedRel(q_ij, t_ij_unit, rot_weight,
                                            trans_weight));
  }

 private:
  Eigen::Quaterniond q_ij_;
  Eigen::Vector3d t_ij_unit_;
  double rot_weight_;
  double trans_weight_;
};

// 同一相机，不同时间快照，共享 rel_pose
struct SE3RelativePoseCostRigSharedRelFull {
  SE3RelativePoseCostRigSharedRelFull(const Eigen::Quaterniond& q_ij,
                                      const Eigen::Vector3d& t_ij,
                                      double rot_weight, double trans_weight)
      : q_ij_(q_ij),
        t_ij_(t_ij),
        rot_weight_(rot_weight),
        trans_weight_(trans_weight) {}

  template <typename T>
  bool operator()(const T* const base_i, const T* const base_j,
                  const T* const rel, T* residuals) const {
    Eigen::Quaternion<T> q_base_i(base_i[0], base_i[1], base_i[2], base_i[3]);
    Eigen::Matrix<T, 3, 1> t_base_i(base_i[4], base_i[5], base_i[6]);
    Eigen::Quaternion<T> q_base_j(base_j[0], base_j[1], base_j[2], base_j[3]);
    Eigen::Matrix<T, 3, 1> t_base_j(base_j[4], base_j[5], base_j[6]);
    Eigen::Quaternion<T> q_rel(rel[0], rel[1], rel[2], rel[3]);
    Eigen::Matrix<T, 3, 1> t_rel(rel[4], rel[5], rel[6]);

    const Eigen::Quaternion<T> q_ij = q_ij_.cast<T>();
    const Eigen::Matrix<T, 3, 1> t_ij_obs = t_ij_.cast<T>();

    const Eigen::Quaternion<T> q_i = q_rel * q_base_i;
    const Eigen::Matrix<T, 3, 1> t_i = t_rel + q_rel * t_base_i;
    const Eigen::Quaternion<T> q_j = q_rel * q_base_j;
    const Eigen::Matrix<T, 3, 1> t_j = t_rel + q_rel * t_base_j;

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
    return new ceres::AutoDiffCostFunction<
        SE3RelativePoseCostRigSharedRelFull, 6, 7, 7, 7>(
        new SE3RelativePoseCostRigSharedRelFull(q_ij, t_ij, rot_weight,
                                                trans_weight));
  }

 private:
  Eigen::Quaterniond q_ij_;
  Eigen::Vector3d t_ij_;
  double rot_weight_;
  double trans_weight_;
};

// 同一时间快照，不同相机，共享 base_pose
struct SE3RelativePoseCostRigSharedBase {
  SE3RelativePoseCostRigSharedBase(const Eigen::Quaterniond& q_ij,
                                   const Eigen::Vector3d& t_ij_unit,
                                   double rot_weight, double trans_weight)
      : q_ij_(q_ij),
        t_ij_unit_(t_ij_unit),
        rot_weight_(rot_weight),
        trans_weight_(trans_weight) {}

  template <typename T>
  bool operator()(const T* const base, const T* const rel_i,
                  const T* const rel_j, T* residuals) const {
    Eigen::Quaternion<T> q_base(base[0], base[1], base[2], base[3]);
    Eigen::Matrix<T, 3, 1> t_base(base[4], base[5], base[6]);
    Eigen::Quaternion<T> q_rel_i(rel_i[0], rel_i[1], rel_i[2], rel_i[3]);
    Eigen::Matrix<T, 3, 1> t_rel_i(rel_i[4], rel_i[5], rel_i[6]);
    Eigen::Quaternion<T> q_rel_j(rel_j[0], rel_j[1], rel_j[2], rel_j[3]);
    Eigen::Matrix<T, 3, 1> t_rel_j(rel_j[4], rel_j[5], rel_j[6]);

    const Eigen::Quaternion<T> q_ij = q_ij_.cast<T>();
    const Eigen::Matrix<T, 3, 1> t_ij_obs = t_ij_unit_.cast<T>();

    const Eigen::Quaternion<T> q_i = q_rel_i * q_base;
    const Eigen::Matrix<T, 3, 1> t_i = t_rel_i + q_rel_i * t_base;
    const Eigen::Quaternion<T> q_j = q_rel_j * q_base;
    const Eigen::Matrix<T, 3, 1> t_j = t_rel_j + q_rel_j * t_base;

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
    return new ceres::AutoDiffCostFunction<SE3RelativePoseCostRigSharedBase, 6,
                                           7, 7, 7>(
        new SE3RelativePoseCostRigSharedBase(q_ij, t_ij_unit, rot_weight,
                                             trans_weight));
  }

 private:
  Eigen::Quaterniond q_ij_;
  Eigen::Vector3d t_ij_unit_;
  double rot_weight_;
  double trans_weight_;
};

// 同一时间快照，不同相机，共享 base_pose
struct SE3RelativePoseCostRigSharedBaseFull {
  SE3RelativePoseCostRigSharedBaseFull(const Eigen::Quaterniond& q_ij,
                                       const Eigen::Vector3d& t_ij,
                                       double rot_weight, double trans_weight)
      : q_ij_(q_ij),
        t_ij_(t_ij),
        rot_weight_(rot_weight),
        trans_weight_(trans_weight) {}

  template <typename T>
  bool operator()(const T* const base, const T* const rel_i,
                  const T* const rel_j, T* residuals) const {
    Eigen::Quaternion<T> q_base(base[0], base[1], base[2], base[3]);
    Eigen::Matrix<T, 3, 1> t_base(base[4], base[5], base[6]);
    Eigen::Quaternion<T> q_rel_i(rel_i[0], rel_i[1], rel_i[2], rel_i[3]);
    Eigen::Matrix<T, 3, 1> t_rel_i(rel_i[4], rel_i[5], rel_i[6]);
    Eigen::Quaternion<T> q_rel_j(rel_j[0], rel_j[1], rel_j[2], rel_j[3]);
    Eigen::Matrix<T, 3, 1> t_rel_j(rel_j[4], rel_j[5], rel_j[6]);

    const Eigen::Quaternion<T> q_ij = q_ij_.cast<T>();
    const Eigen::Matrix<T, 3, 1> t_ij_obs = t_ij_.cast<T>();

    const Eigen::Quaternion<T> q_i = q_rel_i * q_base;
    const Eigen::Matrix<T, 3, 1> t_i = t_rel_i + q_rel_i * t_base;
    const Eigen::Quaternion<T> q_j = q_rel_j * q_base;
    const Eigen::Matrix<T, 3, 1> t_j = t_rel_j + q_rel_j * t_base;

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
    return new ceres::AutoDiffCostFunction<
        SE3RelativePoseCostRigSharedBaseFull, 6, 7, 7, 7>(
        new SE3RelativePoseCostRigSharedBaseFull(q_ij, t_ij, rot_weight,
                                                 trans_weight));
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

enum class EdgeType { kOdom, kLoop, kRig };

struct EdgeDefaults {
  double odom_rot_weight = 2.0;
  double odom_trans_weight = 2.0;
  double loop_rot_weight = 1.0;
  double loop_trans_weight = 2.0;
  double rig_rot_weight = 5.0;
  double rig_trans_weight = 5.0;
  bool odom_translation_is_unit = false;
  bool loop_translation_is_unit = true;
  bool rig_translation_is_unit = false;
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

struct ImagePoseParams {
  double* base_pose = nullptr;  // 不同快照下的相机基准位姿
  double* rel_pose = nullptr;   // 统一快照，不同相机的相对位姿
};

struct Sim3EstimationOptions {
  size_t min_point_pairs = 12;
  double max_reproj_error_px = 8.0;
  size_t min_track_length = 3;
  double max_depth_ratio = 10.0;
  // 如果小于0，则自动使用深度中间值乘以 max_depth_ratio 作为阈值
  double ransac_max_error = -1.0;
  double ransac_min_inlier_ratio = 0.25;
};

struct PipelineOptions {
  std::string sparse_path;
  std::string database_path;
  std::string image_path;
  std::string output_path;
  std::string edge_input_path;
  std::string edge_output_path;
  std::string manual_loop_path;
  std::string rig_config_path;
  std::string gt_reconstruction_path;  // 真值重建路径（可选，用于评估回环误差）
  bool build_odom_edges = true;
  bool build_loop_edges = true;
  bool run_self_test = false;
  size_t odom_window = 5;
  double eval_threshold_px = -1.0;
  double min_loop_time_gap = 100.0;
  size_t min_loop_inliers = 200;
  double min_loop_inlier_ratio = 0.2;
  double max_loop_geom_error_px = 4.0;
  double min_loop_tri_angle_deg = 5.0;
  size_t max_loop_edges_per_image = 30;
  EdgeDefaults defaults;
  Sim3EstimationOptions sim3_options;
  int stage1_max_iterations = 2;
  double stage1_tri_max_project_error = 18.0;
  double stage1_filter_max_reproj_error = 18.0;
  int stage2_max_iterations = 2;
  double stage2_tri_max_project_error = 4.0;
  double stage2_filter_max_reproj_error = 4.0;
  int stage3_max_iterations = 3;
  double stage3_tri_max_project_error = 4.0;
  double stage3_filter_max_reproj_error = 4.0;
};

bool IsNumericToken(const std::string& token) {
  char* end_ptr = nullptr;
  std::strtod(token.c_str(), &end_ptr);
  return end_ptr != token.c_str() && *end_ptr == '\0';
}

std::string GetBaseName(const std::string& name) {
  const auto pos = name.find_last_of("/\\");
  if (pos == std::string::npos) {
    return name;
  }
  return name.substr(pos + 1);
}

std::string StripExtension(const std::string& name) {
  const auto pos = name.find_last_of('.');
  if (pos == std::string::npos || pos == 0) {
    return name;
  }
  return name.substr(0, pos);
}

bool TryParseImageTimestamp(const std::string& name, double* timestamp) {
  if (timestamp == nullptr) {
    return false;
  }
  const std::string base = StripExtension(GetBaseName(name));
  size_t end = base.size();
  size_t start = end;
  bool has_digit = false;
  bool has_dot = false;
  while (start > 0) {
    const unsigned char ch = static_cast<unsigned char>(base[start - 1]);
    if (std::isdigit(ch)) {
      has_digit = true;
      --start;
      continue;
    }
    if (ch == '.' && !has_dot) {
      has_dot = true;
      --start;
      continue;
    }
    break;
  }
  if (!has_digit || start == end) {
    return false;
  }
  const std::string token = base.substr(start, end - start);
  char* end_ptr = nullptr;
  const double parsed = std::strtod(token.c_str(), &end_ptr);
  if (end_ptr == token.c_str() || *end_ptr != '\0' || !std::isfinite(parsed)) {
    return false;
  }
  *timestamp = parsed;
  return true;
}

std::vector<image_t> GetOrderedRegImageIdsByTime(
    const Reconstruction& reconstruction) {
  std::vector<image_t> ordered = reconstruction.RegImageIds();
  std::unordered_map<image_t, double> time_by_image;
  time_by_image.reserve(ordered.size());
  for (const auto image_id : ordered) {
    const auto& image = reconstruction.Image(image_id);
    double t = 0.0;
    if (TryParseImageTimestamp(image.Name(), &t)) {
      time_by_image[image_id] = t;
    } else {
      time_by_image[image_id] = static_cast<double>(image_id);
    }
  }
  std::sort(ordered.begin(), ordered.end(),
            [&](const image_t a, const image_t b) {
              const double ta = time_by_image[a];
              const double tb = time_by_image[b];
              if (ta == tb) {
                return a < b;
              }
              return ta < tb;
            });
  return ordered;
}

std::vector<std::vector<image_t>> BuildOverlappingClusters(
    const std::vector<image_t>& ordered_ids,
    const size_t cluster_size, const size_t overlap) {
  std::vector<std::vector<image_t>> clusters;
  if (ordered_ids.empty()) {
    return clusters;
  }
  const size_t effective_cluster_size = std::max<size_t>(2, cluster_size);
  const size_t effective_overlap =
      std::min(overlap, effective_cluster_size - 1);
  const size_t step = effective_cluster_size - effective_overlap;
  for (size_t start = 0; start < ordered_ids.size(); start += step) {
    const size_t end =
        std::min(start + effective_cluster_size, ordered_ids.size());
    if (end <= start) {
      break;
    }
    std::vector<image_t> cluster(ordered_ids.begin() + start,
                                 ordered_ids.begin() + end);
    clusters.push_back(std::move(cluster));
    if (end == ordered_ids.size()) {
      break;
    }
  }
  return clusters;
}

std::vector<std::vector<image_t>> BuildHierarchicalClusters(
    const Reconstruction& reconstruction, const Database& database,
    const SceneClustering::Options& clustering_options) {
  std::vector<std::vector<image_t>> clusters;

  SceneClustering scene_clustering =
      SceneClustering::Create(clustering_options, database);
  const auto leaf_clusters = scene_clustering.GetLeafClusters();

  const auto reg_ids = reconstruction.RegImageIds();
  std::unordered_set<image_t> reg_set(reg_ids.begin(), reg_ids.end());

  for (const auto* cluster : leaf_clusters) {
    if (cluster == nullptr) {
      continue;
    }
    std::unordered_set<image_t> unique_ids;
    unique_ids.reserve(cluster->image_ids.size());
    for (const auto image_id : cluster->image_ids) {
      if (reg_set.count(image_id) > 0) {
        unique_ids.insert(image_id);
      }
    }
    if (unique_ids.size() < 2) {
      continue;
    }
    std::vector<image_t> ids(unique_ids.begin(), unique_ids.end());
    std::sort(ids.begin(), ids.end());
    clusters.push_back(std::move(ids));
  }

  return clusters;
}

std::string EdgeTypeToString(const EdgeType type) {
  if (type == EdgeType::kOdom) {
    return "odom";
  }
  if (type == EdgeType::kRig) {
    return "rig";
  }
  return "loop";
}

bool ParsePtreeVector(const boost::property_tree::ptree& node,
                      const size_t expected_size,
                      std::vector<double>* values) {
  if (values == nullptr) {
    return false;
  }
  values->clear();
  values->reserve(expected_size);
  for (const auto& child : node) {
    values->push_back(child.second.get_value<double>());
  }
  return values->size() == expected_size;
}

/**
 * [功能描述]：从JSON文件读取相机rig配置（与RigBundleAdjuster格式一致）
 * @param path：JSON配置文件的路径
 * @param reconstruction：用于解析快照分组与估计相对位姿
 * @param camera_rigs：输出参数，解析后的相机rig列表
 * @param error：输出参数，解析失败时的错误信息
 * @return 解析成功返回true，失败返回false
 */
bool ReadCameraRigConfigFromFile(const std::string& path,
                                 const Reconstruction& reconstruction,
                                 std::vector<CameraRig>* camera_rigs,
                                 std::string* error) {
  // 检查输出参数是否为空
  if (camera_rigs == nullptr) {
    if (error != nullptr) {
      *error = "camera rigs output is null";
    }
    return false;
  }
  camera_rigs->clear();

  // 使用boost解析JSON文件
  boost::property_tree::ptree pt;
  try {
    boost::property_tree::read_json(path.c_str(), pt);
  } catch (const std::exception& e) {
    if (error != nullptr) {
      *error = e.what();
    }
    return false;
  }

  // 遍历配置文件中的每个相机rig配置
  for (const auto& rig_config : pt) {
    CameraRig camera_rig;
    bool estimate_rig_relative_poses = false;  // 标记是否需要估计相对位姿

    // 获取cameras节点
    std::vector<std::string> image_prefixes;
    const auto cameras_node = rig_config.second.get_child_optional("cameras");
    if (!cameras_node) {
      if (error != nullptr) {
        *error = "rig config missing cameras";
      }
      return false;
    }

    // 遍历每个相机配置
    for (const auto& camera : cameras_node.get()) {
      // 读取相机ID
      const camera_t camera_id =
          camera.second.get<camera_t>("camera_id", kInvalidCameraId);
      if (camera_id == kInvalidCameraId) {
        if (error != nullptr) {
          *error = "rig config has invalid camera_id";
        }
        return false;
      }
      // 检查相机ID是否重复
      if (camera_rig.HasCamera(camera_id)) {
        if (error != nullptr) {
          *error = "rig config has duplicate camera_id";
        }
        return false;
      }

      // 读取图像前缀（用于匹配属于该相机的图像）
      const std::string image_prefix =
          camera.second.get<std::string>("image_prefix", "");
      if (image_prefix.empty()) {
        if (error != nullptr) {
          *error = "rig config missing image_prefix";
        }
        return false;
      }
      image_prefixes.push_back(image_prefix);

      // 初始化相对平移向量和旋转四元数
      Eigen::Vector3d rel_tvec = Eigen::Vector3d::Zero();
      Eigen::Vector4d rel_qvec = ComposeIdentityQuaternion();

      // 解析相对平移向量rel_tvec（如果存在）
      const auto rel_tvec_node = camera.second.get_child_optional("rel_tvec");
      if (rel_tvec_node) {
        std::vector<double> tvec;
        if (!ParsePtreeVector(rel_tvec_node.get(), 3, &tvec)) {
          if (error != nullptr) {
            *error = "rig config invalid rel_tvec size";
          }
          return false;
        }
        rel_tvec = Eigen::Vector3d(tvec[0], tvec[1], tvec[2]);
      } else {
        estimate_rig_relative_poses = true;  // 缺少平移向量，需要后续估计
      }

      // 解析相对旋转四元数rel_qvec（如果存在）
      const auto rel_qvec_node = camera.second.get_child_optional("rel_qvec");
      if (rel_qvec_node) {
        std::vector<double> qvec;
        if (!ParsePtreeVector(rel_qvec_node.get(), 4, &qvec)) {
          if (error != nullptr) {
            *error = "rig config invalid rel_qvec size";
          }
          return false;
        }
        // 构建四元数并归一化
        Eigen::Quaterniond q(qvec[0], qvec[1], qvec[2], qvec[3]);
        if (q.norm() > 1e-12) {
          q.normalize();
        } else {
          q = Eigen::Quaterniond::Identity();
        }
        rel_qvec = Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
      } else {
        estimate_rig_relative_poses = true;  // 缺少旋转四元数，需要后续估计
      }

      // 将相机添加到rig中
      camera_rig.AddCamera(camera_id, rel_qvec, rel_tvec);
    }

    // 设置参考相机ID
    const camera_t ref_camera_id =
        rig_config.second.get<camera_t>("ref_camera_id", kInvalidCameraId);
    if (ref_camera_id == kInvalidCameraId ||
        !camera_rig.HasCamera(ref_camera_id)) {
      if (error != nullptr) {
        *error = "rig config has invalid ref_camera_id";
      }
      return false;
    }
    camera_rig.SetRefCameraId(ref_camera_id);

    // 根据图像前缀将图像分组为快照(snapshots)
    // 快照是指同一时刻由多个相机拍摄的图像集合
    std::unordered_map<std::string, std::vector<image_t>> snapshots;
    for (const auto image_id : reconstruction.RegImageIds()) {
      const auto& image = reconstruction.Image(image_id);
      for (const auto& image_prefix : image_prefixes) {
        if (StringContains(image.Name(), image_prefix)) {
          // 提取图像后缀作为快照的key
          const std::string image_suffix =
              StringGetAfter(image.Name(), image_prefix);
          snapshots[image_suffix].push_back(image_id);
        }
      }
    }

    // 只添加包含参考相机的快照
    for (const auto& snapshot : snapshots) {
      bool has_ref_camera = false;
      for (const auto image_id : snapshot.second) {
        const auto& image = reconstruction.Image(image_id);
        if (image.CameraId() == camera_rig.RefCameraId()) {
          has_ref_camera = true;
        }
      }

      if (has_ref_camera) {
        camera_rig.AddSnapshot(snapshot.second);
      }
    }

    // 验证相机rig配置的一致性
    camera_rig.Check(reconstruction);

    // 如果配置中缺少相对位姿信息，则从重建结果中估计
    if (estimate_rig_relative_poses) {
      if (camera_rig.NumSnapshots() == 0) {
        if (error != nullptr) {
          *error =
              "no rig snapshots with reference camera to estimate poses";
        }
        return false;
      }
      PrintHeading2("Estimating relative rig poses");
      if (!camera_rig.ComputeRelativePoses(reconstruction)) {
        if (error != nullptr) {
          *error =
              "failed to estimate rig poses from reconstruction for rig config";
        }
        return false;
      }
    }

    camera_rigs->push_back(camera_rig);
  }

  // 确保至少解析出一个相机rig配置
  if (camera_rigs->empty()) {
    if (error != nullptr) {
      *error = "no rig configs parsed";
    }
    return false;
  }
  return true;
}

/**
 * [功能描述]：解析一行文本，提取位姿图边(Edge)的信息
 * 支持两种格式：
 *   1. 带类型: type(odom/loop/rig) id1 id2 qw qx qy qz tx ty tz [is_unit rot_weight trans_weight]
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
  if (type_token == "odom") {
    parsed.type = EdgeType::kOdom;
  } else if (type_token == "rig") {
    parsed.type = EdgeType::kRig;
  } else {
    parsed.type = EdgeType::kLoop;
  }
  parsed.image_id1 = static_cast<image_t>(id1);
  parsed.image_id2 = static_cast<image_t>(id2);
  parsed.q_ij = Eigen::Quaterniond(qw, qx, qy, qz);
  // 对四元数进行归一化（避免数值误差）
  if (parsed.q_ij.norm() > 1e-12) {
    parsed.q_ij.normalize();
  }
  Eigen::Vector3d t_ij = Eigen::Vector3d(tx, ty, tz);
  parsed.t_ij = t_ij;

  // 根据边类型设置默认权重参数
  if (parsed.type == EdgeType::kOdom) {
    parsed.translation_is_unit = defaults.odom_translation_is_unit;
    parsed.rot_weight = defaults.odom_rot_weight;
    parsed.trans_weight = defaults.odom_trans_weight;
  } else if (parsed.type == EdgeType::kRig) {
    parsed.translation_is_unit = defaults.rig_translation_is_unit;
    parsed.rot_weight = defaults.rig_rot_weight;
    parsed.trans_weight = defaults.rig_trans_weight;
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
 * @param options：Sim3估计参数（过滤与RANSAC阈值）
 * @param scale：输出参数，估计得到的尺度
 * @param q_ij_out：输出参数，估计得到的相对旋转四元数
 * @param t_ij_out：输出参数，估计得到的相对平移
 * @return 估计成功返回true，否则返回false
 */
bool EstimateLoopScaleFromPointPairsSim3(
    const Reconstruction& reconstruction, const TwoViewGeometry& geom,
    const Image& image_i, const Image& image_j,
    const Sim3EstimationOptions& options, double* scale,
    Eigen::Quaterniond& q_ij_out, Eigen::Vector3d& t_ij_out) {
  if (scale == nullptr) {
    return false;
  }

  const size_t kMinPointPairs = std::max<size_t>(1, options.min_point_pairs);
  const double kMinDepth = 1e-6;  // 最小深度阈值
  const double kMinAbsTransCos = 0.2;  // 最小绝对平移余弦值
  const double kMaxRotDiffRad = DegToRad(30.0);  // 最大旋转差异阈值
  const double kMaxReprojErrorPx =
      std::max(1e-12, options.max_reproj_error_px);
  const size_t kMinTrackLength = std::max<size_t>(1, options.min_track_length);
  const double kMaxDepthRatio = std::max(1.0, options.max_depth_ratio);

  // 获取两幅图像对应的相机参数（用于计算重投影误差）
  const Camera& camera_i = reconstruction.Camera(image_i.CameraId());
  const Camera& camera_j = reconstruction.Camera(image_j.CameraId());

  // 获取两幅图像的位姿参数
  const Eigen::Vector4d qvec_i = image_i.Qvec();
  const Eigen::Vector4d qvec_j = image_j.Qvec();
  const Eigen::Vector3d tvec_i = image_i.Tvec();
  const Eigen::Vector3d tvec_j = image_j.Tvec();

  // 获取两幅图像的旋转和平移矩阵
  const Eigen::Matrix3d R_i = image_i.RotationMatrix();
  const Eigen::Vector3d t_i = image_i.Tvec();
  const Eigen::Matrix3d R_j = image_j.RotationMatrix();
  const Eigen::Vector3d t_j = image_j.Tvec();

  // 第一次遍历：收集所有候选点并计算深度，用于计算中值深度
  std::vector<double> all_depths;
  all_depths.reserve(geom.inlier_matches.size() * 2);

  for (const auto& match : geom.inlier_matches) {
    if (match.point2D_idx1 >= image_i.NumPoints2D() ||
        match.point2D_idx2 >= image_j.NumPoints2D()) {
      continue;
    }

    const Point2D& point2D_i = image_i.Point2D(match.point2D_idx1);
    const Point2D& point2D_j = image_j.Point2D(match.point2D_idx2);
    if (!point2D_i.HasPoint3D() || !point2D_j.HasPoint3D()) {
      continue;
    }

    const point3D_t point3D_id_i = point2D_i.Point3DId();
    const point3D_t point3D_id_j = point2D_j.Point3DId();
    if (!reconstruction.ExistsPoint3D(point3D_id_i) ||
        !reconstruction.ExistsPoint3D(point3D_id_j)) {
      continue;
    }

    const Point3D& point3D_i = reconstruction.Point3D(point3D_id_i);
    const Point3D& point3D_j = reconstruction.Point3D(point3D_id_j);

    // 计算3D点在相机坐标系下的坐标，获取深度
    const Eigen::Vector3d xyz_i = R_i * point3D_i.XYZ() + t_i;
    const Eigen::Vector3d xyz_j = R_j * point3D_j.XYZ() + t_j;

    if (xyz_i.allFinite() && xyz_i.z() > kMinDepth) {
      all_depths.push_back(xyz_i.z());
    }
    if (xyz_j.allFinite() && xyz_j.z() > kMinDepth) {
      all_depths.push_back(xyz_j.z());
    }
  }

  // 计算中值深度，用于后续的极远点过滤
  if (all_depths.size() < kMinPointPairs * 2) {
    return false;
  }
  const double median_depth = Median(all_depths);
  if (!std::isfinite(median_depth) || median_depth <= kMinDepth) {
    return false;
  }
  // 极远点深度阈值
  const double max_depth = median_depth * kMaxDepthRatio;

  // 提取匹配点对应的3D点（第二次遍历，应用所有过滤条件）
  std::vector<Eigen::Vector3d> points_i;
  std::vector<Eigen::Vector3d> points_j;
  points_i.reserve(geom.inlier_matches.size());
  points_j.reserve(geom.inlier_matches.size());

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

    // 获取匹配点对应的3D点ID
    const point3D_t point3D_id_i = point2D_i.Point3DId();
    const point3D_t point3D_id_j = point2D_j.Point3DId();
    if (!reconstruction.ExistsPoint3D(point3D_id_i) ||
        !reconstruction.ExistsPoint3D(point3D_id_j)) {
      continue;
    }

    // 获取3D点
    const Point3D& point3D_i = reconstruction.Point3D(point3D_id_i);
    const Point3D& point3D_j = reconstruction.Point3D(point3D_id_j);

    // 过滤1：检查3D点的观测数量（track length）
    if (point3D_i.Track().Length() < kMinTrackLength ||
        point3D_j.Track().Length() < kMinTrackLength) {
      continue;
    }

    // 过滤2：计算并检查重投影误差
    // 将3D点投影到对应图像上，计算与观测2D点的误差
    const Eigen::Vector3d proj_i = QuaternionRotatePoint(qvec_i, point3D_i.XYZ()) + tvec_i;
    const Eigen::Vector3d proj_j = QuaternionRotatePoint(qvec_j, point3D_j.XYZ()) + tvec_j;

    // 确保点在相机前方
    if (proj_i.z() < std::numeric_limits<double>::epsilon() ||
        proj_j.z() < std::numeric_limits<double>::epsilon()) {
      continue;
    }

    // 投影到图像平面并计算重投影误差
    const Eigen::Vector2d proj_point2D_i = camera_i.WorldToImage(proj_i.hnormalized());
    const Eigen::Vector2d proj_point2D_j = camera_j.WorldToImage(proj_j.hnormalized());
    const double reproj_error_i = (proj_point2D_i - point2D_i.XY()).norm();
    const double reproj_error_j = (proj_point2D_j - point2D_j.XY()).norm();

    if (reproj_error_i > kMaxReprojErrorPx || reproj_error_j > kMaxReprojErrorPx) {
      continue;
    }

    // 计算3D点在两视图相机坐标系下的坐标
    const Eigen::Vector3d xyz_i = R_i * point3D_i.XYZ() + t_i;
    const Eigen::Vector3d xyz_j = R_j * point3D_j.XYZ() + t_j;
    if (!xyz_i.allFinite() || !xyz_j.allFinite()) {
      continue;
    }

    // 过滤3：检查深度是否有效（最小深度）
    if (xyz_i.z() <= kMinDepth || xyz_j.z() <= kMinDepth) {
      continue;
    }

    // 过滤4：过滤极远点（深度大于中值深度的kMaxDepthRatio倍）
    if (xyz_i.z() > max_depth || xyz_j.z() > max_depth) {
      continue;
    }

    points_i.push_back(xyz_i);
    points_j.push_back(xyz_j);
  }

  // 检查是否满足最少匹配点数要求
  if (points_i.size() < kMinPointPairs) {
    return false;
  }

  // 计算匹配点对应的3D点在两视图相机坐标系下的尺度（用于设置RANSAC误差阈值）
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

  // 设置LORANSAC参数
  RANSACOptions ransac_options;
  if (options.ransac_max_error > 0.0) {
    ransac_options.max_error = options.ransac_max_error;
  } else {
    ransac_options.max_error =
        std::max(1e-12, std::pow(0.05 * median_norm, 2));
  }
  ransac_options.min_inlier_ratio =
      std::max(1e-6, options.ransac_min_inlier_ratio);
  ransac_options.confidence = 0.999;
  ransac_options.min_num_trials = 100;
  ransac_options.max_num_trials = 5000;  // 增加最大迭代次数

  // 使用LORANSAC进行鲁棒估计（局部优化RANSAC）
  using Sim3Estimator = SimilarityTransformEstimator<3, true>;
  LORANSAC<Sim3Estimator, Sim3Estimator> loransac(ransac_options);
  const auto report = loransac.Estimate(points_i, points_j);
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

  // 使用所有内点重新估计Sim3变换
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

  // 计算相对平移的模长
  const double t_norm = t.norm();
  // 检查相对平移的模长是否满足最小深度阈值
  if (!std::isfinite(t_norm) || t_norm <= kMinDepth) {
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
 * [功能描述]：回环相对位姿误差记录结构体
 */
struct LoopPoseError {
  image_t image_id1;               // 图像1的ID
  image_t image_id2;               // 图像2的ID
  std::string image_name1;         // 图像1的名称
  std::string image_name2;         // 图像2的名称
  double rot_error_deg;            // 旋转误差（度）
  double trans_error;              // 平移误差（米）
  double trans_error_normalized;   // 归一化平移误差（方向误差角度，度）
  Eigen::Quaterniond q_est;        // 估计的相对旋转
  Eigen::Vector3d t_est;           // 估计的相对平移
  Eigen::Quaterniond q_gt;         // 真值相对旋转
  Eigen::Vector3d t_gt;            // 真值相对平移
};

/**
 * [功能描述]：从真值重建中计算两幅图像的相对位姿
 * @param gt_reconstruction：真值重建
 * @param image_name_i：图像i的名称
 * @param image_name_j：图像j的名称
 * @param q_ij_out：输出参数，从i到j的相对旋转（四元数）
 * @param t_ij_out：输出参数，从i到j的相对平移
 * @return 成功返回true，失败返回false
 */
bool ComputeRelativePoseFromGT(
    const Reconstruction& gt_reconstruction,
    const std::string& image_name_i,
    const std::string& image_name_j,
    Eigen::Quaterniond* q_ij_out,
    Eigen::Vector3d* t_ij_out) {
  if (q_ij_out == nullptr || t_ij_out == nullptr) {
    return false;
  }

  // 在真值重建中查找对应的图像
  const Image* gt_image_i = nullptr;
  const Image* gt_image_j = nullptr;
  for (const auto& pair : gt_reconstruction.Images()) {
    const std::string& name = pair.second.Name();
    if (name == image_name_i) {
      gt_image_i = &pair.second;
    }
    if (name == image_name_j) {
      gt_image_j = &pair.second;
    }
    if (gt_image_i != nullptr && gt_image_j != nullptr) {
      break;
    }
  }

  if (gt_image_i == nullptr || gt_image_j == nullptr) {
    return false;
  }

  // 检查图像是否已注册
  if (!gt_image_i->IsRegistered() || !gt_image_j->IsRegistered()) {
    return false;
  }

  // 计算相对位姿
  Eigen::Vector4d qvec_ij;
  Eigen::Vector3d tvec_ij;
  ComputeRelativePose(gt_image_i->Qvec(), gt_image_i->Tvec(),
                      gt_image_j->Qvec(), gt_image_j->Tvec(),
                      &qvec_ij, &tvec_ij);

  *q_ij_out = Eigen::Quaterniond(qvec_ij(0), qvec_ij(1), qvec_ij(2), qvec_ij(3));
  *t_ij_out = tvec_ij;
  return true;
}

/**
 * [功能描述]：计算两个旋转之间的角度误差
 * @param q1：旋转1（四元数）
 * @param q2：旋转2（四元数）
 * @return 角度误差（度）
 */
double ComputeRotationError(const Eigen::Quaterniond& q1,
                            const Eigen::Quaterniond& q2) {
  // 计算相对旋转
  const Eigen::Quaterniond dq = q1.conjugate() * q2;
  // 计算旋转角度
  const double angle_rad = 2.0 * std::acos(std::min(1.0, std::abs(dq.w())));
  return RadToDeg(angle_rad);
}

/**
 * [功能描述]：计算两个平移向量之间的方向误差（角度）
 * @param t1：平移向量1
 * @param t2：平移向量2
 * @return 方向误差（度）
 */
double ComputeTranslationDirectionError(const Eigen::Vector3d& t1,
                                        const Eigen::Vector3d& t2) {
  const double norm1 = t1.norm();
  const double norm2 = t2.norm();
  if (norm1 < 1e-12 || norm2 < 1e-12) {
    return 180.0;  // 平移向量太小，返回最大误差
  }
  const double cos_angle = t1.dot(t2) / (norm1 * norm2);
  const double angle_rad = std::acos(std::max(-1.0, std::min(1.0, cos_angle)));
  return RadToDeg(angle_rad);
}

/**
 * [功能描述]：计算回环相对位姿与真值之间的误差
 * @param gt_reconstruction：真值重建
 * @param image_i：当前重建中的图像i
 * @param image_j：当前重建中的图像j
 * @param q_ij_est：估计的相对旋转
 * @param t_ij_est：估计的相对平移
 * @param error_out：输出参数，误差记录
 * @return 成功返回true，失败返回false
 */
bool ComputeLoopPoseError(
    const Reconstruction& gt_reconstruction,
    const Image& image_i,
    const Image& image_j,
    const Eigen::Quaterniond& q_ij_est,
    const Eigen::Vector3d& t_ij_est,
    LoopPoseError* error_out) {
  if (error_out == nullptr) {
    return false;
  }

  // 从真值重建中计算相对位姿
  Eigen::Quaterniond q_ij_gt;
  Eigen::Vector3d t_ij_gt;
  if (!ComputeRelativePoseFromGT(gt_reconstruction, image_i.Name(),
                                  image_j.Name(), &q_ij_gt, &t_ij_gt)) {
    return false;
  }

  // 填充误差记录
  error_out->image_id1 = image_i.ImageId();
  error_out->image_id2 = image_j.ImageId();
  error_out->image_name1 = image_i.Name();
  error_out->image_name2 = image_j.Name();
  error_out->q_est = q_ij_est;
  error_out->t_est = t_ij_est;
  error_out->q_gt = q_ij_gt;
  error_out->t_gt = t_ij_gt;

  // 计算旋转误差
  error_out->rot_error_deg = ComputeRotationError(q_ij_est, q_ij_gt);

  // 计算平移误差（欧氏距离）
  error_out->trans_error = (t_ij_est - t_ij_gt).norm();

  // 计算平移方向误差
  error_out->trans_error_normalized =
      ComputeTranslationDirectionError(t_ij_est, t_ij_gt);

  return true;
}

/**
 * [功能描述]：将回环相对位姿误差写入文件
 * @param path：输出文件路径
 * @param errors：误差记录列表
 * @param error：输出参数，错误信息
 * @return 成功返回true，失败返回false
 */
bool WriteLoopPoseErrorsToFile(const std::string& path,
                               const std::vector<LoopPoseError>& errors,
                               std::string* error) {
  std::ofstream file(path);
  if (!file.is_open()) {
    if (error != nullptr) {
      *error = "Failed to open file: " + path;
    }
    return false;
  }

  // 写入表头
  file << "# Loop Pose Errors\n";
  file << "# image_id1, image_id2, image_name1, image_name2, "
       << "rot_error_deg, trans_error, trans_dir_error_deg, "
       << "q_est_w, q_est_x, q_est_y, q_est_z, t_est_x, t_est_y, t_est_z, "
       << "q_gt_w, q_gt_x, q_gt_y, q_gt_z, t_gt_x, t_gt_y, t_gt_z\n";

  file << std::fixed << std::setprecision(6);

  // 写入每条误差记录
  for (const auto& err : errors) {
    file << err.image_id1 << ", " << err.image_id2 << ", "
         << err.image_name1 << ", " << err.image_name2 << ", "
         << err.rot_error_deg << ", " << err.trans_error << ", "
         << err.trans_error_normalized << ", "
         << err.q_est.w() << ", " << err.q_est.x() << ", "
         << err.q_est.y() << ", " << err.q_est.z() << ", "
         << err.t_est.x() << ", " << err.t_est.y() << ", " << err.t_est.z() << ", "
         << err.q_gt.w() << ", " << err.q_gt.x() << ", "
         << err.q_gt.y() << ", " << err.q_gt.z() << ", "
         << err.t_gt.x() << ", " << err.t_gt.y() << ", " << err.t_gt.z() << "\n";
  }

  // 写入统计信息
  if (!errors.empty()) {
    double sum_rot = 0.0, sum_trans = 0.0, sum_trans_dir = 0.0;
    double max_rot = 0.0, max_trans = 0.0, max_trans_dir = 0.0;
    for (const auto& err : errors) {
      sum_rot += err.rot_error_deg;
      sum_trans += err.trans_error;
      sum_trans_dir += err.trans_error_normalized;
      max_rot = std::max(max_rot, err.rot_error_deg);
      max_trans = std::max(max_trans, err.trans_error);
      max_trans_dir = std::max(max_trans_dir, err.trans_error_normalized);
    }
    const size_t n = errors.size();
    file << "# Statistics:\n";
    file << "# Total loop edges: " << n << "\n";
    file << "# Mean rot error (deg): " << (sum_rot / n) << "\n";
    file << "# Mean trans error: " << (sum_trans / n) << "\n";
    file << "# Mean trans dir error (deg): " << (sum_trans_dir / n) << "\n";
    file << "# Max rot error (deg): " << max_rot << "\n";
    file << "# Max trans error: " << max_trans << "\n";
    file << "# Max trans dir error (deg): " << max_trans_dir << "\n";
  }

  file.close();
  return true;
}

/**
 * [功能描述]：评估回环边相对位姿与真值的误差，并输出到文件
 * @param reconstruction：当前重建结果（用于计算相对位姿）
 * @param gt_reconstruction：真值重建
 * @param edges：位姿图边列表
 * @param output_path：输出文件路径
 * @return 成功返回true，失败返回false
 */
bool EvaluateAndWriteLoopPoseErrors(
    const Reconstruction& reconstruction,
    const Reconstruction& gt_reconstruction,
    const std::vector<PoseGraphEdge>& edges,
    const std::string& output_path) {
  std::vector<LoopPoseError> loop_errors;

  // 遍历所有边，只处理回环边
  for (const auto& edge : edges) {
    if (edge.type != EdgeType::kLoop) {
      continue;
    }

    // 检查图像是否存在
    if (!reconstruction.ExistsImage(edge.image_id1) ||
        !reconstruction.ExistsImage(edge.image_id2)) {
      continue;
    }

    const Image& image_i = reconstruction.Image(edge.image_id1);
    const Image& image_j = reconstruction.Image(edge.image_id2);

    // 从重建中计算相对位姿
    Eigen::Vector4d qvec_ij;
    Eigen::Vector3d tvec_ij;
    ComputeRelativePose(image_i.Qvec(), image_i.Tvec(),
                        image_j.Qvec(), image_j.Tvec(),
                        &qvec_ij, &tvec_ij);
    Eigen::Quaterniond q_ij(qvec_ij(0), qvec_ij(1), qvec_ij(2), qvec_ij(3));

    // 计算与真值的误差
    LoopPoseError error;
    if (ComputeLoopPoseError(gt_reconstruction, image_i, image_j,
                             q_ij, tvec_ij, &error)) {
      loop_errors.push_back(error);
    }
  }

  // 如果没有有效的误差记录，返回
  if (loop_errors.empty()) {
    std::cout << "No valid loop edges for GT comparison.\n";
    return false;
  }

  // 输出误差文件
  std::string write_error;
  if (!WriteLoopPoseErrorsToFile(output_path, loop_errors, &write_error)) {
    std::cout << "Failed to write loop pose errors: " << write_error << "\n";
    return false;
  }

  // 计算并输出统计信息
  double sum_rot = 0.0, sum_trans = 0.0, sum_trans_dir = 0.0;
  for (const auto& err : loop_errors) {
    sum_rot += err.rot_error_deg;
    sum_trans += err.trans_error;
    sum_trans_dir += err.trans_error_normalized;
  }
  const size_t n = loop_errors.size();

  std::cout << "Wrote loop pose errors to: " << output_path << "\n";
  std::cout << "Loop pose error statistics:\n";
  std::cout << "  Total loop edges with GT: " << n << "\n";
  std::cout << "  Mean rotation error: " << (sum_rot / n) << " deg\n";
  std::cout << "  Mean translation error: " << (sum_trans / n) << " m\n";
  std::cout << "  Mean translation direction error: " << (sum_trans_dir / n) << " deg\n";

  return true;
}

/**
 * [功能描述]：从重建结果构建里程计边（按快照排序，连接后续N帧）
 * @param reconstruction：3D重建结果
 * @param defaults：边的默认参数配置
 * @param odom_window：每帧连接的后续帧数（至少为1）
 * @return 里程计边列表
 */
std::vector<PoseGraphEdge> BuildOdometryEdges(
    const Reconstruction& reconstruction, const EdgeDefaults& defaults,
    const size_t odom_window) {
  struct SnapshotEntry {
    image_t image_id = kInvalidImageId;
    std::string name;
    bool has_number = false;
    double number = 0.0;
  };

  const size_t window = std::max<size_t>(1, odom_window);

  // 按相机ID分组图像
  std::unordered_map<camera_t, std::vector<SnapshotEntry>> camera_entries;
  camera_entries.reserve(reconstruction.Cameras().size());

  // 遍历重建中的所有图像，构建每个相机的图像列表
  for (const auto image_id : reconstruction.RegImageIds()) {
    const Image& image = reconstruction.Image(image_id);
    SnapshotEntry entry;
    entry.image_id = image_id;
    entry.name = image.Name();
    double value = 0.0;
    entry.has_number = TryParseImageTimestamp(entry.name, &value);
    entry.number = value;
    camera_entries[image.CameraId()].push_back(std::move(entry));
  }

  std::vector<PoseGraphEdge> edges;
  // 遍历每个相机的图像序列
  for (auto& pair : camera_entries) {
    auto& entries = pair.second;
    std::sort(entries.begin(), entries.end(),
              [](const SnapshotEntry& a, const SnapshotEntry& b) {
                if (a.has_number && b.has_number) {
                  if (a.number != b.number) {
                    return a.number < b.number;
                  }
                } else if (a.name != b.name) {
                  return a.name < b.name;
                }
                if (a.name != b.name) {
                  return a.name < b.name;
                }
                return a.image_id < b.image_id;
              });  // 按快照排序
    if (entries.size() < 2) {
      continue;  // 至少需要2帧才能构建边
    }

    // 构建每帧与后续window帧之间的里程计边
    for (size_t idx = 0; idx + 1 < entries.size(); ++idx) {
      const image_t image_id_i = entries[idx].image_id;
      const Image& image_i = reconstruction.Image(image_id_i);

      for (size_t step = 1; step <= window && idx + step < entries.size();
           ++step) {
        const image_t image_id_j = entries[idx + step].image_id;
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
 * [功能描述]：从相机rig配置构建位姿图的rig边
 *            rig边表示同一时刻多相机系统中不同相机之间的相对位姿约束
 * @param reconstruction：三维重建结果，包含已注册的图像信息
 * @param camera_rigs：相机rig配置列表，定义多相机系统的结构
 * @param defaults：边的默认参数配置
 * @param occupied_pairs：已占用的图像对集合（输入/输出），用于避免重复创建边
 * @return 构建的rig边列表
 */
std::vector<PoseGraphEdge> BuildRigEdges(
    const Reconstruction& reconstruction,
    const std::vector<CameraRig>& camera_rigs, const EdgeDefaults& defaults,
    std::unordered_set<image_pair_t>* occupied_pairs) {
  std::vector<PoseGraphEdge> edges;
  if (camera_rigs.empty()) {
    return edges;
  }

  // 遍历所有相机rig配置
  for (const auto& camera_rig : camera_rigs) {
    // 遍历rig的所有快照（同一时刻多相机拍摄的图像组）
    for (const auto& snapshot : camera_rig.Snapshots()) {
      if (snapshot.size() < 2) {
        continue;
      }

      // 建立相机ID到图像ID的映射，每个相机只保留最小的image_id
      std::unordered_map<camera_t, image_t> images_by_camera;
      images_by_camera.reserve(snapshot.size());
      for (const auto image_id : snapshot) {
        const auto& image = reconstruction.Image(image_id);
        const camera_t camera_id = image.CameraId();
        if (!camera_rig.HasCamera(camera_id)) {
          continue;
        }
        auto it = images_by_camera.find(camera_id);
        if (it == images_by_camera.end() || image_id < it->second) {
          images_by_camera[camera_id] = image_id;
        }
      }

      if (images_by_camera.size() < 2) {
        continue;
      }

      // 收集当前快照中存在的相机ID
      std::vector<camera_t> cameras_present;
      cameras_present.reserve(images_by_camera.size());
      for (const auto& entry : images_by_camera) {
        cameras_present.push_back(entry.first);
      }

      // 对快照内的相机两两配对，构建rig约束边
      for (size_t i = 0; i + 1 < cameras_present.size(); ++i) {
        const camera_t cam_i_id = cameras_present[i];
        const image_t image_i_id = images_by_camera.at(cam_i_id);

        // 获取相机i相对于rig参考坐标系的位姿
        const Eigen::Vector4d& qvec_i = camera_rig.RelativeQvec(cam_i_id);
        const Eigen::Vector3d& tvec_i = camera_rig.RelativeTvec(cam_i_id);
        Eigen::Quaterniond q_i(qvec_i(0), qvec_i(1), qvec_i(2), qvec_i(3));
        if (q_i.norm() > 1e-12) {
          q_i.normalize();
        } else {
          q_i = Eigen::Quaterniond::Identity();
        }

        for (size_t j = i + 1; j < cameras_present.size(); ++j) {
          const camera_t cam_j_id = cameras_present[j];
          const image_t image_j_id = images_by_camera.at(cam_j_id);

          // 检查该图像对是否已被占用
          const image_pair_t pair_id =
              Database::ImagePairToPairId(image_i_id, image_j_id);
          if (occupied_pairs != nullptr &&
              occupied_pairs->count(pair_id) > 0) {
            continue;
          }

          // 获取相机j相对于rig参考坐标系的位姿
          const Eigen::Vector4d& qvec_j = camera_rig.RelativeQvec(cam_j_id);
          const Eigen::Vector3d& tvec_j = camera_rig.RelativeTvec(cam_j_id);
          Eigen::Quaterniond q_j(qvec_j(0), qvec_j(1), qvec_j(2), qvec_j(3));
          if (q_j.norm() > 1e-12) {
            q_j.normalize();
          } else {
            q_j = Eigen::Quaterniond::Identity();
          }

          // 计算相机i到相机j的相对位姿变换
          // R_ij = R_j * R_i^T, t_ij = t_j - R_ij * t_i
          const Eigen::Matrix3d R_i = q_i.toRotationMatrix();
          const Eigen::Matrix3d R_j = q_j.toRotationMatrix();
          const Eigen::Matrix3d R_ij = R_j * R_i.transpose();
          const Eigen::Vector3d t_ij = tvec_j - R_ij * tvec_i;

          // 构建rig约束边
          PoseGraphEdge edge;
          edge.type = EdgeType::kRig;
          edge.image_id1 = image_i_id;
          edge.image_id2 = image_j_id;
          edge.q_ij = Eigen::Quaterniond(R_ij);
          edge.translation_is_unit = defaults.rig_translation_is_unit;
          edge.t_ij = t_ij;
          // 如果需要单位化平移向量
          if (edge.translation_is_unit) {
            const double t_norm = edge.t_ij.norm();
            if (t_norm > 1e-12) {
              edge.t_ij /= t_norm;
            }
          }
          // 设置旋转和平移的权重
          edge.rot_weight = defaults.rig_rot_weight;
          edge.trans_weight = defaults.rig_trans_weight;

          edges.push_back(edge);
          // 将该图像对标记为已占用
          if (occupied_pairs != nullptr) {
            occupied_pairs->insert(pair_id);
          }
        }
      }
    }
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

  std::unordered_map<image_t, double> image_timestamps;
  image_timestamps.reserve(reconstruction.RegImageIds().size());
  // 获取图像时间戳
  auto GetTimestamp = [&](const image_t image_id, double* timestamp) {
    if (timestamp == nullptr) {
      return false;
    }
    auto it = image_timestamps.find(image_id);
    if (it != image_timestamps.end()) {
      if (std::isfinite(it->second)) {
        *timestamp = it->second;
        return true;
      }
      return false;
    }
    const Image& image = reconstruction.Image(image_id);
    double value = 0.0;
    if (TryParseImageTimestamp(image.Name(), &value)) {
      image_timestamps.emplace(image_id, value);
      *timestamp = value;
      return true;
    }
    image_timestamps.emplace(image_id, std::numeric_limits<double>::quiet_NaN());
    return false;
  };

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

    // 过滤3：图像时间间隔足够大（避免相邻帧）
    double time_i = 0.0;
    double time_j = 0.0;
    if (!GetTimestamp(i, &time_i) || !GetTimestamp(j, &time_j)) {
      continue;
    }
    if (std::abs(time_i - time_j) < options.min_loop_time_gap) {
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
    edge.t_ij = t_ij_dir;
    edge.translation_is_unit = true;
    continue;
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
 * [功能描述]：使用Sim3变换构建回环边
 * @param reconstruction：当前3D重建结果
 * @param database：特征数据库
 * @param options：管线选项
 * @param occupied_pairs：已占用的图像对（输入输出）
 * @param gt_reconstruction：真值重建（可选，用于误差评估）
 * @param loop_errors：回环误差列表（可选，用于存储误差）
 * @return 回环边列表
 */
std::vector<PoseGraphEdge> BuildLoopEdgesSim3(
    const Reconstruction& reconstruction, Database* database,
    const PipelineOptions& options,
    std::unordered_set<image_pair_t>* occupied_pairs,
    const Reconstruction* gt_reconstruction = nullptr,
    std::vector<LoopPoseError>* loop_errors = nullptr) {
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

  std::unordered_map<image_t, double> image_timestamps;
  image_timestamps.reserve(reconstruction.RegImageIds().size());
  // 获取图像时间戳
  auto GetTimestamp = [&](const image_t image_id, double* timestamp) {
    if (timestamp == nullptr) {
      return false;
    }
    auto it = image_timestamps.find(image_id);
    if (it != image_timestamps.end()) {
      if (std::isfinite(it->second)) {
        *timestamp = it->second;
        return true;
      }
      return false;
    }
    const Image& image = reconstruction.Image(image_id);
    double value = 0.0;
    if (TryParseImageTimestamp(image.Name(), &value)) {
      image_timestamps.emplace(image_id, value);
      *timestamp = value;
      return true;
    }
    image_timestamps.emplace(image_id, std::numeric_limits<double>::quiet_NaN());
    return false;
  };

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

    // 过滤3：图像时间间隔足够大（避免相邻帧）
    double time_i = 0.0;
    double time_j = 0.0;
    if (!GetTimestamp(i, &time_i) || !GetTimestamp(j, &time_j)) {
      continue;
    }
    if (std::abs(time_i - time_j) < options.min_loop_time_gap) {
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

    // 过滤9：每个图像的回环边数量不超过上限
    if (loop_degree[i] >= options.max_loop_edges_per_image ||
        loop_degree[j] >= options.max_loop_edges_per_image) {
      continue;
    }

    // 如果图像对在初始的重建中距离大于10m，则跳过
    const double dist = (image_i.ProjectionCenter() - image_j.ProjectionCenter()).norm();
    if (dist > 10.0) {
      continue;
    }

    bool added_any = false;

    // 先添加平移方向约束（不包含旋转）
    Eigen::Quaterniond q_ij_dir = Eigen::Quaterniond::Identity();
    Eigen::Vector3d t_ij_dir = Eigen::Vector3d::Zero();
    bool has_dir = false;
    // 判断数据库中是否已有相对位姿，如果有且有效则直接使用，否则重新估计
    if (geom.qvec.squaredNorm() > 1e-12 && geom.tvec.squaredNorm() > 1e-12) {
      const Eigen::Vector4d normalized_qvec = NormalizeQuaternion(geom.qvec);
      q_ij_dir = Eigen::Quaterniond(normalized_qvec(0), normalized_qvec(1),
                                    normalized_qvec(2), normalized_qvec(3));
      t_ij_dir = geom.tvec.normalized();
      has_dir = true;
    } else {
      if (EstimateRelativePoseFromInliers(geom, image_i, image_j, camera_i,
                                          camera_j, database, &points_cache,
                                          &q_ij_dir, &t_ij_dir)) {
        has_dir = true;
      }
    }

    // 如果有有效的平移方向约束，则添加一个只包含平移方向的边（旋转权重为0），以提供额外的约束信息
    if (has_dir && t_ij_dir.squaredNorm() > 1e-12) {
      // TODO: 是否需要将平移方向反向，使其与重建中相对位姿的平移方向一致???
      // Eigen::Vector4d qvec_pred;
      // Eigen::Vector3d tvec_pred;
      // ComputeRelativePose(image_i.Qvec(), image_i.Tvec(), image_j.Qvec(),
      //                     image_j.Tvec(), &qvec_pred, &tvec_pred);
      // if (tvec_pred.squaredNorm() > 1e-12 &&
      //     t_ij_dir.dot(tvec_pred) < 0) {
      //   t_ij_dir = -t_ij_dir;
      // }

      // 平移方向边只包含单位化的平移约束，旋转约束权重为0
      PoseGraphEdge dir_edge;
      dir_edge.type = EdgeType::kLoop;
      dir_edge.image_id1 = i;
      dir_edge.image_id2 = j;
      dir_edge.q_ij = Eigen::Quaterniond::Identity();
      dir_edge.t_ij = t_ij_dir.normalized();
      dir_edge.translation_is_unit = true;
      dir_edge.rot_weight = 0.0;
      dir_edge.trans_weight = options.defaults.loop_trans_weight;
      edges.push_back(dir_edge);
      added_any = true;
    }

    // 构建回环Sim3边（旋转+平移）
    double loop_scale = 0.0;
    Eigen::Quaterniond q_ij_sim3;
    Eigen::Vector3d t_ij_sim3;
    if (EstimateLoopScaleFromPointPairsSim3(
            reconstruction, geom, image_i, image_j, options.sim3_options,
            &loop_scale, q_ij_sim3, t_ij_sim3)) {
      PoseGraphEdge edge;
      edge.type = EdgeType::kLoop;
      edge.image_id1 = i;
      edge.image_id2 = j;
      edge.q_ij = q_ij_sim3;
      edge.t_ij = t_ij_sim3;
      edge.translation_is_unit = false;
      edge.rot_weight = options.defaults.loop_rot_weight;
      edge.trans_weight = options.defaults.loop_trans_weight;
      edges.push_back(edge);
      added_any = true;

      // 如果提供了真值重建，计算并记录回环相对位姿误差
      if (gt_reconstruction != nullptr && loop_errors != nullptr) {
        LoopPoseError error;
        if (ComputeLoopPoseError(*gt_reconstruction, image_i, image_j,
                                 q_ij_sim3, t_ij_sim3, &error)) {
          loop_errors->push_back(error);
        }
      }
    }

    if (!added_any) {
      continue;
    }

    occupied_pairs->insert(pair_id);  // 标记该对已使用
    loop_degree[i] += 1;
    loop_degree[j] += 1;
  }

  return edges;
}

std::vector<PoseGraphEdge> BuildManualLoopEdgesWithConstraints(
    const Reconstruction& reconstruction, Database* database,
    const PipelineOptions& options, const std::string& manual_loop_path,
    std::unordered_set<image_pair_t>* occupied_pairs,
    const Reconstruction* gt_reconstruction = nullptr,
    std::vector<LoopPoseError>* loop_errors = nullptr) {
  std::vector<PoseGraphEdge> edges;
  if (manual_loop_path.empty()) {
    return edges;
  }

  // 读取手动指定的回环边
  std::vector<PoseGraphEdge> manual_edges;
  std::string error;
  if (!ReadPoseGraphEdgesFromFile(manual_loop_path, options.defaults,
                                  &manual_edges, &error)) {
    std::cout << "Failed to read manual loop edges: " << error << "\n";
    return edges;
  }

  // 点的缓存,避免重复读取
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>> points_cache;
  points_cache.reserve(reconstruction.RegImageIds().size());

  size_t added_pairs = 0;
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
    if (occupied_pairs != nullptr && occupied_pairs->count(pair_id) > 0) {
      continue;  // 跳过已存在的边
    }

    const Image& image_i = reconstruction.Image(edge.image_id1);
    const Image& image_j = reconstruction.Image(edge.image_id2);
    const Camera& camera_i = reconstruction.Camera(image_i.CameraId());
    const Camera& camera_j = reconstruction.Camera(image_j.CameraId());

    TwoViewGeometry geom =
        database->ReadTwoViewGeometry(edge.image_id1, edge.image_id2);

    bool added_any = false;

    // 平移方向约束（不包含旋转）
    Eigen::Vector3d t_ij_dir = edge.t_ij;
    bool has_dir = t_ij_dir.squaredNorm() > 1e-12;
    if (!has_dir) {
      if (geom.qvec.squaredNorm() > 1e-12 && geom.tvec.squaredNorm() > 1e-12) {
        t_ij_dir = geom.tvec.normalized();
        has_dir = true;
      } else {
        Eigen::Quaterniond q_ij_est;
        Eigen::Vector3d t_ij_dir_est;
        if (EstimateRelativePoseFromInliers(geom, image_i, image_j, camera_i,
                                            camera_j, database, &points_cache,
                                            &q_ij_est, &t_ij_dir_est)) {
          t_ij_dir = t_ij_dir_est;
          has_dir = true;
        }
      }
    }

    if (has_dir && t_ij_dir.squaredNorm() > 1e-12) {
      t_ij_dir.normalize();
      // TODO: 是否需要将平移方向反向，使其与重建中相对位姿的平移方向一致???
      // Eigen::Vector4d qvec_pred;
      // Eigen::Vector3d tvec_pred;
      // ComputeRelativePose(image_i.Qvec(), image_i.Tvec(), image_j.Qvec(),
      //                     image_j.Tvec(), &qvec_pred, &tvec_pred);
      // if (tvec_pred.squaredNorm() > 1e-12 &&
      //     t_ij_dir.dot(tvec_pred) < 0) {
      //   t_ij_dir = -t_ij_dir;
      // }

      PoseGraphEdge dir_edge;
      dir_edge.type = EdgeType::kLoop;
      dir_edge.image_id1 = edge.image_id1;
      dir_edge.image_id2 = edge.image_id2;
      dir_edge.q_ij = Eigen::Quaterniond::Identity();
      dir_edge.t_ij = t_ij_dir;
      dir_edge.translation_is_unit = true;
      dir_edge.rot_weight = 0.0;
      dir_edge.trans_weight = options.defaults.loop_trans_weight;
      edges.push_back(dir_edge);
      added_any = true;
    }

    // Sim3约束（旋转+平移）
    double loop_scale = 0.0;
    Eigen::Quaterniond q_ij_sim3;
    Eigen::Vector3d t_ij_sim3;
    if (EstimateLoopScaleFromPointPairsSim3(
            reconstruction, geom, image_i, image_j, options.sim3_options,
            &loop_scale, q_ij_sim3, t_ij_sim3)) {
      PoseGraphEdge sim3_edge;
      sim3_edge.type = EdgeType::kLoop;
      sim3_edge.image_id1 = edge.image_id1;
      sim3_edge.image_id2 = edge.image_id2;
      sim3_edge.q_ij = q_ij_sim3;
      sim3_edge.t_ij = t_ij_sim3;
      sim3_edge.translation_is_unit = false;
      sim3_edge.rot_weight = options.defaults.loop_rot_weight;
      sim3_edge.trans_weight = options.defaults.loop_trans_weight;
      edges.push_back(sim3_edge);
      added_any = true;

      // 如果有真值重建，计算与真值重建的相对位姿误差
      if (gt_reconstruction != nullptr && loop_errors != nullptr) {
        LoopPoseError error_out;
        if (ComputeLoopPoseError(*gt_reconstruction, image_i, image_j,
                                 q_ij_sim3, t_ij_sim3, &error_out)) {
          loop_errors->push_back(error_out);
        }
      }
    } else if (!edge.translation_is_unit &&
               edge.q_ij.squaredNorm() > 1e-12 &&
               edge.t_ij.squaredNorm() > 1e-12) {
      // 无法估计Sim3时，退化为使用手动提供的全量约束
      PoseGraphEdge full_edge = edge;
      full_edge.type = EdgeType::kLoop;
      full_edge.translation_is_unit = false;
      full_edge.rot_weight = options.defaults.loop_rot_weight;
      full_edge.trans_weight = options.defaults.loop_trans_weight;
      edges.push_back(full_edge);
      added_any = true;

      // 如果有真值重建，计算与真值重建的相对位姿误差
      if (gt_reconstruction != nullptr && loop_errors != nullptr) {
        LoopPoseError error_out;
        if (ComputeLoopPoseError(*gt_reconstruction, image_i, image_j,
                                 full_edge.q_ij, full_edge.t_ij,
                                 &error_out)) {
          loop_errors->push_back(error_out);
        }
      }
    }

    if (!added_any) {
      continue;
    }

    if (occupied_pairs != nullptr) {
      occupied_pairs->insert(pair_id);
    }
    added_pairs += 1;
  }

  if (!edges.empty()) {
    std::cout << "Added manual loop edges: " << edges.size()
              << " (pairs=" << added_pairs << ")\n";
  }

  return edges;
}

/**
 * [功能描述]：将位姿图边作为残差块添加到Ceres优化问题中
 * @param edges：位姿图边列表
 * @param image_params：图像ID到位姿参数指针的映射（base/rel）
 * @param problem：Ceres优化问题
 * @param num_loop_edges：输出参数，添加的回环边数量（可为nullptr）
 * @param num_odom_edges：输出参数，添加的里程计边数量（可为nullptr）
 * @param num_rig_edges：输出参数，添加的rig边数量（可为nullptr）
 */
void AddEdgesToProblem(const std::vector<PoseGraphEdge>& edges,
                       const std::unordered_map<image_t, ImagePoseParams>&
                           image_params,
                       ceres::Problem* problem,
                       size_t* num_loop_edges,
                       size_t* num_odom_edges,
                       size_t* num_rig_edges) {
  // 初始化计数器
  if (num_loop_edges) {
    *num_loop_edges = 0;
  }
  if (num_odom_edges) {
    *num_odom_edges = 0;
  }
  if (num_rig_edges) {
    *num_rig_edges = 0;
  }

  for (const auto& edge : edges) {
    // 查找两端图像的位姿参数
    auto it1 = image_params.find(edge.image_id1);
    auto it2 = image_params.find(edge.image_id2);
    if (it1 == image_params.end() || it2 == image_params.end()) {
      continue;  // 位姿参数不存在则跳过
    }

    const double rot_weight = edge.rot_weight;
    const double trans_weight = edge.trans_weight;

    // 根据平移类型选择代价函数
    ceres::CostFunction* cost = nullptr;
    const bool same_base = it1->second.base_pose == it2->second.base_pose;
    const bool same_rel = it1->second.rel_pose == it2->second.rel_pose;
    if (same_base && same_rel) {
      // 同一快照，同一相机，跳过该边
      continue;
    }
    if (edge.translation_is_unit) {
      if (same_base) {
        cost = SE3RelativePoseCostRigSharedBase::Create(
            edge.q_ij, edge.t_ij, rot_weight, trans_weight);
      } else if (same_rel) {
        cost = SE3RelativePoseCostRigSharedRel::Create(
            edge.q_ij, edge.t_ij, rot_weight, trans_weight);
      } else {
        cost = SE3RelativePoseCostRig::Create(edge.q_ij, edge.t_ij, rot_weight,
                                              trans_weight);
      }
    } else {
      if (same_base) {
        cost = SE3RelativePoseCostRigSharedBaseFull::Create(
            edge.q_ij, edge.t_ij, rot_weight, trans_weight);
      } else if (same_rel) {
        cost = SE3RelativePoseCostRigSharedRelFull::Create(
            edge.q_ij, edge.t_ij, rot_weight, trans_weight);
      } else {
        cost = SE3RelativePoseCostRigFull::Create(edge.q_ij, edge.t_ij,
                                                  rot_weight, trans_weight);
      }
    }

    // 回环边使用Huber鲁棒核函数，里程计/rig边不使用
    ceres::LossFunction* loss = nullptr;
    if (edge.type == EdgeType::kLoop) {
      loss = new ceres::HuberLoss(1.0);
    }

    // 添加残差块到优化问题
    if (same_base) {
      problem->AddResidualBlock(cost, loss, it1->second.base_pose,
                                it1->second.rel_pose, it2->second.rel_pose);
    } else if (same_rel) {
      problem->AddResidualBlock(cost, loss, it1->second.base_pose,
                                it2->second.base_pose, it1->second.rel_pose);
    } else {
      problem->AddResidualBlock(cost, loss, it1->second.base_pose,
                                it1->second.rel_pose, it2->second.base_pose,
                                it2->second.rel_pose);
    }

    // 统计边数量
    if (edge.type == EdgeType::kLoop) {
      if (num_loop_edges) {
        *num_loop_edges += 1;
      }
    } else if (edge.type == EdgeType::kRig) {
      if (num_rig_edges) {
        *num_rig_edges += 1;
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
         "  --image-path <path>  Extract colors from images (optional).\n"
         "  --edge-input <path>   Read pose-graph edges (manual recall).\n"
         "  --edge-output <path>  Write pose-graph edges.\n"
         "  --manual-loop <path>  Add manual loop edges file.\n"
         "  --rig-config <path>   Add fixed rig constraints.\n"
         "  --odom-window <n>     Connect each frame to next n frames.\n"
         "  --skip-odom           Disable odometry edges.\n"
         "  --skip-loop           Disable auto loop edges.\n"
         "  --sim3-min-point-pairs <n>        Min 3D-3D pairs for Sim3.\n"
         "  --sim3-max-reproj-error <px>      Max reproj error in pixels.\n"
         "  --sim3-min-track-length <n>       Min track length.\n"
         "  --sim3-max-depth-ratio <r>        Max depth ratio to median.\n"
         "  --sim3-ransac-max-error <v>       RANSAC max error (<=0 for auto).\n"
         "  --sim3-ransac-min-inlier-ratio <r> RANSAC min inlier ratio.\n"
         "  --stage1-max-iterations <n>       Warmup stage iterations.\n"
         "  --stage1-tri-max-project-error <px> Warmup triangulation max reproj error.\n"
         "  --stage1-filter-max-reproj-error <px> Warmup filter max reproj error.\n"
         "  --stage2-max-iterations <n>       Refine stage iterations.\n"
         "  --stage2-tri-max-project-error <px> Refine triangulation max reproj error.\n"
         "  --stage2-filter-max-reproj-error <px> Refine filter max reproj error.\n"
         "  --stage3-max-iterations <n>       Final stage iterations.\n"
         "  --stage3-tri-max-project-error <px> Final triangulation max reproj error.\n"
         "  --stage3-filter-max-reproj-error <px> Final filter max reproj error.\n"
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
    // 可选参数：相机rig配置文件
    if (arg == "--rig-config" && i + 1 < argc) {
      options->rig_config_path = argv[++i];
      continue;
    }
    // 可选参数：里程计窗口大小
    if (arg == "--odom-window" && i + 1 < argc) {
      options->odom_window = std::stoul(argv[++i]);
      continue;
    }
    if (arg == "--sim3-min-point-pairs" && i + 1 < argc) {
      options->sim3_options.min_point_pairs = std::stoul(argv[++i]);
      continue;
    }
    if (arg == "--sim3-max-reproj-error" && i + 1 < argc) {
      options->sim3_options.max_reproj_error_px = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--sim3-min-track-length" && i + 1 < argc) {
      options->sim3_options.min_track_length = std::stoul(argv[++i]);
      continue;
    }
    if (arg == "--sim3-max-depth-ratio" && i + 1 < argc) {
      options->sim3_options.max_depth_ratio = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--sim3-ransac-max-error" && i + 1 < argc) {
      options->sim3_options.ransac_max_error = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--sim3-ransac-min-inlier-ratio" && i + 1 < argc) {
      options->sim3_options.ransac_min_inlier_ratio = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--stage1-max-iterations" && i + 1 < argc) {
      options->stage1_max_iterations = std::stoi(argv[++i]);
      continue;
    }
    if (arg == "--stage1-tri-max-project-error" && i + 1 < argc) {
      options->stage1_tri_max_project_error = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--stage1-filter-max-reproj-error" && i + 1 < argc) {
      options->stage1_filter_max_reproj_error = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--stage2-max-iterations" && i + 1 < argc) {
      options->stage2_max_iterations = std::stoi(argv[++i]);
      continue;
    }
    if (arg == "--stage2-tri-max-project-error" && i + 1 < argc) {
      options->stage2_tri_max_project_error = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--stage2-filter-max-reproj-error" && i + 1 < argc) {
      options->stage2_filter_max_reproj_error = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--stage3-max-iterations" && i + 1 < argc) {
      options->stage3_max_iterations = std::stoi(argv[++i]);
      continue;
    }
    if (arg == "--stage3-tri-max-project-error" && i + 1 < argc) {
      options->stage3_tri_max_project_error = std::stod(argv[++i]);
      continue;
    }
    if (arg == "--stage3-filter-max-reproj-error" && i + 1 < argc) {
      options->stage3_filter_max_reproj_error = std::stod(argv[++i]);
      continue;
    }
    // 可选参数：评估阈值（像素）
    if (arg == "--eval-threshold" && i + 1 < argc) {
      options->eval_threshold_px = std::stod(argv[++i]);
      continue;
    }
    // 可选参数：图像路径（用于颜色提取）
    if (arg == "--image-path" && i + 1 < argc) {
      options->image_path = argv[++i];
      continue;
    }
    // 可选参数：真值重建路径（用于评估回环相对位姿误差）
    if (arg == "--gt-reconstruction" && i + 1 < argc) {
      options->gt_reconstruction_path = argv[++i];
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

struct ObservationKey {
  image_t image_id = kInvalidImageId;
  point2D_t point2D_idx = kInvalidPoint2DIdx;

  bool operator==(const ObservationKey& other) const {
    return image_id == other.image_id && point2D_idx == other.point2D_idx;
  }
};

struct ObservationKeyHash {
  size_t operator()(const ObservationKey& key) const {
    const size_t h1 = std::hash<image_t>()(key.image_id);
    const size_t h2 = std::hash<point2D_t>()(key.point2D_idx);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

struct TrackTriangulationOptions {
  EstimateTriangulationOptions triangulation;
  size_t min_inlier_track_length = 3;
};

struct TrackTriangulationReport {
  size_t num_triangulated_points = 0;
  size_t num_added_observations = 0;
};

struct IterativeStageOptions {
  std::string name;
  int max_iterations = 0;
  bool triangulate_tracks = true; // 是否在阶段开始三角化新的3d点
  bool retriangulate_pairs = true; // 是否在阶段开始执行重三角化
  double complete_max_reproj_error = -1.0; // 轨迹补全阈值
  size_t min_track_length = 3; // 每轮BA前删除短轨迹阈值
  double far_point_max_dist_ratio = 10.0; // 每轮BA前删除极远点阈值(相对中位数)
  int landmark_uniform_num = 128; // 每轮BA前均匀化保留数量
  bool refine_rig_relative_poses = true; // 是否优化rig相对位姿
  TrackTriangulationOptions triangulation_options;
  BundleAdjustmentOptions ba_options;
  IncrementalMapper::Options filter_options;
};

// 过滤短轨迹点
size_t RemoveShortTracks(Reconstruction* reconstruction,
                         const size_t min_track_length) {
  if (reconstruction == nullptr || min_track_length <= 1) {
    return 0;
  }

  const auto point3D_ids = reconstruction->Point3DIds();
  size_t removed = 0;
  for (const auto point3D_id : point3D_ids) {
    if (!reconstruction->ExistsPoint3D(point3D_id)) {
      continue;
    }
    if (reconstruction->Point3D(point3D_id).Track().Length() < min_track_length) {
      reconstruction->DeletePoint3D(point3D_id);
      removed += 1;
    }
  }
  return removed;
}

size_t RemoveFarPointsByCameraDistance(Reconstruction* reconstruction,
                                       const double max_dist_ratio) {
  if (reconstruction == nullptr || max_dist_ratio <= 0.0) {
    return 0;
  }

  // 按每张已注册图像做局部距离统计，并为可见点累计远点投票
  std::unordered_map<point3D_t, size_t> point_total_votes;
  std::unordered_map<point3D_t, size_t> point_far_votes;
  point_total_votes.reserve(reconstruction->NumPoints3D());
  point_far_votes.reserve(reconstruction->NumPoints3D());

  for (const auto image_id : reconstruction->RegImageIds()) {
    if (!reconstruction->ExistsImage(image_id)) {
      continue;
    }

    const auto& image = reconstruction->Image(image_id);
    if (!image.IsRegistered()) {
      continue;
    }

    std::vector<std::pair<point3D_t, double>> point_distances;
    point_distances.reserve(image.NumPoints2D());
    std::vector<double> dists;
    dists.reserve(image.NumPoints2D());

    // 遍历图像中所有有3D点的观测，计算3D点到相机中心的距离
    for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
         ++point2D_idx) {
      const auto& point2D = image.Point2D(point2D_idx);
      if (!point2D.HasPoint3D()) {
        continue;
      }
      const point3D_t point3D_id = point2D.Point3DId();
      if (!reconstruction->ExistsPoint3D(point3D_id)) {
        continue;
      }

      const auto& point3D = reconstruction->Point3D(point3D_id);
      const double dist = (point3D.XYZ() - image.ProjectionCenter()).norm();
      if (!std::isfinite(dist)) {
        continue;
      }

      point_distances.emplace_back(point3D_id, dist);
      dists.push_back(dist);
    }

    if (dists.empty()) {
      continue;
    }

    // 基于当前图像可见点的距离中位数，确定该图像下的远点阈值。
    const size_t image_median_idx = dists.size() / 2;
    std::nth_element(
        dists.begin(), dists.begin() + image_median_idx, dists.end());
    const double image_median_dist = dists[image_median_idx];
    if (!std::isfinite(image_median_dist) || image_median_dist <= 0.0) {
      continue;
    }
    const double max_allowed_dist = image_median_dist * max_dist_ratio;

    for (const auto& point_dist : point_distances) {
      const point3D_t point3D_id = point_dist.first;
      const double dist = point_dist.second;
      point_total_votes[point3D_id] += 1;
      if (dist > max_allowed_dist) {
        point_far_votes[point3D_id] += 1;
      }
    }
  }

  if (point_total_votes.empty()) {
    return 0;
  }

  // 若一个点在其被观测的图像中，至少一半视角都判为远点，则删除该点
  size_t removed = 0;
  for (const auto& kv : point_total_votes) {
    const point3D_t point3D_id = kv.first;
    const size_t total_votes = kv.second;
    if (total_votes == 0 || !reconstruction->ExistsPoint3D(point3D_id)) {
      continue;
    }

    const auto far_it = point_far_votes.find(point3D_id);
    const size_t far_votes =
        far_it == point_far_votes.end() ? 0 : far_it->second;
    if (far_votes > 0 && far_votes * 2 >= total_votes) {
      reconstruction->DeletePoint3D(point3D_id);
      removed += 1;
    }
  }
  return removed;
}

/**
 * [功能描述]：从内点匹配图构建多视图轨迹（track）。
 *   将所有图像对的内点匹配视为图的边，通过连通分量分析合并为多视图轨迹，
 *   每个轨迹中同一图像只保留连接度最高的观测点。
 * @param reconstruction：当前3D重建结果
 * @param database：特征数据库
 * @return 多视图轨迹列表
 */
std::vector<Track> BuildMultiViewTracksFromInlierGraph(
    const Reconstruction& reconstruction, Database* database) {
  std::vector<Track> tracks;
  if (database == nullptr || reconstruction.NumRegImages() < 2) {
    return tracks;
  }

  // 收集所有已注册的图像ID
  std::unordered_set<image_t> reg_image_ids;
  reg_image_ids.reserve(reconstruction.NumRegImages());
  for (const image_t image_id : reconstruction.RegImageIds()) {
    reg_image_ids.insert(image_id);
  }

  // 读取所有图像对的两视图几何
  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  database->ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries);

  // ---- 第一阶段：构建观测点的邻接图 ----
  // 每个节点是一个(image_id, point2D_idx)观测，边连接内点匹配的两个观测
  std::unordered_map<ObservationKey, size_t, ObservationKeyHash> node_index;
  node_index.reserve(image_pair_ids.size() * 8);
  std::vector<ObservationKey> nodes;
  nodes.reserve(image_pair_ids.size() * 8);
  std::vector<std::vector<size_t>> adjacency;  // 邻接表
  adjacency.reserve(image_pair_ids.size() * 8);

  // 查找或创建节点，返回节点索引
  auto GetOrCreateNode = [&](const ObservationKey& key) -> size_t {
    const auto it = node_index.find(key);
    if (it != node_index.end()) {
      return it->second;
    }
    const size_t idx = nodes.size();
    node_index.emplace(key, idx);
    nodes.push_back(key);
    adjacency.emplace_back();
    return idx;
  };

  // 遍历所有图像对，将内点匹配添加为图的边
  size_t num_edges = 0;
  for (size_t pair_idx = 0; pair_idx < image_pair_ids.size(); ++pair_idx) {
    image_t image_id1 = kInvalidImageId;
    image_t image_id2 = kInvalidImageId;
    Database::PairIdToImagePair(image_pair_ids[pair_idx], &image_id1, &image_id2);
    // 跳过未注册的图像
    if (reg_image_ids.count(image_id1) == 0 || reg_image_ids.count(image_id2) == 0) {
      continue;
    }

    const TwoViewGeometry& geom = two_view_geometries[pair_idx];
    if (geom.inlier_matches.empty()) {
      continue;
    }

    const Image& image1 = reconstruction.Image(image_id1);
    const Image& image2 = reconstruction.Image(image_id2);
    for (const auto& match : geom.inlier_matches) {
      if (match.point2D_idx1 >= image1.NumPoints2D() ||
          match.point2D_idx2 >= image2.NumPoints2D()) {
        continue;
      }

      const ObservationKey key1{image_id1, match.point2D_idx1};
      const ObservationKey key2{image_id2, match.point2D_idx2};
      const size_t idx1 = GetOrCreateNode(key1);
      const size_t idx2 = GetOrCreateNode(key2);
      if (idx1 == idx2) {
        continue;
      }

      // 双向添加边
      adjacency[idx1].push_back(idx2);
      adjacency[idx2].push_back(idx1);
      num_edges += 1;
    }
  }

  // ---- 第二阶段：DFS查找连通分量，每个连通分量对应一条轨迹 ----
  std::vector<char> visited(nodes.size(), 0);
  std::vector<size_t> stack;
  stack.reserve(256);
  tracks.reserve(nodes.size() / 3 + 1);

  for (size_t node_idx = 0; node_idx < nodes.size(); ++node_idx) {
    if (visited[node_idx]) {
      continue;
    }

    // DFS遍历收集当前连通分量的所有节点
    stack.clear();
    stack.push_back(node_idx);
    visited[node_idx] = 1;

    std::vector<size_t> component_nodes;
    component_nodes.reserve(16);

    while (!stack.empty()) {
      const size_t cur = stack.back();
      stack.pop_back();
      component_nodes.push_back(cur);

      for (const size_t nbr : adjacency[cur]) {
        if (visited[nbr]) {
          continue;
        }
        visited[nbr] = 1;
        stack.push_back(nbr);
      }
    }

    // 同一图像中只保留连接度最高的观测点（去重）
    std::unordered_map<image_t, size_t> best_node_by_image;
    best_node_by_image.reserve(component_nodes.size());
    for (const size_t idx : component_nodes) {
      const ObservationKey& obs = nodes[idx];
      auto it = best_node_by_image.find(obs.image_id);
      if (it == best_node_by_image.end() ||
          adjacency[idx].size() > adjacency[it->second].size()) {
        best_node_by_image[obs.image_id] = idx;
      }
    }

    // 轨迹至少需要2个不同图像的观测
    if (best_node_by_image.size() < 2) {
      continue;
    }

    // 按(image_id, point2D_idx)排序，保证轨迹元素顺序一致
    std::vector<size_t> selected_nodes;
    selected_nodes.reserve(best_node_by_image.size());
    for (const auto& kv : best_node_by_image) {
      selected_nodes.push_back(kv.second);
    }
    std::sort(selected_nodes.begin(), selected_nodes.end(),
              [&](const size_t lhs, const size_t rhs) {
                const ObservationKey& a = nodes[lhs];
                const ObservationKey& b = nodes[rhs];
                if (a.image_id != b.image_id) {
                  return a.image_id < b.image_id;
                }
                return a.point2D_idx < b.point2D_idx;
              });

    // 构建轨迹
    Track track;
    track.Reserve(selected_nodes.size());
    for (const size_t idx : selected_nodes) {
      const ObservationKey& obs = nodes[idx];
      track.AddElement(obs.image_id, obs.point2D_idx);
    }

    if (track.Length() >= 2) {
      tracks.push_back(std::move(track));
    }
  }

  std::cout << "Built multi-view tracks from inlier graph: nodes=" << nodes.size()
            << " edges=" << num_edges << " tracks=" << tracks.size() << "\n";
  return tracks;
}

struct TrackTriangulationCandidate {
  bool success = false;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  std::vector<TrackElement> inlier_elements;
};

TrackTriangulationCandidate EstimateTrackTriangulationCandidate(
    const Track& input_track, const TrackTriangulationOptions& options,
    const Reconstruction& reconstruction) {
  TrackTriangulationCandidate candidate;
  if (input_track.Length() < 2) {
    return candidate;
  }

  // ---- 第一阶段：收集有效观测的2D/3D数据（只读）----
  std::vector<TriangulationEstimator::PointData> point_data;
  std::vector<TriangulationEstimator::PoseData> pose_data;
  std::vector<TrackElement> candidate_elements;
  point_data.reserve(input_track.Length());
  pose_data.reserve(input_track.Length());
  candidate_elements.reserve(input_track.Length());

  std::unordered_set<image_t> used_images;
  used_images.reserve(input_track.Length());

  for (const auto& track_el : input_track.Elements()) {
    if (!reconstruction.ExistsImage(track_el.image_id)) {
      continue;
    }
    const Image& image = reconstruction.Image(track_el.image_id);
    if (!image.IsRegistered()) {
      continue;
    }
    if (track_el.point2D_idx >= image.NumPoints2D()) {
      continue;
    }
    if (image.Point2D(track_el.point2D_idx).HasPoint3D()) {
      continue;
    }
    if (!used_images.insert(track_el.image_id).second) {
      continue;
    }

    const Camera& camera = reconstruction.Camera(image.CameraId());
    const Eigen::Vector2d point = image.Point2D(track_el.point2D_idx).XY();
    point_data.emplace_back(point, camera.ImageToWorld(point));
    pose_data.emplace_back(image.ProjectionMatrix(), image.ProjectionCenter(),
                           &camera);
    candidate_elements.push_back(track_el);
  }

  if (point_data.size() < 2) {
    return candidate;
  }

  // 对短轨迹使用穷举采样，和增量三角化逻辑对齐。
  EstimateTriangulationOptions tri_options = options.triangulation;
  const size_t kExhaustiveSamplingThreshold = 15;
  if (point_data.size() <= kExhaustiveSamplingThreshold) {
    tri_options.ransac_options.min_num_trials = NChooseK(point_data.size(), 2);
  }

  // ---- 第二阶段：RANSAC鲁棒三角化 ----
  std::vector<char> inlier_mask;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  if (!EstimateTriangulation(tri_options, point_data, pose_data, &inlier_mask,
                             &xyz)) {
    return candidate;
  }

  candidate.inlier_elements.reserve(inlier_mask.size());
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      candidate.inlier_elements.push_back(candidate_elements[i]);
    }
  }
  if (candidate.inlier_elements.size() <
      std::max<size_t>(2, options.min_inlier_track_length)) {
    candidate.inlier_elements.clear();
    return candidate;
  }

  candidate.success = true;
  candidate.xyz = xyz;
  return candidate;
}

bool CommitTrackTriangulationCandidate(
    const TrackTriangulationCandidate& candidate,
    const TrackTriangulationOptions& options, Reconstruction* reconstruction,
    size_t* num_added_observations) {
  if (reconstruction == nullptr || !candidate.success) {
    return false;
  }
  if (num_added_observations != nullptr) {
    *num_added_observations = 0;
  }

  // ---- 第三阶段：提交写回（串行）----
  Track inlier_track;
  inlier_track.Reserve(candidate.inlier_elements.size());
  std::unordered_set<image_t> used_images;
  used_images.reserve(candidate.inlier_elements.size());
  for (const auto& track_el : candidate.inlier_elements) {
    if (!reconstruction->ExistsImage(track_el.image_id)) {
      continue;
    }
    const Image& image = reconstruction->Image(track_el.image_id);
    if (track_el.point2D_idx >= image.NumPoints2D()) {
      continue;
    }
    if (image.Point2D(track_el.point2D_idx).HasPoint3D()) {
      continue;
    }
    if (!used_images.insert(track_el.image_id).second) {
      continue;
    }
    inlier_track.AddElement(track_el);
  }

  const size_t min_track_length = std::max<size_t>(2, options.min_inlier_track_length);
  if (inlier_track.Length() < min_track_length) {
    return false;
  }

  const size_t added = inlier_track.Length();
  reconstruction->AddPoint3D(candidate.xyz, std::move(inlier_track));
  if (num_added_observations != nullptr) {
    *num_added_observations = added;
  }
  return true;
}

/**
 * [功能描述]：使用RANSAC对单条轨迹进行三角化，将估计的3D点添加到重建中。
 *   该函数内部使用“估计候选(只读)+提交写回(串行)”模式，便于与并行批处理复用。
 */
bool TriangulateSingleTrackRANSAC(const Track& input_track,
                                  const TrackTriangulationOptions& options,
                                  Reconstruction* reconstruction,
                                  size_t* num_added_observations) {
  if (reconstruction == nullptr) {
    return false;
  }
  const auto candidate = EstimateTrackTriangulationCandidate(input_track, options, *reconstruction);
  return CommitTrackTriangulationCandidate(candidate, options, reconstruction,
                                           num_added_observations);
}

/**
 * [功能描述]：批量对多条轨迹执行RANSAC三角化，统计成功三角化的点数和观测数。
 * @param tracks：多视图轨迹列表
 * @param options：三角化选项
 * @param reconstruction：3D重建（输入输出）
 * @return 三角化报告（成功点数、添加的观测数）
 */
TrackTriangulationReport TriangulateTracksRANSAC(
    const std::vector<Track>& tracks, const TrackTriangulationOptions& options,
    Reconstruction* reconstruction) {
  TrackTriangulationReport report;
  if (reconstruction == nullptr) {
    return report;
  }
  if (tracks.empty()) {
    return report;
  }

  const int num_eff_threads =
      GetEffectiveNumThreads(ThreadPool::kMaxNumThreads);
  const size_t num_workers =
      std::max<size_t>(1, std::min<size_t>(tracks.size(), num_eff_threads));

  // 阶段1：并行估计每条轨迹的三角化候选（只读重建，线程安全）
  std::vector<TrackTriangulationCandidate> candidates(tracks.size());
  std::atomic<size_t> next_idx(0);
  ThreadPool thread_pool(static_cast<int>(num_workers));
  for (size_t worker = 0; worker < num_workers; ++worker) {
    thread_pool.AddTask([&]() {
      while (true) {
        const size_t idx = next_idx.fetch_add(1);
        if (idx >= tracks.size()) {
          break;
        }
        candidates[idx] = EstimateTrackTriangulationCandidate(
            tracks[idx], options, *reconstruction);
      }
    });
  }
  thread_pool.Wait();

  // 阶段2：串行提交，解决观测冲突并写回重建（AddPoint3D非线程安全）
  size_t num_estimated_candidates = 0;
  for (const auto& candidate : candidates) {
    if (!candidate.success) {
      continue;
    }
    num_estimated_candidates += 1;
    size_t num_added_observations = 0;
    if (!CommitTrackTriangulationCandidate(
            candidate, options, reconstruction, &num_added_observations)) {
      continue;
    }
    report.num_triangulated_points += 1;
    report.num_added_observations += num_added_observations;
  }

  return report;
}

/**
 * [功能描述]：执行迭代式分阶段优化。
 *   每个阶段仅在开始时做一次重三角化。
 *   每轮迭代：负深度过滤 -> 短轨迹过滤 -> 远点过滤 -> 均匀化 -> RigBA
 *            -> 补全/合并轨迹 -> 最终过滤，
 *   当观测变化率低于阈值时提前终止。
 * @param stage_options：阶段配置（名称、最大迭代次数、三角化/BA/过滤参数）
 * @param tracks：多视图轨迹列表
 * @param mapper_options：增量建图选项
 * @param mapper：增量建图器
 * @param reconstruction：3D重建（输入输出）
 * @param camera_rigs：相机rig列表
 * @param rig_ba_options：rig BA选项
 * @param ba_config：BA配置（包含参与优化的图像列表）
 * @return 成功返回true，BA失败返回false
 */
bool RunIterativeStage(const IterativeStageOptions& stage_options,
                       const std::vector<Track>& tracks,
                       const IncrementalMapperOptions& mapper_options,
                       IncrementalMapper* mapper, Reconstruction* reconstruction,
                       std::vector<CameraRig>* camera_rigs,
                       const RigBundleAdjuster::Options& rig_ba_options,
                       const BundleAdjustmentConfig& ba_config) {
  if (mapper == nullptr || reconstruction == nullptr || camera_rigs == nullptr) {
    return false;
  }
  if (stage_options.max_iterations <= 0) {
    return true;
  }

  PrintHeading1(stage_options.name);
  auto tri_options = mapper_options.Triangulation();
  if (stage_options.complete_max_reproj_error > 0.0) {
    tri_options.complete_max_reproj_error = stage_options.complete_max_reproj_error;
    tri_options.merge_max_reproj_error = stage_options.complete_max_reproj_error;
  }

  // 每个阶段只在开始时重三角化一次
  if (stage_options.triangulate_tracks) {
    const auto tri_report = TriangulateTracksRANSAC(
        tracks, stage_options.triangulation_options, reconstruction);
    std::cout << "  => Stage-start triangulated points: "
              << tri_report.num_triangulated_points
              << ", added observations: " << tri_report.num_added_observations
              << "\n";
  }
  if (stage_options.retriangulate_pairs) {
    const size_t num_retri = mapper->Retriangulate(tri_options);
    std::cout << "  => Stage-start retriangulated observations: " << num_retri
              << "\n";
  }

  // 迭代ba优化
  for (int iter = 0; iter < stage_options.max_iterations; ++iter) {
    std::cout << "[" << stage_options.name << "] Iteration " << (iter + 1)
              << "/" << stage_options.max_iterations << "\n";

    // 过滤负深度观测
    const size_t num_neg_depth = reconstruction->FilterObservationsWithNegativeDepth();
    std::cout << "  => Filtered negative-depth observations: " << num_neg_depth
              << "\n";
    const size_t num_observations_before =
        reconstruction->ComputeNumObservations();

    // 过滤短轨迹
    const size_t num_short_tracks_removed =
        RemoveShortTracks(reconstruction, stage_options.min_track_length);
    std::cout << "  => Removed short tracks: " << num_short_tracks_removed
              << "\n";

    // 过滤距离相机过远的3D点
    const size_t num_far_points_removed =
        RemoveFarPointsByCameraDistance(reconstruction,
                                        stage_options.far_point_max_dist_ratio);
    std::cout << "  => Removed far points: " << num_far_points_removed << "\n";

    if (reconstruction->ComputeNumObservations() == 0) {
      std::cout << "  => No observations. Skip BA in this iteration.\n";
      continue;
    }

    // 点云均匀化
    if (stage_options.landmark_uniform_num > 0) {
      DistributeLandmarks(*reconstruction, stage_options.landmark_uniform_num);
    }

    // Rig BA
    auto iter_rig_ba_options = rig_ba_options;
    iter_rig_ba_options.refine_relative_poses =
        stage_options.refine_rig_relative_poses;
    RigBundleAdjuster bundle_adjuster(stage_options.ba_options, iter_rig_ba_options,
                                      ba_config);
    if (!bundle_adjuster.Solve(reconstruction, camera_rigs)) {
      std::cout << "ERROR: bundle adjustment failed in stage "
                << stage_options.name << ".\n";
      return false;
    }

    // complete and merge
    const size_t num_completed = mapper->CompleteTracks(tri_options);
    std::cout << "  => Completed observations: " << num_completed << std::endl;
    const size_t num_merged = mapper->MergeTracks(tri_options);
    std::cout << "  => Merged observations: " << num_merged << std::endl;
    const size_t num_completed_merged = num_completed + num_merged;

    // 最终过滤，根据重投影误差和角度过滤观测
    reconstruction->FilterObservationsWithNegativeDepth();
    const size_t num_filtered = mapper->FilterPoints(stage_options.filter_options);
    std::cout << "  => Filtered observations: " << num_filtered << "\n";

    // 计算观测变化率，低于阈值则提前终止
    const size_t num_changed =
        num_short_tracks_removed + num_far_points_removed + num_completed_merged +
        num_filtered;
    const double changed =
        num_observations_before == 0
            ? 0.0
            : static_cast<double>(num_changed) /
                  static_cast<double>(num_observations_before);
    std::cout << StringPrintf("  => Changed observations: %.6f", changed)
              << std::endl;
    if (changed < mapper_options.ba_global_max_refinement_change) {
      break;
    }
  }

  return true;
}

/**
 * [功能描述]：执行三阶段迭代三角化与BA优化。
 *   阶段1：固定内参 + 固定rig约束，只优化外参。
 *   阶段2：固定内参，优化rig约束和外参。
 *   阶段3：优化内参、rig约束和外参。
 *   最后进行严格过滤和短轨迹清理。
 * @param reconstruction：3D重建（输入输出）
 * @param database：特征数据库
 * @param rig_config_path：相机rig配置路径（可为空）
 * @return 成功返回true，失败返回false
 */
bool TriangulateAndOptimizePaperStyle(Reconstruction* reconstruction,
                                      Database* database,
                                      const std::string& rig_config_path,
                                      const PipelineOptions& options) {
  if (reconstruction == nullptr || database == nullptr) {
    return false;
  }

  // 清除已有的3D点，从头开始三角化
  if (reconstruction->NumPoints3D() > 0) {
    const auto point3D_ids = reconstruction->Point3DIds();
    for (const auto point3D_id : point3D_ids) {
      reconstruction->DeletePoint3D(point3D_id);
    }
    std::cout << "Cleared existing 3D points: " << point3D_ids.size() << "\n";
  }

  if (reconstruction->NumRegImages() < 2) {
    std::cout << "Need at least two registered images for triangulation.\n";
    return false;
  }

  // ---- 准备工作：构建多视图轨迹 ----
  const std::vector<Track> tracks = BuildMultiViewTracksFromInlierGraph(*reconstruction, database);
  if (tracks.empty()) {
    std::cout << "No valid multi-view tracks were built.\n";
    return false;
  }

  // 初始化建图器和BA配置
  IncrementalMapperOptions mapper_options;
  // mapper_options.ba_refine_focal_length = true;
  // mapper_options.ba_refine_principal_point = false;
  // mapper_options.ba_refine_extra_params = true;
  // mapper_options.ba_global_max_refinements = 10;
  // mapper_options.ba_global_max_num_iterations = 50;

  DatabaseCache database_cache;
  const size_t min_num_matches =
      static_cast<size_t>(mapper_options.min_num_matches);
  database_cache.Load(*database, min_num_matches,
                      mapper_options.ignore_watermarks,
                      mapper_options.image_names);

  IncrementalMapper mapper(&database_cache);
  mapper.BeginReconstruction(reconstruction);

  // 将所有已注册图像加入BA优化
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction->RegImageIds()) {
    ba_config.AddImage(image_id);
  }

  // 加载rig配置（可选）
  std::vector<CameraRig> camera_rigs;
  RigBundleAdjuster::Options rig_ba_options;
  if (!rig_config_path.empty()) {
    std::string error;
    if (!ReadCameraRigConfigFromFile(rig_config_path, *reconstruction,
                                     &camera_rigs, &error)) {
      std::cout << "Failed to read rig config: " << error << "\n";
      mapper.EndReconstruction(true);
      return false;
    }
  }

  // ---- Stage 1: 固定内参 + 固定rig约束，只优化外参 ----
  IterativeStageOptions stage1;
  mapper_options.ba_refine_focal_length = false;
  mapper_options.ba_refine_principal_point = false;
  mapper_options.ba_refine_extra_params = false;
  mapper_options.ba_global_max_refinements = 3;
  mapper_options.ba_global_max_num_iterations = 50;
  double tri_max_project_error = options.stage1_tri_max_project_error;
  double filter_max_reproj_error = options.stage1_filter_max_reproj_error;
  stage1.name = "Stage1: Fix intrinsics + fix rig, optimize extrinsics";
  stage1.max_iterations = std::max(0, options.stage1_max_iterations);
  stage1.triangulate_tracks = true;
  stage1.retriangulate_pairs = true;
  stage1.min_track_length = 3;
  stage1.far_point_max_dist_ratio = 12.0;
  stage1.landmark_uniform_num = 128;
  stage1.refine_rig_relative_poses = false;
  stage1.triangulation_options.min_inlier_track_length = 3;
  stage1.triangulation_options.triangulation.min_tri_angle = DegToRad(2.0);
  stage1.triangulation_options.triangulation.residual_type =
      TriangulationEstimator::ResidualType::REPROJECTION_ERROR;
  stage1.triangulation_options.triangulation.ransac_options.max_error = tri_max_project_error;   // 宽松重投影误差阈值
  stage1.triangulation_options.triangulation.ransac_options.confidence = 0.9999;
  stage1.triangulation_options.triangulation.ransac_options.min_inlier_ratio = 0.02;
  stage1.triangulation_options.triangulation.ransac_options.max_num_trials = 10000;
  stage1.filter_options = mapper_options.Mapper();
  stage1.filter_options.filter_max_reproj_error = filter_max_reproj_error;   // 宽松过滤阈值
  stage1.filter_options.filter_min_tri_angle = 2.0;
  stage1.complete_max_reproj_error = tri_max_project_error;                // 宽松补全阈值
  stage1.ba_options = mapper_options.GlobalBundleAdjustment();
  stage1.ba_options.refine_focal_length = false;
  stage1.ba_options.refine_principal_point = false;
  stage1.ba_options.refine_extra_params = false;
  stage1.ba_options.refine_extrinsics = true;
  stage1.ba_options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::HUBER;
  stage1.ba_options.loss_function_scale = filter_max_reproj_error;

  // ---- Stage 2: 固定内参，优化rig约束与外参 ----
  IterativeStageOptions stage2;
  mapper_options.ba_refine_focal_length = false;
  mapper_options.ba_refine_principal_point = false;
  mapper_options.ba_refine_extra_params = false;
  tri_max_project_error = options.stage2_tri_max_project_error;
  filter_max_reproj_error = options.stage2_filter_max_reproj_error;
  stage2.name = "Stage2: Fix intrinsics, optimize rig + extrinsics";
  stage2.max_iterations = std::max(0, options.stage2_max_iterations);
  stage2.triangulate_tracks = true;
  stage2.retriangulate_pairs = true;
  stage2.min_track_length = 3;
  stage2.far_point_max_dist_ratio = 10.0;
  stage2.landmark_uniform_num = 128;
  stage2.refine_rig_relative_poses = true;
  stage2.triangulation_options.min_inlier_track_length = 3;
  stage2.triangulation_options.triangulation.min_tri_angle = DegToRad(2.0);
  stage2.triangulation_options.triangulation.residual_type =
      TriangulationEstimator::ResidualType::REPROJECTION_ERROR;
  stage2.triangulation_options.triangulation.ransac_options.max_error = tri_max_project_error;   // 严格重投影误差阈值
  stage2.triangulation_options.triangulation.ransac_options.confidence = 0.9999;
  stage2.triangulation_options.triangulation.ransac_options.min_inlier_ratio =
      0.02;
  stage2.triangulation_options.triangulation.ransac_options.max_num_trials =
      10000;
  stage2.filter_options = mapper_options.Mapper();
  stage2.filter_options.filter_max_reproj_error = filter_max_reproj_error;   // 严格过滤阈值
  stage2.filter_options.filter_min_tri_angle = 2.0;
  stage2.complete_max_reproj_error = tri_max_project_error;                // 严格补全阈值
  stage2.ba_options = mapper_options.GlobalBundleAdjustment();
  stage2.ba_options.refine_focal_length = false;
  stage2.ba_options.refine_principal_point = false;
  stage2.ba_options.refine_extra_params = false;
  stage2.ba_options.refine_extrinsics = true;
  stage2.ba_options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::HUBER;
  stage2.ba_options.loss_function_scale = filter_max_reproj_error;

  // ---- Stage 3: 优化内参 + rig约束 + 外参 ----
  IterativeStageOptions stage3;
  tri_max_project_error = options.stage3_tri_max_project_error;
  filter_max_reproj_error = options.stage3_filter_max_reproj_error;
  stage3.name = "Stage3: Optimize intrinsics + rig + extrinsics";
  stage3.max_iterations = std::max(0, options.stage3_max_iterations);
  stage3.triangulate_tracks = true;
  stage3.retriangulate_pairs = true;
  stage3.min_track_length = 3;
  stage3.far_point_max_dist_ratio = 8.0;
  stage3.landmark_uniform_num = 128;
  stage3.refine_rig_relative_poses = true;
  stage3.triangulation_options.min_inlier_track_length = 3;
  stage3.triangulation_options.triangulation.min_tri_angle = DegToRad(2.0);
  stage3.triangulation_options.triangulation.residual_type =
      TriangulationEstimator::ResidualType::REPROJECTION_ERROR;
  stage3.triangulation_options.triangulation.ransac_options.max_error =
      tri_max_project_error;
  stage3.triangulation_options.triangulation.ransac_options.confidence = 0.9999;
  stage3.triangulation_options.triangulation.ransac_options.min_inlier_ratio =
      0.02;
  stage3.triangulation_options.triangulation.ransac_options.max_num_trials =
      10000;
  stage3.filter_options = mapper_options.Mapper();
  stage3.filter_options.filter_max_reproj_error = filter_max_reproj_error;
  stage3.filter_options.filter_min_tri_angle = 2.0;
  stage3.complete_max_reproj_error = tri_max_project_error;
  stage3.ba_options = mapper_options.GlobalBundleAdjustment();
  stage3.ba_options.refine_focal_length = true;
  stage3.ba_options.refine_principal_point = true;
  stage3.ba_options.refine_extra_params = true;
  stage3.ba_options.refine_extrinsics = true;
  stage3.ba_options.loss_function_type =
      BundleAdjustmentOptions::LossFunctionType::HUBER;
  stage3.ba_options.loss_function_scale = filter_max_reproj_error;

  // 依次执行三个阶段
  if (!RunIterativeStage(stage1, tracks, mapper_options, &mapper, reconstruction,
                         &camera_rigs, rig_ba_options, ba_config)) {
    mapper.EndReconstruction(true);
    return false;
  }

  // 保存中间结果，便于分析
  reconstruction->Write("/home/xgrids/文档/data_need_pgo/data_1/0/test_pgo/pgo_test_ba1");

  if (!RunIterativeStage(stage2, tracks, mapper_options, &mapper, reconstruction,
                         &camera_rigs, rig_ba_options, ba_config)) {
    mapper.EndReconstruction(true);
    return false;
  }

  // 保存中间结果，便于分析
  reconstruction->Write("/home/xgrids/文档/data_need_pgo/data_1/0/test_pgo/pgo_test_ba2");

  if (!RunIterativeStage(stage3, tracks, mapper_options, &mapper, reconstruction,
                         &camera_rigs, rig_ba_options, ba_config)) {
    mapper.EndReconstruction(true);
    return false;
  }

  // ---- 最终清理：严格过滤 + 删除短轨迹 ----
  const double kStrictMaxReprojError = 4.0;
  const double kStrictMinTriAngle = 2.;
  const size_t kMinTrackLength = 3;
  const size_t final_filtered =
      reconstruction->FilterAllPoints3D(kStrictMaxReprojError, kStrictMinTriAngle);
  if (final_filtered > 0) {
    std::cout << "  => Filtered observations (strict): " << final_filtered
              << "\n";
  }

  // 删除观测数不足的短轨迹
  const auto point3D_ids = reconstruction->Point3DIds();
  size_t removed_tracks = 0;
  for (const auto point3D_id : point3D_ids) {
    if (!reconstruction->ExistsPoint3D(point3D_id)) {
      continue;
    }
    if (reconstruction->Point3D(point3D_id).Track().Length() < kMinTrackLength) {
      reconstruction->DeletePoint3D(point3D_id);
      removed_tracks += 1;
    }
  }
  if (removed_tracks > 0) {
    std::cout << "  => Removed short tracks: " << removed_tracks << "\n";
  }

  // 过滤质量不佳的图像
  const size_t num_filtered_images = mapper.FilterImages(stage3.filter_options);
  std::cout << "  => Filtered images: " << num_filtered_images << "\n";

  mapper.EndReconstruction(false);
  return true;
}

bool TriangulateAndOptimize(Reconstruction* reconstruction, Database* database,
                            const std::string& rig_config_path) {
  if (reconstruction == nullptr || database == nullptr) {
    return false;
  }

  // 清除当前重建中的所有3D点及其观测关联，避免影响后续三角化
  if (reconstruction->NumPoints3D() > 0) {
    const auto point3D_ids = reconstruction->Point3DIds();
    for (const auto point3D_id : point3D_ids) {
      reconstruction->DeletePoint3D(point3D_id);
    }
    std::cout << "Cleared existing 3D points: " << point3D_ids.size() << "\n";
  }

  if (reconstruction->NumRegImages() < 2) {
    std::cout << "Need at least two registered images for triangulation.\n";
    return false;
  }

  // 使用增量建图的默认三角化/过滤参数
  IncrementalMapperOptions mapper_options;
  mapper_options.ba_refine_focal_length = true;
  mapper_options.ba_refine_principal_point = false;
  mapper_options.ba_refine_extra_params = true;
  mapper_options.ba_global_max_refinements = 10; // 调整全局优化迭代次数
  mapper_options.ba_global_max_num_iterations = 50;
  IncrementalMapper::Options mapper_filter_options =
      mapper_options.Mapper();
  mapper_filter_options.filter_max_reproj_error = 3.0;
  mapper_filter_options.filter_min_tri_angle = 2.5;
  const double kStrictMaxReprojError = 3.0; // 严格的重投影误差阈值
  const double kStrictMinTriAngle = 2.5;    // 严格的最小三角化角度阈值（度）
  const size_t kMinTrackLength = 3;         // 最短轨迹长度阈值

  DatabaseCache database_cache;
  const size_t min_num_matches =
      static_cast<size_t>(mapper_options.min_num_matches);
  database_cache.Load(*database, min_num_matches,
                      mapper_options.ignore_watermarks,
                      mapper_options.image_names);

  // 增量建图器
  IncrementalMapper mapper(&database_cache);
  mapper.BeginReconstruction(reconstruction);

  const auto tri_options = mapper_options.Triangulation();
  const auto& reg_image_ids = reconstruction->RegImageIds();

  // 遍历所有已注册的图像，进行三角化
  for (size_t i = 0; i < reg_image_ids.size(); ++i) {
    const image_t image_id = reg_image_ids[i];
    const auto& image = reconstruction->Image(image_id);

    // PrintHeading1(
    //     StringPrintf("Triangulating image #%d (%d)", image_id, i));

    // const size_t num_existing_points3D = image.NumPoints3D();
    // std::cout << "  => Image sees " << num_existing_points3D << " / "
    //           << image.NumObservations() << " points\n";

    mapper.TriangulateImage(tri_options, image_id);

    // std::cout << "  => Triangulated "
    //           << (image.NumPoints3D() - num_existing_points3D) << " points\n";
  }

  PrintHeading1("Retriangulation");
  // 补全和合并轨迹
  CompleteAndMergeTracks(mapper_options, &mapper);
  std::cout << "  => Retriangulated observations: "
            << mapper.Retriangulate(tri_options) << "\n";
  // 过滤掉深度为负的观测点
  reconstruction->FilterObservationsWithNegativeDepth();
  // // 过滤掉重投影误差过大的观测点
  // const size_t filtered_obs = reconstruction->FilterAllPoints3D(kStrictMaxReprojError, kStrictMinTriAngle);
  // if (filtered_obs > 0) {
  //   std::cout << "  => Filtered observations (strict): " << filtered_obs
  //             << "\n";
  // }

  auto ba_options = mapper_options.GlobalBundleAdjustment();
  ba_options.refine_focal_length = true;
  ba_options.refine_principal_point = false;
  ba_options.refine_extra_params = true;
  ba_options.refine_extrinsics = true;

  // 将所有已注册的图像添加到优化中
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reconstruction->RegImageIds()) {
    ba_config.AddImage(image_id);
  }

  // 读取相机rig配置
  std::vector<CameraRig> camera_rigs;
  RigBundleAdjuster::Options rig_ba_options;
  if (!rig_config_path.empty()) {
    std::string error;
    if (!ReadCameraRigConfigFromFile(rig_config_path, *reconstruction,
                                     &camera_rigs, &error)) {
      std::cout << "Failed to read rig config: " << error << "\n";
      mapper.EndReconstruction(true);
      return false;
    }
  }

  if (reconstruction->ComputeNumObservations() == 0) {
    std::cout << "No observations after triangulation. Skipping BA.\n";
  } else {
    // 迭代优化循环
    for (int i = 0; i < mapper_options.ba_global_max_refinements; ++i) {
      // 过滤掉深度为负的观测点，避免BA中的退化情况
      reconstruction->FilterObservationsWithNegativeDepth();

      // 计算当前观测数量
      const size_t num_observations = reconstruction->ComputeNumObservations();

      PrintHeading1("Bundle adjustment");
      // 执行全局BA优化（支持相机rig）
      RigBundleAdjuster bundle_adjuster(ba_options, rig_ba_options, ba_config);
      // 如果BA失败，则结束重建
      if (!bundle_adjuster.Solve(reconstruction, &camera_rigs)) {
        std::cout << "ERROR: bundle adjustment failed.\n";
        mapper.EndReconstruction(true);
        return false;
      }

      // 补全和合并轨迹，统计变化的观测数
      size_t num_changed_observations = 0;
      num_changed_observations +=
          CompleteAndMergeTracks(mapper_options, &mapper);
      // num_changed_observations += mapper.Retriangulate(tri_options);
      const size_t iter_filtered_observations =
          mapper.FilterPoints(mapper_filter_options);
      std::cout << "  => Filtered observations: "
                << iter_filtered_observations << std::endl;
      num_changed_observations += iter_filtered_observations;
      // 过滤掉重投影误差过大的观测点
      // const size_t iter_filtered = reconstruction->FilterAllPoints3D(kStrictMaxReprojError, kStrictMinTriAngle);
      // if (iter_filtered > 0) {
      //   std::cout << "  => Filtered observations (strict): " << iter_filtered
      //             << "\n";
      // }
      const double changed =
          num_observations == 0
              ? 0
              : static_cast<double>(num_changed_observations) /
                    static_cast<double>(num_observations);
      std::cout << StringPrintf("  => Changed observations: %.6f", changed)
                << std::endl;
      if (changed < mapper_options.ba_global_max_refinement_change) {
        break;
      }
    }
  }

  // 过滤图像
  const size_t num_filtered_images =
      mapper.FilterImages(mapper_filter_options);
  std::cout << "  => Filtered images: " << num_filtered_images << std::endl;

  const bool kDiscardReconstruction = false;
  mapper.EndReconstruction(kDiscardReconstruction);
  return true;
}

// 分块优化
bool TriangulateAndOptimizeHierarchical(
    Reconstruction* reconstruction, Database* database,
    const std::string& rig_config_path) {
  if (reconstruction == nullptr || database == nullptr) {
    return false;
  }

  if (reconstruction->NumRegImages() < 2) {
    std::cout << "Need at least two registered images for triangulation.\n";
    return false;
  }

  // 分块参数（可根据数据规模调整）
  const size_t kClusterSize = 200; // 子模型切分所需图像数量
  const size_t kClusterOverlap = 50; // 子模型之间的图像重叠数量
  const int kLocalMaxRefinements = 3; // 局部优化迭代次数

  // 采用增量建图的默认参数作为基础
  IncrementalMapperOptions mapper_options;
  mapper_options.ba_refine_focal_length = true;
  mapper_options.ba_refine_principal_point = false;
  mapper_options.ba_refine_extra_params = true;
  mapper_options.ba_global_max_refinements = 10;
  mapper_options.ba_global_max_num_iterations = 50;

  const double kLooseMaxReprojError = 4.0;
  const double kLooseMinTriAngle = 1.5;
  const double kStrictMaxReprojError = 3.0;
  const double kStrictMinTriAngle = 1.5;
  const size_t kMinTrackLength = 3;
  const double kMergeMaxReprojError = 8.0;

  // 宽松过滤参数
  IncrementalMapper::Options mapper_filter_loose = mapper_options.Mapper();
  mapper_filter_loose.filter_max_reproj_error = kLooseMaxReprojError;
  mapper_filter_loose.filter_min_tri_angle = kLooseMinTriAngle;
  // 严格过滤参数
  IncrementalMapper::Options mapper_filter_strict = mapper_options.Mapper();
  mapper_filter_strict.filter_max_reproj_error = kStrictMaxReprojError;
  mapper_filter_strict.filter_min_tri_angle = kStrictMinTriAngle;

  // 构建数据库缓存（所有子模型共享）
  DatabaseCache database_cache;
  const size_t min_num_matches = static_cast<size_t>(mapper_options.min_num_matches);
  database_cache.Load(*database, min_num_matches,
                      mapper_options.ignore_watermarks,
                      mapper_options.image_names);

  // 按 hierarchical 的 scene graph 分块
  SceneClustering::Options clustering_options;
  clustering_options.is_hierarchical = true;
  clustering_options.branching = 2;
  clustering_options.image_overlap = static_cast<int>(kClusterOverlap);
  clustering_options.leaf_max_num_images = static_cast<int>(kClusterSize);
  const auto clusters = BuildHierarchicalClusters(*reconstruction, *database,
                                                  clustering_options);
  if (clusters.empty()) {
    std::cout << "No valid clusters for hierarchical optimization.\n";
    return false;
  }

  // 并行构建子模型并做局部优化
  std::vector<size_t> cluster_order(clusters.size());
  std::iota(cluster_order.begin(), cluster_order.end(), 0);
  // 按图像数量排序
  // std::sort(cluster_order.begin(), cluster_order.end(),
  //           [&](size_t a, size_t b) {
  //             return clusters[a].size() > clusters[b].size();
  //           });

  // 获取有效线程，动态分配线程数
  const int num_eff_threads = GetEffectiveNumThreads(ThreadPool::kMaxNumThreads);
  const int kDefaultNumWorkers = 8;
  const int num_eff_workers = std::max(
      1, std::min(static_cast<int>(clusters.size()),
                  std::min(kDefaultNumWorkers, num_eff_threads)));
  const int num_threads_per_worker = std::max(1, num_eff_threads / num_eff_workers);

  std::vector<Reconstruction> sub_reconstructions(clusters.size());
  std::vector<bool> sub_success(clusters.size(), false);
  std::atomic<bool> any_failed(false);
  std::mutex error_mutex;
  std::string error_msg;

  auto ProcessCluster = [&](const size_t c) {
    // 如果有线程失败了，终止所有后续的处理
    if (any_failed.load()) {
      return;
    }

    std::cout << "Building sub-model " << (c + 1) << " / " << clusters.size()
              << " (images: " << clusters[c].size() << ")\n";

    IncrementalMapperOptions local_options = mapper_options;
    if (local_options.num_threads < 0) {
      local_options.num_threads = num_threads_per_worker;
    }
    // 局部优化使用宽松的过滤参数
    IncrementalMapper::Options mapper_filter_loose_local = local_options.Mapper();
    mapper_filter_loose_local.filter_max_reproj_error = kLooseMaxReprojError;
    mapper_filter_loose_local.filter_min_tri_angle = kLooseMinTriAngle;

    // 从空模型构建当前子模型，避免拷贝整份重建导致峰值内存过高。
    Reconstruction sub_rec;
    std::unordered_set<camera_t> added_camera_ids;
    added_camera_ids.reserve(clusters[c].size());
    for (const auto image_id : clusters[c]) {
      if (!reconstruction->ExistsImage(image_id)) {
        continue;
      }

      const auto& src_image = reconstruction->Image(image_id);
      const camera_t camera_id = src_image.CameraId();
      if (added_camera_ids.insert(camera_id).second) {
        if (!reconstruction->ExistsCamera(camera_id)) {
          std::lock_guard<std::mutex> lock(error_mutex);
          any_failed.store(true);
          error_msg = "ERROR: missing camera for clustered image.";
          return;
        }
        sub_rec.AddCamera(reconstruction->Camera(camera_id));
      }

      Image sub_image = src_image;
      sub_image.SetRegistered(false);
      for (point2D_t point2D_idx = 0; point2D_idx < sub_image.NumPoints2D();
           ++point2D_idx) {
        sub_image.ResetPoint3DForPoint2D(point2D_idx);
      }
      sub_rec.AddImage(std::move(sub_image));
      sub_rec.RegisterImage(image_id);
    }
    if (sub_rec.NumRegImages() < 2) {
      std::lock_guard<std::mutex> lock(error_mutex);
      any_failed.store(true);
      error_msg = "ERROR: cluster has fewer than two valid images.";
      return;
    }

    IncrementalMapper mapper(&database_cache);
    mapper.BeginReconstruction(&sub_rec);

    const auto tri_options = local_options.Triangulation();
    const auto& sub_reg_ids = sub_rec.RegImageIds();
    for (const image_t image_id : sub_reg_ids) {
      mapper.TriangulateImage(tri_options, image_id);
    }

    CompleteAndMergeTracks(local_options, &mapper);
    mapper.Retriangulate(tri_options);
    sub_rec.FilterObservationsWithNegativeDepth();

    // 调整局部ba参数
    BundleAdjustmentOptions ba_options_local = local_options.GlobalBundleAdjustment();
    ba_options_local.refine_focal_length = true;
    ba_options_local.refine_principal_point = false;
    ba_options_local.refine_extra_params = true;
    ba_options_local.refine_extrinsics = true;
    ba_options_local.loss_function_type =
        BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
    ba_options_local.loss_function_scale = 1.0;

    BundleAdjustmentConfig ba_config;
    for (const image_t image_id : sub_reg_ids) {
      ba_config.AddImage(image_id);
    }

    // 读取相机rig配置
    std::vector<CameraRig> camera_rigs;
    RigBundleAdjuster::Options rig_ba_options;
    if (!rig_config_path.empty()) {
      std::string error;
      if (!ReadCameraRigConfigFromFile(rig_config_path, sub_rec,
                                       &camera_rigs, &error)) {
        mapper.EndReconstruction(true);
        std::lock_guard<std::mutex> lock(error_mutex);
        any_failed.store(true);
        error_msg = "Failed to read rig config: " + error;
        return;
      }
    }

    if (sub_rec.ComputeNumObservations() > 0) {
      // 局部迭代优化
      for (int i = 0; i < kLocalMaxRefinements; ++i) {
        // 过滤负深度观测
        sub_rec.FilterObservationsWithNegativeDepth();
        const size_t num_observations = sub_rec.ComputeNumObservations();

        // ba求解
        RigBundleAdjuster bundle_adjuster(ba_options_local, rig_ba_options, ba_config);
        if (!bundle_adjuster.Solve(&sub_rec, &camera_rigs)) {
          mapper.EndReconstruction(true);
          std::lock_guard<std::mutex> lock(error_mutex);
          any_failed.store(true);
          error_msg = "ERROR: local bundle adjustment failed.";
          return;
        }

        // 观测变化
        size_t num_changed_observations = 0;
        num_changed_observations +=
            CompleteAndMergeTracks(local_options, &mapper);
        const size_t iter_filtered_observations =
            mapper.FilterPoints(mapper_filter_loose_local);
        std::cout << "  => Filtered observations: "
                  << iter_filtered_observations << std::endl;
        num_changed_observations += iter_filtered_observations;

        const double changed =
            num_observations == 0
                ? 0
                : static_cast<double>(num_changed_observations) /
                      static_cast<double>(num_observations);
        std::cout << StringPrintf("  => Changed observations: %.6f", changed)
                  << std::endl;
        if (changed < local_options.ba_global_max_refinement_change) {
          break;
        }
      }
    }

    mapper.EndReconstruction(false);
    sub_reconstructions[c] = std::move(sub_rec);
    sub_success[c] = true;
  };

  // 线程池并行处理子模型
  ThreadPool thread_pool(num_eff_workers);
  for (const auto c : cluster_order) {
    thread_pool.AddTask(ProcessCluster, c);
  }
  thread_pool.Wait();

  // 如果有任一线程失败，直接终止
  if (any_failed.load()) {
    std::cout << error_msg << "\n";
    return false;
  }

  // 合并子模型（参考 hierarchical 的贪心合并策略）
  std::vector<size_t> merge_indices;
  merge_indices.reserve(clusters.size());
  for (const auto c : cluster_order) {
    if (sub_success[c]) {
      merge_indices.push_back(c);
    }
  }
  if (merge_indices.empty()) {
    std::cout << "No valid sub-models were reconstructed.\n";
    return false;
  }

  // 尝试两两合并子模型，直至无法合并
  bool merge_success = true;
  while (merge_indices.size() > 1 && merge_success) {
    merge_success = false;
    for (size_t i = 0; i < merge_indices.size(); ++i) {
      for (size_t j = 0; j < i; ++j) {
        auto& rec_i = sub_reconstructions[merge_indices[i]];
        auto& rec_j = sub_reconstructions[merge_indices[j]];
        if (rec_i.Merge(rec_j, kMergeMaxReprojError)) {
          sub_reconstructions[merge_indices[j]] = Reconstruction();
          merge_indices.erase(merge_indices.begin() + j);
          merge_success = true;
          break;
        }
      }
      if (merge_success) {
        break;
      }
    }
  }

  // 如果还剩多个模型，选取最大的一个作为基准模型
  size_t base_idx = merge_indices.front();
  if (merge_indices.size() > 1) {
    size_t best_idx = merge_indices.front();
    size_t best_size = sub_reconstructions[best_idx].NumRegImages();
    for (const auto idx : merge_indices) {
      const size_t size = sub_reconstructions[idx].NumRegImages();
      if (size > best_size) {
        best_idx = idx;
        best_size = size;
      }
    }
    base_idx = best_idx;
    std::cout << "WARNING: " << merge_indices.size()
              << " sub-models remain unmerged; using largest as base.\n";
  }

  for (size_t idx = 0; idx < sub_reconstructions.size(); ++idx) {
    if (idx != base_idx) {
      sub_reconstructions[idx] = Reconstruction();
    }
  }

  Reconstruction merged = std::move(sub_reconstructions[base_idx]);

  // 全局优化
  IncrementalMapper mapper(&database_cache);
  mapper.BeginReconstruction(&merged);

  // 调整全局ba参数
  BundleAdjustmentOptions ba_options_global = mapper_options.GlobalBundleAdjustment();
  ba_options_global.refine_focal_length = true;
  ba_options_global.refine_principal_point = false;
  ba_options_global.refine_extra_params = true;
  ba_options_global.refine_extrinsics = true;
  BundleAdjustmentOptions ba_options_warmup = ba_options_global;
  ba_options_warmup.loss_function_type = BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  ba_options_warmup.loss_function_scale = 1.0;
  BundleAdjustmentOptions ba_options_refine = ba_options_global;
  ba_options_refine.loss_function_type = BundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  ba_options_refine.loss_function_scale = 0.5;

  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : merged.RegImageIds()) {
    ba_config.AddImage(image_id);
  }

  std::vector<CameraRig> camera_rigs;
  RigBundleAdjuster::Options rig_ba_options;
  if (!rig_config_path.empty()) {
    std::string error;
    if (!ReadCameraRigConfigFromFile(rig_config_path, merged, &camera_rigs,
                                     &error)) {
      std::cout << "Failed to read rig config: " << error << "\n";
      mapper.EndReconstruction(true);
      return false;
    }
  }

  if (merged.ComputeNumObservations() > 0) {
    const int kWarmupIters = std::min(5, mapper_options.ba_global_max_refinements);
    const int kMainIters = std::max(0, mapper_options.ba_global_max_refinements - kWarmupIters);

    for (int i = 0; i < kWarmupIters; ++i) {
      merged.FilterObservationsWithNegativeDepth();
      const size_t num_observations = merged.ComputeNumObservations();

      PrintHeading1("Bundle adjustment (warmup)");
      RigBundleAdjuster bundle_adjuster(ba_options_warmup, rig_ba_options,
                                        ba_config);
      if (!bundle_adjuster.Solve(&merged, &camera_rigs)) {
        std::cout << "ERROR: bundle adjustment failed.\n";
        mapper.EndReconstruction(true);
        return false;
      }

      size_t num_changed_observations = 0;
      num_changed_observations +=
          CompleteAndMergeTracks(mapper_options, &mapper);
      const size_t iter_filtered_observations =
          mapper.FilterPoints(mapper_filter_loose);
      std::cout << "  => Filtered observations: "
                << iter_filtered_observations << std::endl;
      num_changed_observations += iter_filtered_observations;

      const double changed =
          num_observations == 0
              ? 0
              : static_cast<double>(num_changed_observations) /
                    static_cast<double>(num_observations);
      std::cout << StringPrintf("  => Changed observations: %.6f", changed)
                << std::endl;
      if (changed < mapper_options.ba_global_max_refinement_change) {
        break;
      }
    }

    for (int i = 0; i < kMainIters; ++i) {
      merged.FilterObservationsWithNegativeDepth();
      const size_t num_observations = merged.ComputeNumObservations();

      PrintHeading1("Bundle adjustment (refine)");
      RigBundleAdjuster bundle_adjuster(ba_options_refine, rig_ba_options,
                                        ba_config);
      if (!bundle_adjuster.Solve(&merged, &camera_rigs)) {
        std::cout << "ERROR: bundle adjustment failed.\n";
        mapper.EndReconstruction(true);
        return false;
      }

      size_t num_changed_observations = 0;
      num_changed_observations +=
          CompleteAndMergeTracks(mapper_options, &mapper);
      const size_t iter_filtered_observations =
          mapper.FilterPoints(mapper_filter_strict);
      std::cout << "  => Filtered observations: "
                << iter_filtered_observations << std::endl;
      num_changed_observations += iter_filtered_observations;

      const double changed =
          num_observations == 0
              ? 0
              : static_cast<double>(num_changed_observations) /
                    static_cast<double>(num_observations);
      std::cout << StringPrintf("  => Changed observations: %.6f", changed)
                << std::endl;
      if (changed < mapper_options.ba_global_max_refinement_change) {
        break;
      }
    }
  }

  const size_t final_filtered =
      merged.FilterAllPoints3D(kStrictMaxReprojError, kStrictMinTriAngle);
  if (final_filtered > 0) {
    std::cout << "  => Filtered observations (strict): " << final_filtered
              << "\n";
  }
  const auto point3D_ids = merged.Point3DIds();
  size_t removed_tracks = 0;
  for (const auto point3D_id : point3D_ids) {
    if (!merged.ExistsPoint3D(point3D_id)) {
      continue;
    }
    if (merged.Point3D(point3D_id).Track().Length() < kMinTrackLength) {
      merged.DeletePoint3D(point3D_id);
      removed_tracks += 1;
    }
  }
  if (removed_tracks > 0) {
    std::cout << "  => Removed short tracks: " << removed_tracks << "\n";
  }

  const size_t num_filtered_images =
      mapper.FilterImages(mapper_filter_strict);
  std::cout << "  => Filtered images: " << num_filtered_images << std::endl;

  mapper.EndReconstruction(false);
  *reconstruction = std::move(merged);
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

  // ========== 2.1. 加载真值重建（可选，用于回环误差评估） ==========
  std::unique_ptr<Reconstruction> gt_reconstruction;
  const bool use_gt_reconstruction = !options.gt_reconstruction_path.empty();
  if (use_gt_reconstruction) {
    gt_reconstruction = std::make_unique<Reconstruction>();
    gt_reconstruction->Read(options.gt_reconstruction_path);
    if (gt_reconstruction->RegImageIds().empty()) {
      std::cout << "WARNING: GT reconstruction has no registered images.\n";
      gt_reconstruction.reset();
    } else {
      std::cout << "Loaded GT reconstruction with "
                << gt_reconstruction->NumRegImages() << " images.\n";
    }
  }

  // 用于存储回环相对位姿误差
  std::vector<LoopPoseError> loop_pose_errors;

  // ========== 3. 读取rig配置（可选） ==========
  std::vector<CameraRig> camera_rigs;
  const bool use_rig = !options.rig_config_path.empty();
  if (use_rig) {
    std::string error;
    if (!ReadCameraRigConfigFromFile(options.rig_config_path, reconstruction,
                                     &camera_rigs, &error)) {
      std::cout << "Failed to read rig config: " << error << "\n";
      return -1;
    }
  }

  // ========== 4. 初始化Ceres优化问题与参数块 ==========
  ceres::Problem problem;
  std::unordered_map<image_t, ImagePoseParams> image_params;
  image_params.reserve(reconstruction.RegImageIds().size());

  std::vector<std::unique_ptr<double[]>> owned_base_params;
  std::vector<std::unique_ptr<double[]>> owned_rel_params;
  std::unordered_set<double*> added_param_blocks;
  added_param_blocks.reserve(reconstruction.RegImageIds().size());

  auto AddParamBlock = [&](double* ptr, const bool constant) {
    if (added_param_blocks.insert(ptr).second) {
      problem.AddParameterBlock(ptr, 7, new SE3Manifold());
    }
    if (constant) {
      problem.SetParameterBlockConstant(ptr);
    }
  };

  // 全局identity相对位姿（用于非rig图像）
  auto identity_rel = std::make_unique<double[]>(7);
  identity_rel[0] = 1.0;
  identity_rel[1] = 0.0;
  identity_rel[2] = 0.0;
  identity_rel[3] = 0.0;
  identity_rel[4] = 0.0;
  identity_rel[5] = 0.0;
  identity_rel[6] = 0.0;
  double* identity_rel_ptr = identity_rel.get();
  owned_rel_params.push_back(std::move(identity_rel));
  AddParamBlock(identity_rel_ptr, true);

  // 如果启用rig参数化：为每个snapshot创建rig位姿，为每个相机创建相对位姿
  if (use_rig) {
    for (auto& camera_rig : camera_rigs) {
      // 为rig中的每个相机创建相对位姿参数块
      std::unordered_map<camera_t, double*> rel_pose_by_camera;
      rel_pose_by_camera.reserve(camera_rig.GetCameraIds().size());
      for (const auto camera_id : camera_rig.GetCameraIds()) {
        const Eigen::Vector4d& rel_qvec =
            camera_rig.RelativeQvec(camera_id);
        const Eigen::Vector3d& rel_tvec =
            camera_rig.RelativeTvec(camera_id);

        Eigen::Quaterniond q(rel_qvec(0), rel_qvec(1), rel_qvec(2),
                             rel_qvec(3));
        if (q.norm() > 1e-12) {
          q.normalize();
        } else {
          q = Eigen::Quaterniond::Identity();
        }

        auto rel_pose = std::make_unique<double[]>(7);
        rel_pose[0] = q.w();
        rel_pose[1] = q.x();
        rel_pose[2] = q.y();
        rel_pose[3] = q.z();
        rel_pose[4] = rel_tvec.x();
        rel_pose[5] = rel_tvec.y();
        rel_pose[6] = rel_tvec.z();

        double* rel_pose_ptr = rel_pose.get();
        owned_rel_params.push_back(std::move(rel_pose));
        rel_pose_by_camera[camera_id] = rel_pose_ptr;
        AddParamBlock(rel_pose_ptr, false);
      }

      // 固定参考相机的相对位姿，消除rig内部的自由度
      const camera_t ref_camera_id = camera_rig.RefCameraId();
      auto ref_it = rel_pose_by_camera.find(ref_camera_id);
      if (ref_it != rel_pose_by_camera.end()) {
        AddParamBlock(ref_it->second, true);
      }

      // 为每个snapshot创建rig绝对位姿，并绑定该snapshot内的图像
      for (size_t snapshot_idx = 0; snapshot_idx < camera_rig.NumSnapshots();
           ++snapshot_idx) {
        Eigen::Vector4d rig_qvec = ComposeIdentityQuaternion();
        Eigen::Vector3d rig_tvec = Eigen::Vector3d::Zero();
        camera_rig.ComputeAbsolutePose(snapshot_idx, reconstruction, &rig_qvec,
                                       &rig_tvec);

        Eigen::Quaterniond q(rig_qvec(0), rig_qvec(1), rig_qvec(2),
                             rig_qvec(3));
        if (q.norm() > 1e-12) {
          q.normalize();
        } else {
          q = Eigen::Quaterniond::Identity();
        }

        // rig的绝对位姿
        auto rig_pose = std::make_unique<double[]>(7);
        rig_pose[0] = q.w();
        rig_pose[1] = q.x();
        rig_pose[2] = q.y();
        rig_pose[3] = q.z();
        rig_pose[4] = rig_tvec.x();
        rig_pose[5] = rig_tvec.y();
        rig_pose[6] = rig_tvec.z();

        // 添加rig的绝对位姿参数块
        double* rig_pose_ptr = rig_pose.get();
        owned_base_params.push_back(std::move(rig_pose));
        AddParamBlock(rig_pose_ptr, false);

        // 为rig中的每个图像添加相对位姿参数块
        for (const auto image_id : camera_rig.Snapshots()[snapshot_idx]) {
          if (!reconstruction.ExistsImage(image_id)) {
            continue;
          }
          const Image& image = reconstruction.Image(image_id);
          const camera_t camera_id = image.CameraId();
          if (!camera_rig.HasCamera(camera_id)) {
            continue;
          }
          auto rel_it = rel_pose_by_camera.find(camera_id);
          if (rel_it == rel_pose_by_camera.end()) {
            continue;
          }
          image_params[image_id] = ImagePoseParams{rig_pose_ptr,
                                                   rel_it->second};
        }
      }
    }
  }

  // 为非rig图像创建独立位姿参数块
  for (const auto image_id : reconstruction.RegImageIds()) {
    if (image_params.find(image_id) != image_params.end()) {
      continue;
    }
    const Image& image = reconstruction.Image(image_id);
    auto pose = std::make_unique<double[]>(7);
    const auto Q = image.Qvec();
    const auto T = image.Tvec();
    pose[0] = Q[0];  // qw
    pose[1] = Q[1];  // qx
    pose[2] = Q[2];  // qy
    pose[3] = Q[3];  // qz
    pose[4] = T[0];  // tx
    pose[5] = T[1];  // ty
    pose[6] = T[2];  // tz

    double* pose_ptr = pose.get();
    owned_base_params.push_back(std::move(pose));
    AddParamBlock(pose_ptr, false);
    image_params[image.ImageId()] =
        ImagePoseParams{pose_ptr, identity_rel_ptr};
  }

  // ========== 5. 构建位姿图边 ==========
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
      auto odom_edges =
          BuildOdometryEdges(reconstruction, options.defaults,
                             options.odom_window);
      // auto odom_edges = BuildOdometryEdges(reconstruction, &database, options.defaults);
      for (const auto& edge : odom_edges) {
        const image_pair_t pair_id =
            Database::ImagePairToPairId(edge.image_id1, edge.image_id2);
        occupied_pairs.insert(pair_id);
      }
      edges.insert(edges.end(), odom_edges.begin(), odom_edges.end());
    }

    // rig约束已通过参数化建模，无需再添加rig边
    if (use_rig) {
      auto rig_edges =
          BuildRigEdges(reconstruction, camera_rigs, options.defaults,
                        &occupied_pairs);
      std::cout << "Rig parameterization enabled, rig edges skipped: "
                << rig_edges.size() << "\n";
    }

    // 自动构建回环边
    if (options.build_loop_edges) {
      auto loop_edges = BuildLoopEdgesSim3(
          reconstruction, &database, options, &occupied_pairs,
          gt_reconstruction.get(), &loop_pose_errors);
      edges.insert(edges.end(), loop_edges.begin(), loop_edges.end());
    }

    // 添加手动指定的回环边
    if (!options.manual_loop_path.empty()) {
      auto manual_edges = BuildManualLoopEdgesWithConstraints(
          reconstruction, &database, options, options.manual_loop_path,
          &occupied_pairs, gt_reconstruction.get(), &loop_pose_errors);
      edges.insert(edges.end(), manual_edges.begin(), manual_edges.end());
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

    // 保存回环相对位姿误差到文件（如果有真值重建）
    if (gt_reconstruction != nullptr && !loop_pose_errors.empty()) {
      const std::string loop_error_path = JoinPaths(options.output_path, "loop_pose_errors.txt");
      std::string error;
      if (!WriteLoopPoseErrorsToFile(loop_error_path, loop_pose_errors, &error)) {
        std::cout << "Failed to write loop pose errors: " << error << "\n";
      } else {
        std::cout << "Wrote loop pose errors to: " << loop_error_path << "\n";
        // 输出统计信息
        double sum_rot = 0.0, sum_trans = 0.0, sum_trans_dir = 0.0;
        for (const auto& err : loop_pose_errors) {
          sum_rot += err.rot_error_deg;
          sum_trans += err.trans_error;
          sum_trans_dir += err.trans_error_normalized;
        }
        const size_t n = loop_pose_errors.size();
        std::cout << "Loop pose error statistics:\n";
        std::cout << "  Total loop edges with GT: " << n << "\n";
        std::cout << "  Mean rotation error: " << (sum_rot / n) << " deg\n";
        std::cout << "  Mean translation error: " << (sum_trans / n) << " m\n";
        std::cout << "  Mean translation direction error: "
                  << (sum_trans_dir / n) << " deg\n";
      }
    }
  }

  // ========== 6. 添加边到优化问题 ==========
  size_t num_loop = 0;
  size_t num_odom = 0;
  size_t num_rig = 0;
  AddEdgesToProblem(edges, image_params, &problem, &num_loop, &num_odom,
                    &num_rig);

  // 固定第一个图像的位姿（消除规范自由度）
  const image_t root_id = reconstruction.RegImageIds().front();
  auto root_it = image_params.find(root_id);
  if (root_it != image_params.end()) {
    problem.SetParameterBlockConstant(root_it->second.base_pose);
  }

  std::cout << "PGO edges: odom=" << num_odom << " loop=" << num_loop
            << " rig=" << num_rig << "\n";

  // ========== 7. 运行Ceres求解器 ==========
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

  // ========== 8. 更新重建中的位姿 ==========
  for (const auto image_id : reconstruction.RegImageIds()) {
    Image& image = reconstruction.Image(image_id);
    const auto it = image_params.find(image_id);
    if (it == image_params.end()) {
      continue;
    }
    const double* base = it->second.base_pose;
    const double* rel = it->second.rel_pose;

    Eigen::Quaterniond q_base(base[0], base[1], base[2], base[3]);
    Eigen::Vector3d t_base(base[4], base[5], base[6]);
    Eigen::Quaterniond q_rel(rel[0], rel[1], rel[2], rel[3]);
    Eigen::Vector3d t_rel(rel[4], rel[5], rel[6]);

    Eigen::Quaterniond q = q_rel * q_base;
    Eigen::Vector3d t = t_rel + q_rel * t_base;
    image.SetQvec(Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()));
    image.SetTvec(t);
  }

  // 如果有真值重建，输出PGO后的回环相对位姿与真值重建中回环的相对位姿的误差
  if (gt_reconstruction != nullptr) {
    const std::string post_pgo_error_path =
        JoinPaths(options.output_path, "loop_pose_errors_after_pgo.txt");
    EvaluateAndWriteLoopPoseErrors(reconstruction, *gt_reconstruction, edges,
                                   post_pgo_error_path);
  }

//   // save pgo result
//   {
//     std::cout << "Save pgo result to " << options.output_path + "_pgo" << "\n";
//     CreateDirIfNotExists(options.output_path + "_pgo", true);
//     reconstruction.Write(options.output_path + "_pgo");
//   }
  // ========== 9. 三角化与迭代优化 ==========
  if (!TriangulateAndOptimizePaperStyle(&reconstruction, &database,
                                        options.rig_config_path, options)) {
    return -1;
  }

  // ========== 10. 评估并保存结果 ==========
  const double post_error = reconstruction.ComputeMeanReprojectionError();
  std::cout << "Mean reprojection error after triangulation/BA: "
            << post_error << " px\n";

  // 检查是否超过误差阈值
  if (options.eval_threshold_px > 0.0 && post_error > options.eval_threshold_px) {
    std::cout << "ERROR: reprojection error exceeds threshold ("
              << options.eval_threshold_px << " px)\n";
    return -1;
  }

  if (!options.image_path.empty()) {
    if (!ExistsDir(options.image_path)) {
      std::cout << "WARNING: image_path does not exist: "
                << options.image_path << "\n";
    } else {
      PrintHeading1("Extracting colors");
      reconstruction.ExtractColorsForAllImages(options.image_path);
    }
  }

  // 保存优化后的重建结果
  reconstruction.Write(options.output_path);
  std::cout << "PGO done. Output: " << options.output_path << "\n";
  return 0;
}
