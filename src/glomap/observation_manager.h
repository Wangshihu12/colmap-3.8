// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#ifndef COLMAP_SRC_GLOMAP_OBSERVATION_MANAGER_H_
#define COLMAP_SRC_GLOMAP_OBSERVATION_MANAGER_H_

#include <unordered_set>

#include "util/types.h"

namespace colmap {

class Reconstruction;

/**
 * @brief 功能描述：对旧版 Reconstruction 提供一层接近新版 COLMAP 的观测管理接口。
 *
 * 当前 COLMAP 3.8 没有新版独立的 ObservationManager 实现，因此这里用一个轻量
 * 兼容层把“删点、过滤、统计”这些职责从主流程里拆出来，便于后续继续向新版结构靠拢。
 */
class ObservationManager {
 public:
  /**
   * @brief 功能描述：构造一个兼容新版 COLMAP 职责划分的观测管理器。
   * @param reconstruction 当前重建对象，负责实际存取观测和三维点。
   * @return 返回值说明：无返回值。
   */
  explicit ObservationManager(class Reconstruction& reconstruction);

  /**
   * @brief 功能描述：删除当前重建中的全部 2D-3D 关联和三维点。
   * @return 返回值说明：无返回值。
   */
  void DeleteAllPoints3D();

  /**
   * @brief 功能描述：过滤全部三维点的重投影误差和三角化角度异常。
   * @param max_reproj_error 最大重投影误差阈值，单位为像素。
   * @param min_tri_angle 最小三角化夹角阈值，单位为度。
   * @return 返回值说明：返回被过滤的观测数量。
   */
  size_t FilterAllPoints3D(double max_reproj_error, double min_tri_angle);

  /**
   * @brief 功能描述：过滤指定图像集合中的三维点观测。
   * @param max_reproj_error 最大重投影误差阈值，单位为像素。
   * @param min_tri_angle 最小三角化夹角阈值，单位为度。
   * @param image_ids 待过滤的图像 ID 集合。
   * @return 返回值说明：返回被过滤的观测数量。
   */
  size_t FilterPoints3DInImages(double max_reproj_error,
                                double min_tri_angle,
                                const std::unordered_set<image_t>& image_ids);

  /**
   * @brief 功能描述：过滤轨迹长度过短的三维点。
   * @param min_track_length 允许保留的最小轨迹长度。
   * @return 返回值说明：返回被删除的观测数量。
   */
  size_t FilterPoints3DWithShortTracks(size_t min_track_length);

  /**
   * @brief 功能描述：过滤负深度观测。
   * @return 返回值说明：返回被删除的观测数量。
   */
  size_t FilterObservationsWithNegativeDepth();

 private:
  class Reconstruction& reconstruction_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_GLOMAP_OBSERVATION_MANAGER_H_
