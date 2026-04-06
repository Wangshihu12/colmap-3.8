// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#ifndef COLMAP_SRC_GLOMAP_VIEW_GRAPH_CALIBRATION_H_
#define COLMAP_SRC_GLOMAP_VIEW_GRAPH_CALIBRATION_H_

#include <vector>

namespace colmap {

class Reconstruction;
struct ViewEdge;

struct ViewGraphCalibrationOptions {
  // 允许优化后的焦距相对原始焦距的最小比值。
  double min_focal_ratio = 0.1;

  // 允许优化后的焦距相对原始焦距的最大比值。
  double max_focal_ratio = 10.0;

  // 用于过滤异常边的两视图残差阈值。
  double max_two_view_error = 2.0;

  // 鲁棒损失尺度。
  double loss_scale = 1e-2;

  // 最大优化迭代次数。
  int max_num_iterations = 100;
};

/**
 * @brief 功能描述：使用 Fetzer 焦距约束校准位姿图中的焦距先验。
 *
 * 该步骤的目标不是直接求出最终最优内参，而是先利用两视图几何的一致性：
 * 1. 校正明显偏差的焦距先验；
 * 2. 根据残差筛除异常边；
 * 3. 为后续 PoseLib 和全局优化提供更干净的位姿图输入。
 * @param options 位姿图焦距校准选项。
 * @param edges 输入输出位姿图边集合，函数会按残差过滤异常边。
 * @param reconstruction 输入输出重建对象，函数会更新其中的相机焦距。
 * @return 返回值说明：成功时返回 true，否则返回 false。
 */
bool CalibrateViewGraph(const ViewGraphCalibrationOptions& options,
                        std::vector<ViewEdge>* edges,
                        Reconstruction* reconstruction);

}  // namespace colmap

#endif  // COLMAP_SRC_GLOMAP_VIEW_GRAPH_CALIBRATION_H_
