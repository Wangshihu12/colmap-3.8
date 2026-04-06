// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#ifndef COLMAP_SRC_GLOMAP_POSE_GRAPH_H_
#define COLMAP_SRC_GLOMAP_POSE_GRAPH_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/database.h"
#include "util/types.h"

namespace colmap {

class Reconstruction;

struct ViewEdge {
  // 两端图像 ID 与 pair ID 共同标识位姿图中的一条边。
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  image_pair_t pair_id = kInvalidImagePairId;
  // weight 通常取两视图内点数，用于 MST 初始化和边可靠性排序。
  double weight = 0.0;
  // valid 表示该边当前是否仍参与后续优化流程。
  bool valid = true;
  // geometry 保存相对位姿、基础矩阵和内点匹配等两视图信息。
  TwoViewGeometry geometry;
};

class PoseGraph {
 public:
  /**
   * @brief 功能描述：从数据库中的两视图几何加载位姿图边。
   * @param database 已打开的 COLMAP 数据库对象。
   * @param reconstruction 当前重建对象，用于检查图像是否存在。
   * @param min_num_matches 最小内点匹配数阈值。
   * @param ignore_watermarks 是否忽略 watermark 图像对。
   * @return 返回值说明：成功加载后返回 true，否则返回 false。
   */
  bool Load(const Database& database,
            const Reconstruction& reconstruction,
            size_t min_num_matches,
            bool ignore_watermarks);

  /**
   * @brief 功能描述：返回全部位姿图边。
   * @return 返回值说明：返回内部边数组的常量引用。
   */
  const std::vector<ViewEdge>& Edges() const;

  /**
   * @brief 功能描述：返回可修改的位姿图边集合。
   * @return 返回值说明：返回内部边数组的可写引用。
   */
  std::vector<ViewEdge>& MutableEdges();

  /**
   * @brief 功能描述：返回当前有效边集合的拷贝。
   * @return 返回值说明：仅包含 `valid=true` 且权重大于零的边。
   */
  std::vector<ViewEdge> ValidEdges() const;

  /**
   * @brief 功能描述：判断位姿图是否为空。
   * @return 返回值说明：没有边时返回 true，否则返回 false。
   */
  bool Empty() const;

  /**
   * @brief 功能描述：统计有效边数量。
   * @return 返回值说明：返回当前有效边条数。
   */
  size_t NumValidEdges() const;

  /**
   * @brief 功能描述：按图像连通性计算最大连通分量。
   * @return 返回值说明：返回最大连通分量中的图像 ID 集合。
   */
  std::unordered_set<image_t> ComputeLargestConnectedImageComponent() const;

  /**
   * @brief 功能描述：将不在活动图像集合中的边标记为无效。
   * @param active_image_ids 允许保留的图像 ID 集合。
   * @return 返回值说明：无返回值。
   */
  void InvalidatePairsOutsideActiveImageIds(
      const std::unordered_set<image_t>& active_image_ids);

 private:
  std::vector<ViewEdge> edges_;
  std::unordered_map<image_pair_t, size_t> pair_id_to_index_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_GLOMAP_POSE_GRAPH_H_
