// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#ifndef COLMAP_SRC_GLOMAP_GLOBAL_PIPELINE_H_
#define COLMAP_SRC_GLOMAP_GLOBAL_PIPELINE_H_

#include <memory>
#include <string>
#include <vector>

#include "base/database.h"
#include "base/reconstruction_manager.h"
#include "glomap/global_mapper.h"

namespace colmap {

struct GlomapPipelineOptions {
  // 进入数据库缓存的最小匹配数阈值。
  int min_num_matches = 15;

  // 是否忽略 watermark 图像对。
  bool ignore_watermarks = true;

  // 若不为空，仅对这些图像名进行重建。
  std::vector<std::string> image_names;

  // 线程数，-1 表示自动。
  int num_threads = -1;

  // 随机种子，-1 表示非确定性。
  int random_seed = -1;

  // 全局 mapper 组件选项。
  GlomapOptions mapper;
};

/**
 * @brief 功能描述：全局 SfM 顶层控制器，职责对齐新版 COLMAP 的 GlobalPipeline。
 *
 * 这个类不直接实现几何算法，而是负责：
 * 1. 建立数据库缓存；
 * 2. 创建输出 Reconstruction；
 * 3. 调用 GlomapMapper 执行完整全局重建；
 * 4. 将结果写入 ReconstructionManager。
 */
class GlomapPipeline {
 public:
  /**
   * @brief 功能描述：构造一个接近新版 COLMAP 的全局管线控制器。
   * @param options 顶层管线配置，包括数据库缓存和 mapper 选项。
   * @param database 数据库对象，用于加载缓存与位姿图。
   * @param reconstruction_manager 输出重建管理器。
   * @return 返回值说明：无返回值。
   */
  GlomapPipeline(GlomapPipelineOptions options,
                 std::shared_ptr<Database> database,
                 std::shared_ptr<ReconstructionManager> reconstruction_manager);

  /**
   * @brief 功能描述：执行完整全局 SfM 管线并输出到重建管理器。
   * @return 返回值说明：成功时返回 true，否则返回 false。
   */
  bool Run();

 private:
  GlomapPipelineOptions options_;
  std::shared_ptr<Database> database_;
  std::shared_ptr<DatabaseCache> database_cache_;
  std::shared_ptr<ReconstructionManager> reconstruction_manager_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_GLOMAP_GLOBAL_PIPELINE_H_
