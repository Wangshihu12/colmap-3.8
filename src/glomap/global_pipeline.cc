// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "glomap/global_pipeline.h"

#include <unordered_set>

#include "base/database_cache.h"

namespace colmap {

GlomapPipeline::GlomapPipeline(
    GlomapPipelineOptions options,
    std::shared_ptr<Database> database,
    std::shared_ptr<ReconstructionManager> reconstruction_manager)
    : options_(std::move(options)),
      database_(std::move(database)),
      reconstruction_manager_(std::move(reconstruction_manager)) {
  CHECK_NOTNULL(database_.get());
  CHECK_NOTNULL(reconstruction_manager_.get());

  // 顶层控制器在构造阶段就把数据库缓存准备好，后续 mapper 可直接复用。
  database_cache_ = std::make_shared<DatabaseCache>();
  database_cache_->Load(*database_,
                        static_cast<size_t>(options_.min_num_matches),
                        options_.ignore_watermarks,
                        std::unordered_set<std::string>(options_.image_names.begin(),
                                                        options_.image_names.end()));
}

bool GlomapPipeline::Run() {
  std::shared_ptr<Reconstruction> reconstruction =
      std::make_shared<Reconstruction>();
  // 顶层参数中的线程数、随机种子等需要显式透传到 mapper 选项。
  GlomapOptions mapper_options = options_.mapper;
  mapper_options.min_num_matches =
      static_cast<size_t>(std::max(options_.min_num_matches, 1));
  mapper_options.num_threads = options_.num_threads;
  mapper_options.random_seed = options_.random_seed;

  GlomapMapper mapper(database_cache_, database_.get());
  mapper.BeginReconstruction(reconstruction);
  if (!mapper.Solve(mapper_options)) {
    return false;
  }

  Reconstruction& output_reconstruction =
      reconstruction_manager_->Get(reconstruction_manager_->Add());
  output_reconstruction = *reconstruction;
  return true;
}

}  // namespace colmap
