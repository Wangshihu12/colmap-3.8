// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "feature/matching.h"

#include <fstream>
#include <numeric>

#include "SiftGPU/SiftGPU.h"
#include "base/gps.h"
#include "feature/utils.h"
#include "retrieval/visual_index.h"
#include "util/cuda.h"
#include "util/misc.h"

namespace colmap {
namespace {

void PrintElapsedTime(const Timer& timer) {
  std::cout << StringPrintf(" in %.3fs", timer.ElapsedSeconds()) << std::endl;
}

/**
 * [功能描述]：将所有图像的特征索引到视觉索引（词汇树）中。
 *             该函数遍历每张图像，提取其关键点和描述子，然后将特征量化为视觉词汇
 *             并添加到倒排索引中，最后计算TF-IDF权重用于图像检索。
 * @param num_threads：并行处理使用的线程数
 * @param num_checks：近似最近邻搜索时的检查次数（影响搜索精度和速度）
 * @param max_num_features：每张图像使用的最大特征数量（0表示不限制）
 * @param image_ids：待索引的图像ID列表
 * @param thread：线程对象指针，用于检查是否被外部停止
 * @param cache：特征缓存对象指针，用于获取图像的关键点和描述子
 * @param visual_index：视觉索引对象指针，用于存储索引结果
 */
void IndexImagesInVisualIndex(const int num_threads, const int num_checks,
                              const int max_num_features,
                              const std::vector<image_t>& image_ids,
                              Thread* thread, FeatureMatcherCache* cache,
                              retrieval::VisualIndex<>* visual_index) {
  // 设置索引选项
  retrieval::VisualIndex<>::IndexOptions index_options;
  index_options.num_threads = num_threads;  // 设置并行线程数
  index_options.num_checks = num_checks;    // 设置近似最近邻搜索的检查次数

  // 遍历所有图像，逐一将特征添加到视觉索引中
  for (size_t i = 0; i < image_ids.size(); ++i) {
    // 检查是否被外部停止（如用户取消操作）
    if (thread->IsStopped()) {
      return;
    }

    // 启动计时器，用于统计单张图像的索引耗时
    Timer timer;
    timer.Start();

    // 打印当前处理进度
    std::cout << StringPrintf("Indexing image [%d/%d]", i + 1, image_ids.size())
              << std::flush;

    // 从缓存中获取当前图像的关键点和描述子
    auto keypoints = *cache->GetKeypoints(image_ids[i]);
    auto descriptors = *cache->GetDescriptors(image_ids[i]);
    
    // 如果特征数量超过限制，只保留尺度最大的特征点
    // 尺度大的特征点通常更加稳定和具有区分性
    if (max_num_features > 0 && descriptors.rows() > max_num_features) {
      ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
    }

    // 将当前图像的特征添加到视觉索引中
    // 这一步会将描述子量化为视觉词汇，并更新倒排索引
    visual_index->Add(index_options, image_ids[i], keypoints, descriptors);

    // 打印当前图像的索引耗时
    PrintElapsedTime(timer);
  }

  // 完成所有图像索引后，计算TF-IDF权重等统计信息
  // TF-IDF用于衡量视觉词汇对图像检索的重要程度
  visual_index->Prepare();
}

/**
 * [功能描述]：在视觉索引中进行最近邻检索并执行特征匹配。
 *             该函数采用生产者-消费者模式：线程池并行执行图像检索（生产者），
 *             主线程对检索到的相似图像对进行特征匹配（消费者）。
 * @param num_threads：并行检索使用的线程数
 * @param num_images：每次查询返回的最大图像数
 * @param num_neighbors：近似最近邻搜索中每个特征的邻居数
 * @param num_checks：近似最近邻搜索的检查次数
 * @param num_images_after_verification：几何验证后保留的图像数量
 * @param max_num_features：每张图像使用的最大特征数量
 * @param image_ids：待匹配的图像ID列表
 * @param thread：线程对象指针，用于检查是否被外部停止
 * @param cache：特征缓存对象指针，用于获取图像的关键点和描述子
 * @param visual_index：视觉索引对象指针，用于执行图像检索
 * @param matcher：SIFT特征匹配器指针，用于执行详细的特征匹配
 */
void MatchNearestNeighborsInVisualIndex(
    const int num_threads, const int num_images, const int num_neighbors,
    const int num_checks, const int num_images_after_verification,
    const int max_num_features, const std::vector<image_t>& image_ids,
    Thread* thread, FeatureMatcherCache* cache,
    retrieval::VisualIndex<>* visual_index, SiftFeatureMatcher* matcher) {
  // 定义检索结果结构体，包含查询图像ID和相似图像得分列表
  struct Retrieval {
    image_t image_id = kInvalidImageId;                   // 查询图像的ID
    std::vector<retrieval::ImageScore> image_scores;      // 检索到的相似图像及其得分
  };

  // 创建线程池用于并行执行图像检索任务
  ThreadPool retrieval_thread_pool(num_threads);
  // 创建任务队列用于存储检索结果（生产者-消费者模式）
  JobQueue<Retrieval> retrieval_queue(num_threads);

  // 设置视觉索引查询选项
  // 注意：描述子的提取需要在此函数外部顺序执行，
  // 以避免并发访问数据库导致的竞态条件
  retrieval::VisualIndex<>::QueryOptions query_options;
  query_options.max_num_images = num_images;                                // 返回的最大图像数
  query_options.num_neighbors = num_neighbors;                              // 每个特征的近邻数
  query_options.num_checks = num_checks;                                    // 近似搜索检查次数
  query_options.num_images_after_verification = num_images_after_verification;  // 几何验证后保留数
  
  // 定义查询函数（Lambda），作为线程池的工作单元
  auto QueryFunc = [&](const image_t image_id) {
    // 从缓存获取关键点和描述子
    auto keypoints = *cache->GetKeypoints(image_id);
    auto descriptors = *cache->GetDescriptors(image_id);
    
    // 如果特征数超过限制，只保留尺度最大的特征
    if (max_num_features > 0 && descriptors.rows() > max_num_features) {
      ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
    }

    // 执行视觉索引查询，获取相似图像列表
    Retrieval retrieval;
    retrieval.image_id = image_id;
    visual_index->Query(query_options, keypoints, descriptors,
                        &retrieval.image_scores);

    // 将检索结果推入队列，供主线程消费
    CHECK(retrieval_queue.Push(std::move(retrieval)));
  };

  // 初始化阶段：预先向线程池提交任务，让所有线程开始工作
  // 提交数量为 min(图像总数, 2*线程数)，确保线程池有足够任务
  size_t image_idx = 0;
  const size_t init_num_tasks =
      std::min(image_ids.size(), 2 * retrieval_thread_pool.NumThreads());
  for (; image_idx < init_num_tasks; ++image_idx) {
    retrieval_thread_pool.AddTask(QueryFunc, image_ids[image_idx]);
  }

  // 存储待匹配的图像对
  std::vector<std::pair<image_t, image_t>> image_pairs;

  // 主循环：从队列中获取检索结果，并执行特征匹配
  for (size_t i = 0; i < image_ids.size(); ++i) {
    // 检查是否被外部停止
    if (thread->IsStopped()) {
      retrieval_queue.Stop();
      return;
    }

    // 启动计时器
    Timer timer;
    timer.Start();

    // 打印当前匹配进度
    std::cout << StringPrintf("Matching image [%d/%d]", i + 1, image_ids.size())
              << std::flush;

    // 向线程池提交下一个检索任务（保持流水线运行）
    if (image_idx < image_ids.size()) {
      retrieval_thread_pool.AddTask(QueryFunc, image_ids[image_idx]);
      image_idx += 1;
    }

    // 从队列中取出下一个检索结果（阻塞等待）
    auto retrieval = retrieval_queue.Pop();
    CHECK(retrieval.IsValid());

    // 获取检索结果中的图像ID和相似图像得分列表
    const auto& image_id = retrieval.Data().image_id;
    const auto& image_scores = retrieval.Data().image_scores;

    // 根据检索得分构建图像对列表
    // 每个图像对由查询图像和检索到的相似图像组成
    image_pairs.clear();
    image_pairs.reserve(image_scores.size());
    for (const auto image_score : image_scores) {
      image_pairs.emplace_back(image_id, image_score.image_id);
    }

    // 对构建的图像对执行详细的SIFT特征匹配
    matcher->Match(image_pairs);

    // 打印当前图像的匹配耗时
    PrintElapsedTime(timer);
  }
}

}  // namespace

bool ExhaustiveMatchingOptions::Check() const {
  CHECK_OPTION_GT(block_size, 1);
  return true;
}

bool SequentialMatchingOptions::Check() const {
  CHECK_OPTION_GT(overlap, 0);
  CHECK_OPTION_GT(loop_detection_period, 0);
  CHECK_OPTION_GT(loop_detection_num_images, 0);
  CHECK_OPTION_GT(loop_detection_num_nearest_neighbors, 0);
  CHECK_OPTION_GT(loop_detection_num_checks, 0);
  return true;
}

bool VocabTreeMatchingOptions::Check() const {
  CHECK_OPTION_GT(num_images, 0);
  CHECK_OPTION_GT(num_nearest_neighbors, 0);
  CHECK_OPTION_GT(num_checks, 0);
  return true;
}

bool SpatialMatchingOptions::Check() const {
  CHECK_OPTION_GT(max_num_neighbors, 0);
  CHECK_OPTION_GT(max_distance, 0.0);
  return true;
}

bool TransitiveMatchingOptions::Check() const {
  CHECK_OPTION_GT(batch_size, 0);
  CHECK_OPTION_GT(num_iterations, 0);
  return true;
}

bool ImagePairsMatchingOptions::Check() const {
  CHECK_OPTION_GT(block_size, 0);
  return true;
}

bool FeaturePairsMatchingOptions::Check() const { return true; }

FeatureMatcherCache::FeatureMatcherCache(const size_t cache_size,
                                         const Database* database)
    : cache_size_(cache_size), database_(database) {
  CHECK_NOTNULL(database_);
}

void FeatureMatcherCache::Setup() {
  const std::vector<Camera> cameras = database_->ReadAllCameras();
  cameras_cache_.reserve(cameras.size());
  for (const auto& camera : cameras) {
    cameras_cache_.emplace(camera.CameraId(), camera);
  }

  const std::vector<Image> images = database_->ReadAllImages();
  images_cache_.reserve(images.size());
  for (const auto& image : images) {
    images_cache_.emplace(image.ImageId(), image);
  }

  keypoints_cache_ = std::make_unique<LRUCache<image_t, FeatureKeypointsPtr>>(
      cache_size_, [this](const image_t image_id) {
        return std::make_shared<FeatureKeypoints>(
            database_->ReadKeypoints(image_id));
      });

  descriptors_cache_ =
      std::make_unique<LRUCache<image_t, FeatureDescriptorsPtr>>(
          cache_size_, [this](const image_t image_id) {
            return std::make_shared<FeatureDescriptors>(
                database_->ReadDescriptors(image_id));
          });

  keypoints_exists_cache_ = std::make_unique<LRUCache<image_t, bool>>(
      images.size(), [this](const image_t image_id) {
        return database_->ExistsKeypoints(image_id);
      });

  descriptors_exists_cache_ = std::make_unique<LRUCache<image_t, bool>>(
      images.size(), [this](const image_t image_id) {
        return database_->ExistsDescriptors(image_id);
      });
}

const Camera& FeatureMatcherCache::GetCamera(const camera_t camera_id) const {
  return cameras_cache_.at(camera_id);
}

const Image& FeatureMatcherCache::GetImage(const image_t image_id) const {
  return images_cache_.at(image_id);
}

FeatureKeypointsPtr FeatureMatcherCache::GetKeypoints(const image_t image_id) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  return keypoints_cache_->Get(image_id);
}

FeatureDescriptorsPtr FeatureMatcherCache::GetDescriptors(
    const image_t image_id) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  return descriptors_cache_->Get(image_id);
}

FeatureMatches FeatureMatcherCache::GetMatches(const image_t image_id1,
                                               const image_t image_id2) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  return database_->ReadMatches(image_id1, image_id2);
}

std::vector<image_t> FeatureMatcherCache::GetImageIds() const {
  std::vector<image_t> image_ids;
  image_ids.reserve(images_cache_.size());
  for (const auto& image : images_cache_) {
    image_ids.push_back(image.first);
  }
  return image_ids;
}

bool FeatureMatcherCache::ExistsKeypoints(const image_t image_id) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  return keypoints_exists_cache_->Get(image_id);
}

bool FeatureMatcherCache::ExistsDescriptors(const image_t image_id) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  return descriptors_exists_cache_->Get(image_id);
}

bool FeatureMatcherCache::ExistsMatches(const image_t image_id1,
                                        const image_t image_id2) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  return database_->ExistsMatches(image_id1, image_id2);
}

bool FeatureMatcherCache::ExistsInlierMatches(const image_t image_id1,
                                              const image_t image_id2) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  return database_->ExistsInlierMatches(image_id1, image_id2);
}

void FeatureMatcherCache::WriteMatches(const image_t image_id1,
                                       const image_t image_id2,
                                       const FeatureMatches& matches) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  database_->WriteMatches(image_id1, image_id2, matches);
}

void FeatureMatcherCache::WriteTwoViewGeometry(
    const image_t image_id1, const image_t image_id2,
    const TwoViewGeometry& two_view_geometry) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  database_->WriteTwoViewGeometry(image_id1, image_id2, two_view_geometry);
}

void FeatureMatcherCache::DeleteMatches(const image_t image_id1,
                                        const image_t image_id2) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  database_->DeleteMatches(image_id1, image_id2);
}

void FeatureMatcherCache::DeleteInlierMatches(const image_t image_id1,
                                              const image_t image_id2) {
  std::unique_lock<std::mutex> lock(database_mutex_);
  database_->DeleteInlierMatches(image_id1, image_id2);
}

FeatureMatcherThread::FeatureMatcherThread(const SiftMatchingOptions& options,
                                           FeatureMatcherCache* cache)
    : options_(options), cache_(cache) {}

void FeatureMatcherThread::SetMaxNumMatches(const int max_num_matches) {
  options_.max_num_matches = max_num_matches;
}

SiftCPUFeatureMatcher::SiftCPUFeatureMatcher(const SiftMatchingOptions& options,
                                             FeatureMatcherCache* cache,
                                             JobQueue<Input>* input_queue,
                                             JobQueue<Output>* output_queue)
    : FeatureMatcherThread(options, cache),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  CHECK(options_.Check());
}

void SiftCPUFeatureMatcher::Run() {
  SignalValidSetup();

  while (true) {
    if (IsStopped()) {
      break;
    }

    auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto& data = input_job.Data();

      if (!cache_->ExistsDescriptors(data.image_id1) ||
          !cache_->ExistsDescriptors(data.image_id2)) {
        CHECK(output_queue_->Push(std::move(data)));
        continue;
      }

      const auto descriptors1 = cache_->GetDescriptors(data.image_id1);
      const auto descriptors2 = cache_->GetDescriptors(data.image_id2);
      MatchSiftFeaturesCPU(options_, *descriptors1, *descriptors2,
                           &data.matches);

      CHECK(output_queue_->Push(std::move(data)));
    }
  }
}

SiftGPUFeatureMatcher::SiftGPUFeatureMatcher(const SiftMatchingOptions& options,
                                             FeatureMatcherCache* cache,
                                             JobQueue<Input>* input_queue,
                                             JobQueue<Output>* output_queue)
    : FeatureMatcherThread(options, cache),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  CHECK(options_.Check());

  prev_uploaded_image_ids_[0] = kInvalidImageId;
  prev_uploaded_image_ids_[1] = kInvalidImageId;

#ifndef CUDA_ENABLED
  opengl_context_ = std::make_unique<OpenGLContextManager>();
#endif
}

void SiftGPUFeatureMatcher::Run() {
#ifndef CUDA_ENABLED
  CHECK(opengl_context_);
  CHECK(opengl_context_->MakeCurrent());
#endif

  SiftMatchGPU sift_match_gpu;
  if (!CreateSiftGPUMatcher(options_, &sift_match_gpu)) {
    std::cout << "ERROR: SiftGPU not fully supported" << std::endl;
    SignalInvalidSetup();
    return;
  }

  SignalValidSetup();

  while (true) {
    if (IsStopped()) {
      break;
    }

    auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto& data = input_job.Data();

      if (!cache_->ExistsDescriptors(data.image_id1) ||
          !cache_->ExistsDescriptors(data.image_id2)) {
        CHECK(output_queue_->Push(std::move(data)));
        continue;
      }

      const FeatureDescriptors* descriptors1_ptr;
      GetDescriptorData(0, data.image_id1, &descriptors1_ptr);
      const FeatureDescriptors* descriptors2_ptr;
      GetDescriptorData(1, data.image_id2, &descriptors2_ptr);
      MatchSiftFeaturesGPU(options_, descriptors1_ptr, descriptors2_ptr,
                           &sift_match_gpu, &data.matches);

      CHECK(output_queue_->Push(std::move(data)));
    }
  }
}

void SiftGPUFeatureMatcher::GetDescriptorData(
    const int index, const image_t image_id,
    const FeatureDescriptors** descriptors_ptr) {
  CHECK_GE(index, 0);
  CHECK_LE(index, 1);
  if (prev_uploaded_image_ids_[index] == image_id) {
    *descriptors_ptr = nullptr;
  } else {
    prev_uploaded_descriptors_[index] = cache_->GetDescriptors(image_id);
    *descriptors_ptr = prev_uploaded_descriptors_[index].get();
    prev_uploaded_image_ids_[index] = image_id;
  }
}

GuidedSiftCPUFeatureMatcher::GuidedSiftCPUFeatureMatcher(
    const SiftMatchingOptions& options, FeatureMatcherCache* cache,
    JobQueue<Input>* input_queue, JobQueue<Output>* output_queue)
    : FeatureMatcherThread(options, cache),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  CHECK(options_.Check());
}

void GuidedSiftCPUFeatureMatcher::Run() {
  SignalValidSetup();

  while (true) {
    if (IsStopped()) {
      break;
    }

    auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto& data = input_job.Data();

      if (data.two_view_geometry.inlier_matches.size() <
          static_cast<size_t>(options_.min_num_inliers)) {
        CHECK(output_queue_->Push(std::move(data)));
        continue;
      }

      if (!cache_->ExistsKeypoints(data.image_id1) ||
          !cache_->ExistsKeypoints(data.image_id2) ||
          !cache_->ExistsDescriptors(data.image_id1) ||
          !cache_->ExistsDescriptors(data.image_id2)) {
        CHECK(output_queue_->Push(std::move(data)));
        continue;
      }

      const auto keypoints1 = cache_->GetKeypoints(data.image_id1);
      const auto keypoints2 = cache_->GetKeypoints(data.image_id2);
      const auto descriptors1 = cache_->GetDescriptors(data.image_id1);
      const auto descriptors2 = cache_->GetDescriptors(data.image_id2);
      MatchGuidedSiftFeaturesCPU(options_, *keypoints1, *keypoints2,
                                 *descriptors1, *descriptors2,
                                 &data.two_view_geometry);

      CHECK(output_queue_->Push(std::move(data)));
    }
  }
}

GuidedSiftGPUFeatureMatcher::GuidedSiftGPUFeatureMatcher(
    const SiftMatchingOptions& options, FeatureMatcherCache* cache,
    JobQueue<Input>* input_queue, JobQueue<Output>* output_queue)
    : FeatureMatcherThread(options, cache),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  CHECK(options_.Check());

  prev_uploaded_image_ids_[0] = kInvalidImageId;
  prev_uploaded_image_ids_[1] = kInvalidImageId;

#ifndef CUDA_ENABLED
  opengl_context_ = std::make_unique<OpenGLContextManager>();
#endif
}

void GuidedSiftGPUFeatureMatcher::Run() {
#ifndef CUDA_ENABLED
  CHECK(opengl_context_);
  CHECK(opengl_context_->MakeCurrent());
#endif

  SiftMatchGPU sift_match_gpu;
  if (!CreateSiftGPUMatcher(options_, &sift_match_gpu)) {
    std::cout << "ERROR: SiftGPU not fully supported" << std::endl;
    SignalInvalidSetup();
    return;
  }

  SignalValidSetup();

  while (true) {
    if (IsStopped()) {
      break;
    }

    auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto& data = input_job.Data();

      if (data.two_view_geometry.inlier_matches.size() <
          static_cast<size_t>(options_.min_num_inliers)) {
        CHECK(output_queue_->Push(std::move(data)));
        continue;
      }

      if (!cache_->ExistsKeypoints(data.image_id1) ||
          !cache_->ExistsKeypoints(data.image_id2) ||
          !cache_->ExistsDescriptors(data.image_id1) ||
          !cache_->ExistsDescriptors(data.image_id2)) {
        CHECK(output_queue_->Push(std::move(data)));
        continue;
      }

      const FeatureDescriptors* descriptors1_ptr;
      const FeatureKeypoints* keypoints1_ptr;
      GetFeatureData(0, data.image_id1, &keypoints1_ptr, &descriptors1_ptr);
      const FeatureDescriptors* descriptors2_ptr;
      const FeatureKeypoints* keypoints2_ptr;
      GetFeatureData(1, data.image_id2, &keypoints2_ptr, &descriptors2_ptr);

      MatchGuidedSiftFeaturesGPU(options_, keypoints1_ptr, keypoints2_ptr,
                                 descriptors1_ptr, descriptors2_ptr,
                                 &sift_match_gpu, &data.two_view_geometry);

      CHECK(output_queue_->Push(std::move(data)));
    }
  }
}

void GuidedSiftGPUFeatureMatcher::GetFeatureData(
    const int index, const image_t image_id,
    const FeatureKeypoints** keypoints_ptr,
    const FeatureDescriptors** descriptors_ptr) {
  CHECK_GE(index, 0);
  CHECK_LE(index, 1);
  if (prev_uploaded_image_ids_[index] == image_id) {
    *keypoints_ptr = nullptr;
    *descriptors_ptr = nullptr;
  } else {
    prev_uploaded_keypoints_[index] = cache_->GetKeypoints(image_id);
    prev_uploaded_descriptors_[index] = cache_->GetDescriptors(image_id);
    *keypoints_ptr = prev_uploaded_keypoints_[index].get();
    *descriptors_ptr = prev_uploaded_descriptors_[index].get();
    prev_uploaded_image_ids_[index] = image_id;
  }
}

TwoViewGeometryVerifier::TwoViewGeometryVerifier(
    const SiftMatchingOptions& options, FeatureMatcherCache* cache,
    JobQueue<Input>* input_queue, JobQueue<Output>* output_queue)
    : options_(options),
      cache_(cache),
      input_queue_(input_queue),
      output_queue_(output_queue) {
  CHECK(options_.Check());

  two_view_geometry_options_.min_num_inliers =
      static_cast<size_t>(options_.min_num_inliers);
  two_view_geometry_options_.ransac_options.max_error = options_.max_error;
  two_view_geometry_options_.ransac_options.confidence = options_.confidence;
  two_view_geometry_options_.ransac_options.min_num_trials =
      static_cast<size_t>(options_.min_num_trials);
  two_view_geometry_options_.ransac_options.max_num_trials =
      static_cast<size_t>(options_.max_num_trials);
  two_view_geometry_options_.ransac_options.min_inlier_ratio =
      options_.min_inlier_ratio;
  two_view_geometry_options_.force_H_use = options_.planar_scene;
  two_view_geometry_options_.compute_relative_pose = options_.compute_relative_pose;
}

void TwoViewGeometryVerifier::Run() {
  while (true) {
    if (IsStopped()) {
      break;
    }

    auto input_job = input_queue_->Pop();
    if (input_job.IsValid()) {
      auto& data = input_job.Data();

      if (data.matches.size() < static_cast<size_t>(options_.min_num_inliers)) {
        CHECK(output_queue_->Push(std::move(data)));
        continue;
      }

      const auto& camera1 =
          cache_->GetCamera(cache_->GetImage(data.image_id1).CameraId());
      const auto& camera2 =
          cache_->GetCamera(cache_->GetImage(data.image_id2).CameraId());
      const auto keypoints1 = cache_->GetKeypoints(data.image_id1);
      const auto keypoints2 = cache_->GetKeypoints(data.image_id2);
      const auto& points1 = FeatureKeypointsToPointsVector(*keypoints1);
      const auto& points2 = FeatureKeypointsToPointsVector(*keypoints2);

      if (options_.multiple_models) {
        data.two_view_geometry.EstimateMultiple(camera1, points1, camera2,
                                                points2, data.matches,
                                                two_view_geometry_options_);
      } else {
        data.two_view_geometry.Estimate(camera1, points1, camera2, points2,
                                        data.matches,
                                        two_view_geometry_options_);
      }

      CHECK(output_queue_->Push(std::move(data)));
    }
  }
}

SiftFeatureMatcher::SiftFeatureMatcher(const SiftMatchingOptions& options,
                                       Database* database,
                                       FeatureMatcherCache* cache)
    : options_(options), database_(database), cache_(cache), is_setup_(false) {
  CHECK(options_.Check());

  const int num_threads = GetEffectiveNumThreads(options_.num_threads);
  CHECK_GT(num_threads, 0);

  std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
  CHECK_GT(gpu_indices.size(), 0);

#ifdef CUDA_ENABLED
  if (options_.use_gpu && gpu_indices.size() == 1 && gpu_indices[0] == -1) {
    const int num_cuda_devices = GetNumCudaDevices();
    CHECK_GT(num_cuda_devices, 0);
    gpu_indices.resize(num_cuda_devices);
    std::iota(gpu_indices.begin(), gpu_indices.end(), 0);
  }
#endif  // CUDA_ENABLED

  if (options_.use_gpu) {
    auto gpu_options = options_;
    matchers_.reserve(gpu_indices.size());
    for (const auto& gpu_index : gpu_indices) {
      gpu_options.gpu_index = std::to_string(gpu_index);
      matchers_.emplace_back(std::make_unique<SiftGPUFeatureMatcher>(
          gpu_options, cache, &matcher_queue_, &verifier_queue_));
    }
  } else {
    matchers_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      matchers_.emplace_back(std::make_unique<SiftCPUFeatureMatcher>(
          options_, cache, &matcher_queue_, &verifier_queue_));
    }
  }

  verifiers_.reserve(num_threads);
  if (options_.guided_matching) {
    for (int i = 0; i < num_threads; ++i) {
      verifiers_.emplace_back(std::make_unique<TwoViewGeometryVerifier>(
          options_, cache, &verifier_queue_, &guided_matcher_queue_));
    }

    if (options_.use_gpu) {
      auto gpu_options = options_;
      guided_matchers_.reserve(gpu_indices.size());
      for (const auto& gpu_index : gpu_indices) {
        gpu_options.gpu_index = std::to_string(gpu_index);
        guided_matchers_.emplace_back(
            std::make_unique<GuidedSiftGPUFeatureMatcher>(
                gpu_options, cache, &guided_matcher_queue_, &output_queue_));
      }
    } else {
      guided_matchers_.reserve(num_threads);
      for (int i = 0; i < num_threads; ++i) {
        guided_matchers_.emplace_back(
            std::make_unique<GuidedSiftCPUFeatureMatcher>(
                options_, cache, &guided_matcher_queue_, &output_queue_));
      }
    }
  } else {
    for (int i = 0; i < num_threads; ++i) {
      verifiers_.emplace_back(std::make_unique<TwoViewGeometryVerifier>(
          options_, cache, &verifier_queue_, &output_queue_));
    }
  }
}

SiftFeatureMatcher::~SiftFeatureMatcher() {
  matcher_queue_.Wait();
  verifier_queue_.Wait();
  guided_matcher_queue_.Wait();
  output_queue_.Wait();

  for (auto& matcher : matchers_) {
    matcher->Stop();
  }

  for (auto& verifier : verifiers_) {
    verifier->Stop();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->Stop();
  }

  matcher_queue_.Stop();
  verifier_queue_.Stop();
  guided_matcher_queue_.Stop();
  output_queue_.Stop();

  for (auto& matcher : matchers_) {
    matcher->Wait();
  }

  for (auto& verifier : verifiers_) {
    verifier->Wait();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->Wait();
  }
}

bool SiftFeatureMatcher::Setup() {
  const int max_num_features = CHECK_NOTNULL(database_)->MaxNumDescriptors();
  options_.max_num_matches =
      std::min(options_.max_num_matches, max_num_features);

  for (auto& matcher : matchers_) {
    matcher->SetMaxNumMatches(options_.max_num_matches);
    matcher->Start();
  }

  for (auto& verifier : verifiers_) {
    verifier->Start();
  }

  for (auto& guided_matcher : guided_matchers_) {
    guided_matcher->SetMaxNumMatches(options_.max_num_matches);
    guided_matcher->Start();
  }

  for (auto& matcher : matchers_) {
    if (!matcher->CheckValidSetup()) {
      return false;
    }
  }

  for (auto& guided_matcher : guided_matchers_) {
    if (!guided_matcher->CheckValidSetup()) {
      return false;
    }
  }

  is_setup_ = true;

  return true;
}

void SiftFeatureMatcher::Match(
    const std::vector<std::pair<image_t, image_t>>& image_pairs) {
  CHECK_NOTNULL(database_);
  CHECK_NOTNULL(cache_);
  CHECK(is_setup_);

  if (image_pairs.empty()) {
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Match the image pairs
  //////////////////////////////////////////////////////////////////////////////

  std::unordered_set<image_pair_t> image_pair_ids;
  image_pair_ids.reserve(image_pairs.size());

  size_t num_outputs = 0;
  for (const auto& image_pair : image_pairs) {
    // Avoid self-matches.
    if (image_pair.first == image_pair.second) {
      continue;
    }

    // Avoid duplicate image pairs.
    const image_pair_t pair_id =
        Database::ImagePairToPairId(image_pair.first, image_pair.second);
    if (image_pair_ids.count(pair_id) > 0) {
      continue;
    }

    image_pair_ids.insert(pair_id);

    const bool exists_matches =
        cache_->ExistsMatches(image_pair.first, image_pair.second);
    const bool exists_inlier_matches =
        cache_->ExistsInlierMatches(image_pair.first, image_pair.second);

    if (exists_matches && exists_inlier_matches) {
      continue;
    }

    num_outputs += 1;

    // If only one of the matches or inlier matches exist, we recompute them
    // from scratch and delete the existing results. This must be done before
    // pushing the jobs to the queue, otherwise database constraints might fail
    // when writing an existing result into the database.

    if (exists_inlier_matches) {
      cache_->DeleteInlierMatches(image_pair.first, image_pair.second);
    }

    internal::FeatureMatcherData data;
    data.image_id1 = image_pair.first;
    data.image_id2 = image_pair.second;

    if (exists_matches) {
      data.matches = cache_->GetMatches(image_pair.first, image_pair.second);
      cache_->DeleteMatches(image_pair.first, image_pair.second);
      CHECK(verifier_queue_.Push(std::move(data)));
    } else {
      CHECK(matcher_queue_.Push(std::move(data)));
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Write results to database
  //////////////////////////////////////////////////////////////////////////////

  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_job = output_queue_.Pop();
    CHECK(output_job.IsValid());
    auto& output = output_job.Data();

    if (output.matches.size() < static_cast<size_t>(options_.min_num_inliers)) {
      output.matches = {};
    }

    if (output.two_view_geometry.inlier_matches.size() <
        static_cast<size_t>(options_.min_num_inliers)) {
      output.two_view_geometry = TwoViewGeometry();
    }

    cache_->WriteMatches(output.image_id1, output.image_id2, output.matches);
    cache_->WriteTwoViewGeometry(output.image_id1, output.image_id2,
                                 output.two_view_geometry);
  }

  CHECK_EQ(output_queue_.Size(), 0);
}

ExhaustiveFeatureMatcher::ExhaustiveFeatureMatcher(
    const ExhaustiveMatchingOptions& options,
    const SiftMatchingOptions& match_options, const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(5 * options_.block_size, &database_),
      matcher_(match_options, &database_, &cache_) {
  CHECK(options_.Check());
  CHECK(match_options_.Check());
}

void ExhaustiveFeatureMatcher::Run() {
  PrintHeading1("Exhaustive feature matching");

  if (!matcher_.Setup()) {
    return;
  }

  cache_.Setup();

  const std::vector<image_t> image_ids = cache_.GetImageIds();

  const size_t block_size = static_cast<size_t>(options_.block_size);
  const size_t num_blocks = static_cast<size_t>(
      std::ceil(static_cast<double>(image_ids.size()) / block_size));
  const size_t num_pairs_per_block = block_size * (block_size - 1) / 2;

  std::vector<std::pair<image_t, image_t>> image_pairs;
  image_pairs.reserve(num_pairs_per_block);

  for (size_t start_idx1 = 0; start_idx1 < image_ids.size();
       start_idx1 += block_size) {
    const size_t end_idx1 =
        std::min(image_ids.size(), start_idx1 + block_size) - 1;
    for (size_t start_idx2 = 0; start_idx2 < image_ids.size();
         start_idx2 += block_size) {
      const size_t end_idx2 =
          std::min(image_ids.size(), start_idx2 + block_size) - 1;

      if (IsStopped()) {
        GetTimer().PrintMinutes();
        return;
      }

      Timer timer;
      timer.Start();

      std::cout << StringPrintf("Matching block [%d/%d, %d/%d]",
                                start_idx1 / block_size + 1, num_blocks,
                                start_idx2 / block_size + 1, num_blocks)
                << std::flush;

      image_pairs.clear();
      for (size_t idx1 = start_idx1; idx1 <= end_idx1; ++idx1) {
        for (size_t idx2 = start_idx2; idx2 <= end_idx2; ++idx2) {
          const size_t block_id1 = idx1 % block_size;
          const size_t block_id2 = idx2 % block_size;
          if ((idx1 > idx2 && block_id1 <= block_id2) ||
              (idx1 < idx2 &&
               block_id1 < block_id2)) {  // Avoid duplicate pairs
            image_pairs.emplace_back(image_ids[idx1], image_ids[idx2]);
          }
        }
      }

      DatabaseTransaction database_transaction(&database_);
      matcher_.Match(image_pairs);

      PrintElapsedTime(timer);
    }
  }

  GetTimer().PrintMinutes();
}

SequentialFeatureMatcher::SequentialFeatureMatcher(
    const SequentialMatchingOptions& options,
    const SiftMatchingOptions& match_options, const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(std::max(5 * options_.loop_detection_num_images,
                      5 * options_.overlap),
             &database_),
      matcher_(match_options, &database_, &cache_) {
  CHECK(options_.Check());
  CHECK(match_options_.Check());
}

/**
 * [功能描述]：顺序特征匹配器的主运行函数。
 *             该函数用于处理按时间或空间顺序采集的图像序列（如视频帧），
 *             先对相邻图像进行特征匹配，然后可选地执行回环检测以建立
 *             非相邻图像之间的匹配关系。
 */
void SequentialFeatureMatcher::Run() {
  // 打印标题信息
  PrintHeading1("Sequential feature matching");

  // 初始化特征匹配器，如果失败则直接返回
  if (!matcher_.Setup()) {
    return;
  }

  // 初始化特征缓存
  cache_.Setup();

  // 获取按顺序排列的图像ID列表
  // 图像顺序通常由文件名或数据库中的记录顺序决定
  const std::vector<image_t> ordered_image_ids = GetOrderedImageIds();

  // 执行顺序匹配：对相邻的图像对进行特征匹配
  // 例如：图像1与图像2匹配，图像2与图像3匹配，以此类推
  RunSequentialMatching(ordered_image_ids);
  
  // 如果启用了回环检测，则执行回环检测
  // 回环检测用于发现序列中相隔较远但场景相似的图像
  if (options_.loop_detection) {
    RunLoopDetection(ordered_image_ids);
  }

  // 打印总耗时
  GetTimer().PrintMinutes();
}

std::vector<image_t> SequentialFeatureMatcher::GetOrderedImageIds() const {
  const std::vector<image_t> image_ids = cache_.GetImageIds();

  std::vector<Image> ordered_images;
  ordered_images.reserve(image_ids.size());
  for (const auto image_id : image_ids) {
    ordered_images.push_back(cache_.GetImage(image_id));
  }

  std::sort(ordered_images.begin(), ordered_images.end(),
            [](const Image& image1, const Image& image2) {
              return image1.Name() < image2.Name();
            });

  std::vector<image_t> ordered_image_ids;
  ordered_image_ids.reserve(image_ids.size());
  for (const auto& image : ordered_images) {
    ordered_image_ids.push_back(image.ImageId());
  }

  return ordered_image_ids;
}

void SequentialFeatureMatcher::RunSequentialMatching(
    const std::vector<image_t>& image_ids) {
  std::vector<std::pair<image_t, image_t>> image_pairs;
  image_pairs.reserve(options_.overlap);

  for (size_t image_idx1 = 0; image_idx1 < image_ids.size(); ++image_idx1) {
    if (IsStopped()) {
      return;
    }

    const auto image_id1 = image_ids.at(image_idx1);

    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Matching image [%d/%d]", image_idx1 + 1,
                              image_ids.size())
              << std::flush;

    image_pairs.clear();
    for (int i = 0; i < options_.overlap; ++i) {
      const size_t image_idx2 = image_idx1 + i;
      if (image_idx2 < image_ids.size()) {
        image_pairs.emplace_back(image_id1, image_ids.at(image_idx2));
        if (options_.quadratic_overlap) {
          const size_t image_idx2_quadratic = image_idx1 + (1 << i);
          if (image_idx2_quadratic < image_ids.size()) {
            image_pairs.emplace_back(image_id1,
                                     image_ids.at(image_idx2_quadratic));
          }
        }
      } else {
        break;
      }
    }

    DatabaseTransaction database_transaction(&database_);
    matcher_.Match(image_pairs);

    PrintElapsedTime(timer);
  }
}

/**
 * [功能描述]：执行回环检测（Loop Detection）。
 *             回环检测用于检测相机是否回到了之前访问过的位置，
 *             通过建立非相邻图像之间的匹配关系来减少累积漂移误差。
 *             该函数使用词汇树进行高效的图像相似度检索。
 * @param image_ids：按拍摄顺序排列的图像ID列表
 */
void SequentialFeatureMatcher::RunLoopDetection(
    const std::vector<image_t>& image_ids) {
  // 从磁盘读取预训练的词汇树文件
  retrieval::VisualIndex<> visual_index;
  visual_index.Read(options_.vocab_tree_path);

  // 将所有图像的特征索引到视觉索引中
  // 这一步构建倒排索引，用于后续的快速图像检索
  IndexImagesInVisualIndex(match_options_.num_threads,              // 并行线程数
                           options_.loop_detection_num_checks,      // 近似搜索检查次数
                           options_.loop_detection_max_num_features,  // 每张图像的最大特征数
                           image_ids,                               // 所有图像ID
                           this, &cache_, &visual_index);

  // 检查是否被外部停止
  if (IsStopped()) {
    return;
  }

  // 按周期采样图像，只对每隔 n 张图像执行回环检测
  // 这样可以减少计算量，同时仍能有效检测回环
  // loop_detection_period 定义了采样间隔
  std::vector<image_t> match_image_ids;
  for (size_t i = 0; i < image_ids.size();
       i += options_.loop_detection_period) {
    match_image_ids.push_back(image_ids[i]);
  }

  // 对采样的图像执行最近邻检索和特征匹配
  // 通过词汇树找到与当前图像最相似的历史图像，建立回环约束
  MatchNearestNeighborsInVisualIndex(
      match_options_.num_threads,                           // 并行线程数
      options_.loop_detection_num_images,                   // 返回的最大图像数
      options_.loop_detection_num_nearest_neighbors,        // 每个特征的近邻数
      options_.loop_detection_num_checks,                   // 近似搜索检查次数
      options_.loop_detection_num_images_after_verification,  // 几何验证后保留的图像数
      options_.loop_detection_max_num_features,             // 每张图像的最大特征数
      match_image_ids,                                      // 待匹配的采样图像
      this, &cache_, &visual_index, &matcher_);
}

VocabTreeFeatureMatcher::VocabTreeFeatureMatcher(
    const VocabTreeMatchingOptions& options,
    const SiftMatchingOptions& match_options, const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(5 * options_.num_images, &database_),
      matcher_(match_options, &database_, &cache_) {
  CHECK(options_.Check());
  CHECK(match_options_.Check());
}

/**
 * [功能描述]：词汇树特征匹配器的核心运行函数。
 *             该函数使用预训练的词汇树（Vocabulary Tree）进行图像检索和特征匹配，
 *             通过视觉词汇索引快速找到相似图像对，然后对这些图像对进行详细的特征匹配。
 */
void VocabTreeFeatureMatcher::Run() {
  // 打印标题信息
  PrintHeading1("Vocabulary tree feature matching");

  // 初始化特征匹配器，如果失败则直接返回
  if (!matcher_.Setup()) {
    return;
  }

  // 初始化特征缓存（用于缓存图像特征数据）
  cache_.Setup();

  // 从磁盘读取预训练的词汇树文件
  // 词汇树是通过聚类大量SIFT描述子构建的层次化视觉词汇表
  retrieval::VisualIndex<> visual_index;
  visual_index.Read(options_.vocab_tree_path);

  // 获取数据库中所有图像的ID列表
  const std::vector<image_t> all_image_ids = cache_.GetImageIds();
  
  // 待匹配的图像ID列表（可能是全部图像或用户指定的子集）
  std::vector<image_t> image_ids;
  
  if (options_.match_list_path == "") {
    // 如果没有指定匹配列表文件，则使用所有图像
    image_ids = cache_.GetImageIds();
  } else {
    // 如果指定了匹配列表文件，则只匹配列表中的图像
    
    // 构建图像名称到图像ID的映射表，用于快速查找
    std::unordered_map<std::string, image_t> image_name_to_image_id;
    image_name_to_image_id.reserve(all_image_ids.size());
    for (const auto image_id : all_image_ids) {
      const auto& image = cache_.GetImage(image_id);
      image_name_to_image_id.emplace(image.Name(), image_id);
    }

    // 读取匹配列表文件，文件中每行一个图像名称
    std::ifstream file(options_.match_list_path);
    CHECK(file.is_open()) << options_.match_list_path;
    std::string line;
    while (std::getline(file, line)) {
      // 去除行首尾空白字符
      StringTrim(&line);

      // 跳过空行和注释行（以#开头）
      if (line.empty() || line[0] == '#') {
        continue;
      }

      // 查找图像名称对应的ID并添加到待匹配列表
      if (image_name_to_image_id.count(line) == 0) {
        std::cerr << "ERROR: Image " << line << " does not exist." << std::endl;
      } else {
        image_ids.push_back(image_name_to_image_id.at(line));
      }
    }
  }

  // 将所有图像的特征索引到视觉索引中
  // 这一步将每张图像的SIFT特征量化为视觉词汇，建立倒排索引
  IndexImagesInVisualIndex(match_options_.num_threads,  // 并行线程数
                           options_.num_checks,          // 近似最近邻搜索的检查次数
                           options_.max_num_features,    // 每张图像使用的最大特征数
                           all_image_ids,                // 所有图像ID
                           this,                         // 当前匹配器实例
                           &cache_,                      // 特征缓存
                           &visual_index);               // 视觉索引

  // 检查是否被外部停止（如用户取消操作）
  if (IsStopped()) {
    GetTimer().PrintMinutes();
    return;
  }

  // 在视觉索引中进行最近邻匹配
  // 对于每张待匹配图像，检索其在词汇树中的最近邻图像，然后进行详细特征匹配
  MatchNearestNeighborsInVisualIndex(
      match_options_.num_threads,              // 并行线程数
      options_.num_images,                     // 每次批量处理的图像数
      options_.num_nearest_neighbors,          // 每张图像检索的最近邻数量
      options_.num_checks,                     // 近似最近邻搜索的检查次数
      options_.num_images_after_verification,  // 几何验证后保留的图像数
      options_.max_num_features,               // 每张图像使用的最大特征数
      image_ids,                               // 待匹配的图像ID列表
      this,                                    // 当前匹配器实例
      &cache_,                                 // 特征缓存
      &visual_index,                           // 视觉索引
      &matcher_);                              // 特征匹配器

  // 打印总耗时
  GetTimer().PrintMinutes();
}

SpatialFeatureMatcher::SpatialFeatureMatcher(
    const SpatialMatchingOptions& options,
    const SiftMatchingOptions& match_options, const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(5 * options_.max_num_neighbors, &database_),
      matcher_(match_options, &database_, &cache_) {
  CHECK(options_.Check());
  CHECK(match_options_.Check());
}

void SpatialFeatureMatcher::Run() {
  PrintHeading1("Spatial feature matching");

  if (!matcher_.Setup()) {
    return;
  }

  cache_.Setup();

  const std::vector<image_t> image_ids = cache_.GetImageIds();

  //////////////////////////////////////////////////////////////////////////////
  // Spatial indexing
  //////////////////////////////////////////////////////////////////////////////

  Timer timer;
  timer.Start();

  std::cout << "Indexing images..." << std::flush;

  GPSTransform gps_transform;

  size_t num_locations = 0;
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> location_matrix(
      image_ids.size(), 3);

  std::vector<size_t> location_idxs;
  location_idxs.reserve(image_ids.size());

  std::vector<Eigen::Vector3d> ells(1);

  for (size_t i = 0; i < image_ids.size(); ++i) {
    const auto image_id = image_ids[i];
    const auto& image = cache_.GetImage(image_id);

    if ((image.TvecPrior(0) == 0 && image.TvecPrior(1) == 0 &&
         options_.ignore_z) ||
        (image.TvecPrior(0) == 0 && image.TvecPrior(1) == 0 &&
         image.TvecPrior(2) == 0 && !options_.ignore_z)) {
      continue;
    }

    location_idxs.push_back(i);

    if (options_.is_gps) {
      ells[0](0) = image.TvecPrior(0);
      ells[0](1) = image.TvecPrior(1);
      ells[0](2) = options_.ignore_z ? 0 : image.TvecPrior(2);

      const auto xyzs = gps_transform.EllToXYZ(ells);

      location_matrix(num_locations, 0) = static_cast<float>(xyzs[0](0));
      location_matrix(num_locations, 1) = static_cast<float>(xyzs[0](1));
      location_matrix(num_locations, 2) = static_cast<float>(xyzs[0](2));
    } else {
      location_matrix(num_locations, 0) =
          static_cast<float>(image.TvecPrior(0));
      location_matrix(num_locations, 1) =
          static_cast<float>(image.TvecPrior(1));
      location_matrix(num_locations, 2) =
          static_cast<float>(options_.ignore_z ? 0 : image.TvecPrior(2));
    }

    num_locations += 1;
  }

  PrintElapsedTime(timer);

  if (num_locations == 0) {
    std::cout << " => No images with location data." << std::endl;
    GetTimer().PrintMinutes();
    return;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Building spatial index
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();

  std::cout << "Building search index..." << std::flush;

  flann::Matrix<float> locations(location_matrix.data(), num_locations,
                                 location_matrix.cols());

  flann::LinearIndexParams index_params;
  flann::LinearIndex<flann::L2<float>> search_index(index_params);
  search_index.buildIndex(locations);

  PrintElapsedTime(timer);

  //////////////////////////////////////////////////////////////////////////////
  // Searching spatial index
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();

  std::cout << "Searching for nearest neighbors..." << std::flush;

  const int knn = std::min<int>(options_.max_num_neighbors, num_locations);

  Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      index_matrix(num_locations, knn);
  flann::Matrix<size_t> indices(index_matrix.data(), num_locations, knn);

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      distance_matrix(num_locations, knn);
  flann::Matrix<float> distances(distance_matrix.data(), num_locations, knn);

  flann::SearchParams search_params(flann::FLANN_CHECKS_AUTOTUNED);
  if (match_options_.num_threads == ThreadPool::kMaxNumThreads) {
    search_params.cores = std::thread::hardware_concurrency();
  } else {
    search_params.cores = match_options_.num_threads;
  }
  if (search_params.cores <= 0) {
    search_params.cores = 1;
  }

  search_index.knnSearch(locations, indices, distances, knn, search_params);

  PrintElapsedTime(timer);

  //////////////////////////////////////////////////////////////////////////////
  // Matching
  //////////////////////////////////////////////////////////////////////////////

  const float max_distance =
      static_cast<float>(options_.max_distance * options_.max_distance);

  std::vector<std::pair<image_t, image_t>> image_pairs;
  image_pairs.reserve(knn);

  for (size_t i = 0; i < num_locations; ++i) {
    if (IsStopped()) {
      GetTimer().PrintMinutes();
      return;
    }

    timer.Restart();

    std::cout << StringPrintf("Matching image [%d/%d]", i + 1, num_locations)
              << std::flush;

    image_pairs.clear();

    for (int j = 0; j < knn; ++j) {
      // Check if query equals result.
      if (index_matrix(i, j) == i) {
        continue;
      }

      // Since the nearest neighbors are sorted by distance, we can break.
      if (distance_matrix(i, j) > max_distance) {
        break;
      }

      const size_t idx = location_idxs[i];
      const image_t image_id = image_ids.at(idx);
      const size_t nn_idx = location_idxs.at(index_matrix(i, j));
      const image_t nn_image_id = image_ids.at(nn_idx);
      image_pairs.emplace_back(image_id, nn_image_id);
    }

    DatabaseTransaction database_transaction(&database_);
    matcher_.Match(image_pairs);

    PrintElapsedTime(timer);
  }

  GetTimer().PrintMinutes();
}

TransitiveFeatureMatcher::TransitiveFeatureMatcher(
    const TransitiveMatchingOptions& options,
    const SiftMatchingOptions& match_options, const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(options_.batch_size, &database_),
      matcher_(match_options, &database_, &cache_) {
  CHECK(options_.Check());
  CHECK(match_options_.Check());
}

void TransitiveFeatureMatcher::Run() {
  PrintHeading1("Transitive feature matching");

  if (!matcher_.Setup()) {
    return;
  }

  cache_.Setup();

  const std::vector<image_t> image_ids = cache_.GetImageIds();

  std::vector<std::pair<image_t, image_t>> image_pairs;
  std::unordered_set<image_pair_t> image_pair_ids;

  for (int iteration = 0; iteration < options_.num_iterations; ++iteration) {
    if (IsStopped()) {
      GetTimer().PrintMinutes();
      return;
    }

    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Iteration [%d/%d]", iteration + 1,
                              options_.num_iterations)
              << std::endl;

    std::vector<std::pair<image_t, image_t>> existing_image_pairs;
    std::vector<int> existing_num_inliers;
    database_.ReadTwoViewGeometryNumInliers(&existing_image_pairs,
                                            &existing_num_inliers);

    CHECK_EQ(existing_image_pairs.size(), existing_num_inliers.size());

    std::unordered_map<image_t, std::vector<image_t>> adjacency;
    for (const auto& image_pair : existing_image_pairs) {
      adjacency[image_pair.first].push_back(image_pair.second);
      adjacency[image_pair.second].push_back(image_pair.first);
    }

    const size_t batch_size = static_cast<size_t>(options_.batch_size);

    size_t num_batches = 0;
    image_pairs.clear();
    image_pair_ids.clear();
    for (const auto& image : adjacency) {
      const auto image_id1 = image.first;
      for (const auto& image_id2 : image.second) {
        if (adjacency.count(image_id2) > 0) {
          for (const auto& image_id3 : adjacency.at(image_id2)) {
            const auto image_pair_id =
                Database::ImagePairToPairId(image_id1, image_id3);
            if (image_pair_ids.count(image_pair_id) == 0) {
              image_pairs.emplace_back(image_id1, image_id3);
              image_pair_ids.insert(image_pair_id);
              if (image_pairs.size() >= batch_size) {
                num_batches += 1;
                std::cout << StringPrintf("  Batch %d", num_batches)
                          << std::flush;
                DatabaseTransaction database_transaction(&database_);
                matcher_.Match(image_pairs);
                image_pairs.clear();
                PrintElapsedTime(timer);
                timer.Restart();

                if (IsStopped()) {
                  GetTimer().PrintMinutes();
                  return;
                }
              }
            }
          }
        }
      }
    }

    num_batches += 1;
    std::cout << StringPrintf("  Batch %d", num_batches) << std::flush;
    DatabaseTransaction database_transaction(&database_);
    matcher_.Match(image_pairs);
    PrintElapsedTime(timer);
  }

  GetTimer().PrintMinutes();
}

ImagePairsFeatureMatcher::ImagePairsFeatureMatcher(
    const ImagePairsMatchingOptions& options,
    const SiftMatchingOptions& match_options, const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(options.block_size, &database_),
      matcher_(match_options, &database_, &cache_) {
  CHECK(options_.Check());
  CHECK(match_options_.Check());
}

void ImagePairsFeatureMatcher::Run() {
  PrintHeading1("Custom feature matching");

  if (!matcher_.Setup()) {
    return;
  }

  cache_.Setup();

  //////////////////////////////////////////////////////////////////////////////
  // Reading image pairs list
  //////////////////////////////////////////////////////////////////////////////

  std::unordered_map<std::string, image_t> image_name_to_image_id;
  image_name_to_image_id.reserve(cache_.GetImageIds().size());
  for (const auto image_id : cache_.GetImageIds()) {
    const auto& image = cache_.GetImage(image_id);
    image_name_to_image_id.emplace(image.Name(), image_id);
  }

  std::ifstream file(options_.match_list_path);
  CHECK(file.is_open()) << options_.match_list_path;

  std::string line;
  std::vector<std::pair<image_t, image_t>> image_pairs;
  std::unordered_set<colmap::image_pair_t> image_pairs_set;
  while (std::getline(file, line)) {
    StringTrim(&line);

    if (line.empty() || line[0] == '#') {
      continue;
    }

    std::stringstream line_stream(line);

    std::string image_name1;
    std::string image_name2;

    std::getline(line_stream, image_name1, ' ');
    StringTrim(&image_name1);
    std::getline(line_stream, image_name2, ' ');
    StringTrim(&image_name2);

    if (image_name_to_image_id.count(image_name1) == 0) {
      std::cerr << "ERROR: Image " << image_name1 << " does not exist."
                << std::endl;
      continue;
    }
    if (image_name_to_image_id.count(image_name2) == 0) {
      std::cerr << "ERROR: Image " << image_name2 << " does not exist."
                << std::endl;
      continue;
    }

    const image_t image_id1 = image_name_to_image_id.at(image_name1);
    const image_t image_id2 = image_name_to_image_id.at(image_name2);
    const image_pair_t image_pair =
        Database::ImagePairToPairId(image_id1, image_id2);
    const bool image_pair_exists = image_pairs_set.insert(image_pair).second;
    if (image_pair_exists) {
      image_pairs.emplace_back(image_id1, image_id2);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Feature matching
  //////////////////////////////////////////////////////////////////////////////

  const size_t num_match_blocks = image_pairs.size() / options_.block_size + 1;
  std::vector<std::pair<image_t, image_t>> block_image_pairs;
  block_image_pairs.reserve(options_.block_size);

  for (size_t i = 0; i < image_pairs.size(); i += options_.block_size) {
    if (IsStopped()) {
      GetTimer().PrintMinutes();
      return;
    }

    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Matching block [%d/%d]",
                              i / options_.block_size + 1, num_match_blocks)
              << std::flush;

    const size_t block_end = i + options_.block_size <= image_pairs.size()
                                 ? i + options_.block_size
                                 : image_pairs.size();
    std::vector<std::pair<image_t, image_t>> block_image_pairs;
    block_image_pairs.reserve(options_.block_size);
    for (size_t j = i; j < block_end; ++j) {
      block_image_pairs.push_back(image_pairs[j]);
    }

    DatabaseTransaction database_transaction(&database_);
    matcher_.Match(block_image_pairs);

    PrintElapsedTime(timer);
  }

  GetTimer().PrintMinutes();
}

FeaturePairsFeatureMatcher::FeaturePairsFeatureMatcher(
    const FeaturePairsMatchingOptions& options,
    const SiftMatchingOptions& match_options, const std::string& database_path)
    : options_(options),
      match_options_(match_options),
      database_(database_path),
      cache_(kCacheSize, &database_) {
  CHECK(options_.Check());
  CHECK(match_options_.Check());
}

void FeaturePairsFeatureMatcher::Run() {
  PrintHeading1("Importing matches");

  cache_.Setup();

  std::unordered_map<std::string, const Image*> image_name_to_image;
  image_name_to_image.reserve(cache_.GetImageIds().size());
  for (const auto image_id : cache_.GetImageIds()) {
    const auto& image = cache_.GetImage(image_id);
    image_name_to_image.emplace(image.Name(), &image);
  }

  std::ifstream file(options_.match_list_path);
  CHECK(file.is_open()) << options_.match_list_path;

  std::string line;
  while (std::getline(file, line)) {
    if (IsStopped()) {
      GetTimer().PrintMinutes();
      return;
    }

    StringTrim(&line);
    if (line.empty()) {
      continue;
    }

    std::istringstream line_stream(line);

    std::string image_name1, image_name2;
    try {
      line_stream >> image_name1 >> image_name2;
    } catch (...) {
      std::cerr << "ERROR: Could not read image pair." << std::endl;
      break;
    }

    std::cout << StringPrintf("%s - %s", image_name1.c_str(),
                              image_name2.c_str())
              << std::endl;

    if (image_name_to_image.count(image_name1) == 0) {
      std::cout << StringPrintf("SKIP: Image %s not found in database.",
                                image_name1.c_str())
                << std::endl;
      break;
    }
    if (image_name_to_image.count(image_name2) == 0) {
      std::cout << StringPrintf("SKIP: Image %s not found in database.",
                                image_name2.c_str())
                << std::endl;
      break;
    }

    const Image& image1 = *image_name_to_image[image_name1];
    const Image& image2 = *image_name_to_image[image_name2];

    bool skip_pair = false;
    if (database_.ExistsInlierMatches(image1.ImageId(), image2.ImageId())) {
      std::cout << "SKIP: Matches for image pair already exist in database."
                << std::endl;
      skip_pair = true;
    }

    FeatureMatches matches;
    while (std::getline(file, line)) {
      StringTrim(&line);

      if (line.empty()) {
        break;
      }

      std::istringstream line_stream(line);

      FeatureMatch match;
      try {
        line_stream >> match.point2D_idx1 >> match.point2D_idx2;
      } catch (...) {
        std::cerr << "ERROR: Cannot read feature matches." << std::endl;
        break;
      }

      matches.push_back(match);
    }

    if (skip_pair) {
      continue;
    }

    const Camera& camera1 = cache_.GetCamera(image1.CameraId());
    const Camera& camera2 = cache_.GetCamera(image2.CameraId());

    if (options_.verify_matches) {
      database_.WriteMatches(image1.ImageId(), image2.ImageId(), matches);

      const auto keypoints1 = cache_.GetKeypoints(image1.ImageId());
      const auto keypoints2 = cache_.GetKeypoints(image2.ImageId());

      TwoViewGeometry two_view_geometry;
      TwoViewGeometry::Options two_view_geometry_options;
      two_view_geometry_options.min_num_inliers =
          static_cast<size_t>(match_options_.min_num_inliers);
      two_view_geometry_options.ransac_options.max_error =
          match_options_.max_error;
      two_view_geometry_options.ransac_options.confidence =
          match_options_.confidence;
      two_view_geometry_options.ransac_options.min_num_trials =
          static_cast<size_t>(match_options_.min_num_trials);
      two_view_geometry_options.ransac_options.max_num_trials =
          static_cast<size_t>(match_options_.max_num_trials);
      two_view_geometry_options.ransac_options.min_inlier_ratio =
          match_options_.min_inlier_ratio;

      two_view_geometry.Estimate(
          camera1, FeatureKeypointsToPointsVector(*keypoints1), camera2,
          FeatureKeypointsToPointsVector(*keypoints2), matches,
          two_view_geometry_options);

      database_.WriteTwoViewGeometry(image1.ImageId(), image2.ImageId(),
                                     two_view_geometry);
    } else {
      TwoViewGeometry two_view_geometry;

      if (camera1.HasPriorFocalLength() && camera2.HasPriorFocalLength()) {
        two_view_geometry.config = TwoViewGeometry::CALIBRATED;
      } else {
        two_view_geometry.config = TwoViewGeometry::UNCALIBRATED;
      }

      two_view_geometry.inlier_matches = matches;

      database_.WriteTwoViewGeometry(image1.ImageId(), image2.ImageId(),
                                     two_view_geometry);
    }
  }

  GetTimer().PrintMinutes();
}

}  // namespace colmap
