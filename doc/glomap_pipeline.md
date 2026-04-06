# GLOMAP Pipeline Port for COLMAP 3.8

本文档说明当前仓库中移植的 `GLOMAP` 风格全局 SfM 管线，以及它与最新
`COLMAP` / 原始 `glomap` 项目的关系。

## 目标

本次移植的目标是：

- 在当前 `COLMAP 3.8` 代码结构中增加一个可编译、可调用、可测试的全局
  SfM 管线。
- 将核心流程落到 [`src/glomap/global_mapper.h`](/home/wang/3dReconstruction/colmap-3.8/src/glomap/global_mapper.h)
  和 [`src/glomap/global_mapper.cc`](/home/wang/3dReconstruction/colmap-3.8/src/glomap/global_mapper.cc)。
- 在 [`test/test_glomap.cc`](/home/wang/3dReconstruction/colmap-3.8/test/test_glomap.cc)
  中提供外部调用入口，并包含一个合成数据验证流程。

## 当前实现范围

当前实现移植了 GLOMAP 的核心思想，并对 `COLMAP 3.8` 的旧数据结构做了兼容改写。
相较于上一版单文件实现，当前代码已经补齐了接近新版 COLMAP 的结构层：

- 新增 [`pose_graph.h`](/home/wang/3dReconstruction/colmap-3.8/src/glomap/pose_graph.h)
  和 [`pose_graph.cc`](/home/wang/3dReconstruction/colmap-3.8/src/glomap/pose_graph.cc)
  负责位姿图加载、有效边过滤和连通分量管理
- 新增 [`observation_manager.h`](/home/wang/3dReconstruction/colmap-3.8/src/glomap/observation_manager.h)
  和 [`observation_manager.cc`](/home/wang/3dReconstruction/colmap-3.8/src/glomap/observation_manager.cc)
  负责三维点过滤与兼容版观测管理
- 新增 [`global_pipeline.h`](/home/wang/3dReconstruction/colmap-3.8/src/glomap/global_pipeline.h)
  和 [`global_pipeline.cc`](/home/wang/3dReconstruction/colmap-3.8/src/glomap/global_pipeline.cc)
  作为顶层控制器，接口风格靠近新版 `GlobalPipeline`
- `global_mapper` 改成分阶段流程：
  `rotation averaging -> track establishment -> global positioning -> iterative BA -> iterative retriangulation/refine`
- 基于 `two_view_geometries` 构建全局位姿图
- 支持 `view-graph calibration` 的焦距校准与异常边过滤
- 支持 `PoseLib` 相对位姿重估
- 通过最大生成树初始化全局旋转
- 使用 Ceres 做旋转平均
- 使用基于相对平移方向的平移平均
- 基于匹配连通分量建立 tracks
- 对 tracks 做多视图三角化
- 使用现有 COLMAP BA 做两阶段全局优化
- 使用现有 `IncrementalMapper / IncrementalTriangulator` 做兼容版重三角化与 refine

## 与上游最新实现的差异

由于当前仓库是 `COLMAP 3.8`，而最新 `COLMAP` 已经把 GLOMAP 集成到了更新的
`scene / frame / rig / pose_graph` 架构中，因此这里没有直接逐文件原样搬运，而是
做了兼容性移植。

当前版本有以下限制：

- 当前实现主要针对“单相机、普通图像集”的典型场景
- 未完整覆盖 rig / gravity / calibration refinement 等高级能力
- 由于 `COLMAP 3.8` 的 BA 和 scene abstraction 仍旧是旧接口，
  当前实现虽然补齐了 `PoseGraph / ObservationManager / GlobalPipeline`
  职责层，但内部仍需要通过兼容层映射到旧版 `Reconstruction /
  DatabaseCache / IncrementalMapper`
- `PoseLib` 已接入，但在个别数据上如果重估结果导致后续三角化失败，
  当前实现会自动回退到数据库中的原始相对位姿

换句话说，这一版更准确地说是：

`GLOMAP 核心流程在 COLMAP 3.8 上的兼容实现`

而不是“最新 COLMAP 全量目录结构的逐文件镜像”。

## 调用方式

编译后可以通过 [`test/test_glomap.cc`](/home/wang/3dReconstruction/colmap-3.8/test/test_glomap.cc)
生成的可执行文件调用顶层 `GlomapPipeline`：

```bash
./build/bin/test_glomap --database_path path/to/database.db --output_path path/to/output_model
```

如果不传参数，它会自动运行一套合成数据测试：

```bash
./build/bin/test_glomap --synthetic_test
```

## 输出

当前测试入口会将结果写成 COLMAP 文本模型：

- `cameras.txt`
- `images.txt`
- `points3D.txt`

## 测试说明

[`test/test_glomap.cc`](/home/wang/3dReconstruction/colmap-3.8/test/test_glomap.cc)
中的 synthetic smoke test 默认关闭了 retriangulation：

- 原因不是功能缺失，而是三视图最小样例过小，进入兼容版 retriangulation 后
  容易因为旧版 `COLMAP 3.8` 的统计与 gauge 行为而退化
- 因此这个测试主要验证“新版风格主链路是否正确接通”
- 完整 retriangulation/refine 实现仍然保留在
  [`global_mapper.cc`](/home/wang/3dReconstruction/colmap-3.8/src/glomap/global_mapper.cc)
  中，适合在真实数据上继续迭代验证

## 后续可扩展方向

如果后续要继续向最新 `COLMAP` 对齐，建议按下面顺序推进：

1. 对齐最新 `COLMAP` 的 `GlobalPipeline` / `PoseGraph` / `ObservationManager`
2. 将当前 `src/glomap` 继续拆分成 `estimators / processors / scene`
3. 补齐 gravity、rig、多相机和 generalized relative pose 支持
4. 将入口正式集成到 `exe/colmap.cc`
