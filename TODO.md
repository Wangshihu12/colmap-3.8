# Loop Edge Debug & Tuning TODO

## 目标
- 为 `BuildLoopEdgesSim3` 增加可解释、可追踪、可调参的调试能力。
- 支持精确定位“为什么某个候选回环边被拒绝/被接受”。
- 支持基于数据自动调整回环边权重，而不是仅依赖固定权重。

## 范围
- 主要改动文件：`test/test_vpgo_pipeline.cc`
- 可选联动：
  - `src/optim/bundle_adjustment.cc`（若需输出优化前后残差）
  - `scripts/shell/run_pgo.sh`（增加调试开关）

## 里程碑
1. M1: Pair 级审计日志（CSV）可用
2. M2: 回环边影像可视化可用
3. M3: 权重动态策略可用
4. M4: PGO 前后残差统计可用
5. M5: 调参与验收脚本稳定可复现

---

## M1 - Pair 审计日志（最高优先级）

### 1. 新增调试选项
- 在 `PipelineOptions` 新增开关与输出路径：
  - `bool enable_loop_debug = false`
  - `std::string loop_debug_dir`
  - `size_t loop_debug_max_images = 200`（可选）

### 2. 定义拒绝原因枚举
- 新增 `enum class LoopRejectReason`，至少覆盖：
  - `kImageMissing`
  - `kOccupiedPair`
  - `kTimestampMissing`
  - `kTimeGapTooSmall`
  - `kInvalidConfig`
  - `kTooFewInliers`
  - `kNoMatches`
  - `kLowInlierRatio`
  - `kSmallTriAngle`
  - `kLoopDegreeExceeded`
  - `kDistanceTooLarge`
  - `kDirEstimationFailed`
  - `kSim3EstimationFailed`
  - `kNotAddedAny`

### 3. 定义审计记录结构
- 新增 `struct LoopEdgeAuditRecord` 字段：
  - 基础：`pair_id,image_id1,image_id2`
  - 几何：`geom_config,num_inliers,num_matches,inlier_ratio,tri_angle,dist,time_gap`
  - 估计状态：`has_dir,has_sim3,loop_scale`
  - 权重：`rot_weight,trans_weight,dir_rot_weight,dir_trans_weight`
  - 结果：`selected,reject_reason`
  - 可选：`risk_score`

### 4. 输出 CSV
- 输出文件：
  - `loop_edge_audit_all.csv`（所有候选）
  - `loop_edge_audit_selected.csv`（入选）
  - `loop_edge_audit_rejected.csv`（拒绝）
- 要求：
  - 固定列顺序，便于后续脚本解析
  - 所有浮点保留固定精度（例如 `std::setprecision(6)`）

### 5. 验收标准
- 开启调试后，每个候选 pair 都有审计记录。
- 每个拒绝 pair 都有明确 `reject_reason`。
- 关闭调试开关时行为与当前版本一致（不额外输出文件）。

---

## M2 - 回环边影像可视化

### 1. 导出关键边可视化图
- 对入选边导出拼接图：
  - 左右影像拼接
  - 内点匹配连线（可抽样，避免图过密）
  - 叠加文字：`pair_id/inlier_ratio/tri_angle/dist/scale/type`

### 2. 导出高风险边
- 定义高风险条件（任一满足）：
  - `inlier_ratio` 低于阈值附近
  - `loop_scale` 偏离 1 过大
  - `num_inliers` 接近最小阈值
- 输出 `loop_edge_suspicious_topk.csv` + 对应图片目录。

### 3. 验收标准
- 能在调试目录快速定位每条边对应图片与统计信息。
- 人工抽检可复现“这条边为什么被选/被拒”。

---

## M3 - 回环权重动态策略

### 1. 引入边置信度分数
- 新增 `ComputeLoopEdgeConfidence(...) -> double`，输入：
  - `num_inliers, inlier_ratio, tri_angle, sim3_residual, scale_deviation`
- 输出范围 `[0, 1]`。

### 2. 权重从固定值改为“基准 * 置信度”
- 公式建议：
  - `rot_weight = base_rot_weight * conf`
  - `trans_weight = base_trans_weight * conf`
- 对 `dir-only` 边增加上限：
  - `dir_trans_weight <= dir_trans_weight_cap`
  - `dir_rot_weight = 0`

### 3. 加保护逻辑
- 若 `conf < min_confidence_for_edge` 则拒绝该边。
- 若 `scale` 异常（如过小/过大）降权或拒绝。

### 4. 验收标准
- 审计 CSV 中能看到每条边的 `confidence` 与最终权重。
- 高风险边在权重上明显被抑制。

---

## M4 - PGO 前后残差与稳定性报告

### 1. 输出边级残差报告
- 文件：`loop_edge_residual_before_after.csv`
- 字段建议：
  - `pair_id,image_id1,image_id2,edge_type`
  - `rot_res_before,rot_res_after`
  - `trans_res_before,trans_res_after`
  - `delta_rot,delta_trans`

### 2. 输出汇总统计
- 文件：`loop_edge_residual_summary.txt`
- 内容建议：
  - 均值/中位数/P90/P95
  - 边类型分组统计（dir-only vs sim3）
  - 高残差边比例

### 3. 验收标准
- 能快速判断“加边后是否真的优化了几何一致性”。
- 可支持对比不同参数组（A/B）。

---

## M5 - 调参与实验流程固化

### 1. 参数扫描计划
- 固定前端匹配，分两阶段扫描：
  - 阈值：`min_loop_inliers`, `min_loop_inlier_ratio`, `min_loop_time_gap`, `dist gate`
  - 权重：`loop_rot_weight`, `loop_trans_weight`, `dir_trans_weight_cap`

### 2. 每组实验统一输出
- 必须保留：
  - `loop_edge_audit_*.csv`
  - `loop_edge_residual_before_after.csv`
  - 最终重建指标（均值重投影误差、连通性统计）

### 3. 结果排序指标（建议）
- 主指标：
  - 跨层连通性提升（跨层 shared 3D points）
  - PGO 后残差下降幅度
- 约束指标：
  - 最终重投影误差不过度恶化
  - 高风险边占比可控

---

## 补充优化建议（待选）

### A. 放宽或参数化硬编码距离门限
- 当前 `dist > 10.0` 可考虑改为配置项并纳入审计统计。

### B. `occupied_pairs` 策略优化
- 避免“仅方向边先占位”导致后续高质量 Sim3 边无法加入。
- 可考虑：
  - 先缓存候选，再按质量统一选边
  - 或按边类型分别设置 quota

### C. 日志性能控制
- 调试模式下开启详细输出，默认关闭。
- 图片输出限制 top-k，防止 IO 过大拖慢流程。

---

## 实施顺序（建议）
1. M1 审计日志（先打通可解释性）
2. M4 残差报告（先建立效果评估闭环）
3. M3 动态权重（在可评估前提下调策略）
4. M2 影像可视化（用于人工快速定位问题边）
5. M5 参数扫描与固化

---

## 交付清单（完成标准）
- [ ] `TODO` 中 M1-M5 全部实现
- [ ] 调试开关关闭时，无额外性能损耗和输出
- [ ] 提供至少一组 A/B 对比结果，证明策略有效
- [ ] 文档化每个输出文件格式与字段说明

