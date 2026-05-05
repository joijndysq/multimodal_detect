train15 训练结果分析（51/100 epochs）
实验配置
项目	值
模型	VIF-YOLO (EMA + LMF)
数据集	LLVIP (person + fire, nc=2)
Epochs	100（已跑51）
Batch	16, imgsz=640, device=0
优化器	SGD, lr0=0.01, lr0=0.01, momentum=0.937
---
核心指标趋势
Epoch	Precision
1	38.6%
5	67.9%
10	67.1%
20	83.3%
30	82.4%
40	83.6%
51	85.1%
历史最佳
- Best mAP50: Epoch 19 → *85.02%*（此后未超越）
- Best mAP50-95: Epoch 41 → 45.21%
- Best Precision: Epoch 19 → 85.72%
---
关键发现
1. 前10个epoch收敛极快
- mAP50 从 31% → 71%，cls_loss 从 3.12 → 1.14
- 说明 EMA+LMF 特征融合有效，模型快速学到了多模态判别特征
2. 30epoch后进入明显平台期
- mAP50 近10个epoch波动仅 ±1.4%（81.9%-84.7%）
- mAP50-95 近10个epoch波动仅 ±0.9%（42.8%-44.6%）
- train/cls_loss 后5个epoch: 0.821→0.817→0.815→0.815→0.805 — 下降速度显著放缓
- val/cls_loss 后5个epoch: 1.017→1.024→0.987→0.972→0.967 — 仍在缓慢下降
3. Precision > Recall 的差距
- Precision 85.1% vs Recall 77.7%，差 7.4 个百分点
- 说明漏检（false negative）多于误检（false positive）
- 可能原因：LLVIP中小目标（远距离person）难以被检测到
4. mAP50 vs mAP50-95 差距大
- mAP50=84.7%，但 mAP50-95=44.6%，差距 40 个百分点
- 说明定位精度不够好 — 模型能找到目标，但 bounding box 不够精确
- 这在红外/可见光融合检测中很常见，两模态的空间偏差会影响定位
5. Val loss 仍缓慢下降
- val/cls_loss 从 epoch 30 的 1.035 → epoch 51 的 0.967，仍有改善空间
- 剩余49个epoch大概率还能涨，但幅度不会大
---
建议
1. 继续训练完100epoch — val loss 尚未饱和，预计 mAP50 可达 85-86%
2. 考虑数据增强 — 当前 mosaic=1.0，但 mixup=0.0, flipud=0.0。开启 mixup 可改善泛化
3. 后续优化方向：
   - 定位精度差 → 调整 box_loss 权重（当前7.5，可适当增加）
   - Recall偏低 → 降低 conf threshold 或增加小目标检测能力



train26 评估结果（VIF-YOLO-S，30 epochs，预训练，batch=16）
全模型对比
指标	train25 (n, 250E)
模型	VIF-YOLO n
参数量	2.53M
batch	25
mAP50	88.66%
mAP50-95	49.57%
训练时间/epoch	~2.8min
关键发现
1. S 模型收敛速度快：仅 30 epochs 就逼近 n 模型 250 epochs 的 mAP50-95（50.16% vs 49.57%），预训练权重的优势明显。
2. 曲线尚未收敛：E30 时 mAP50 已从峰值 88.42%（E22）微降至 85.57%，mAP50-95 在 50.16%（E23）后也在下滑。说明 30 epochs 还不够充分。
3. 收益边际：S 模型参数量翻 4.4 倍，但 mAP50-95 仅提升 0.6%。这验证了 LMF 融合架构在 n 尺度下已经非常高效。
建议
当前最佳：train25（VIF-YOLO-n，mAP50-95 = 49.57%）
理由：S 模型虽然 peak mAP50-95 达到 50.16%，但：
- 参数量 11M vs 2.5M，推理速度会慢 2-3 倍
- 提升仅 0.6%，性价比不高
- E30 已在下滑，需要更多 epochs 才能稳定超越
如果追求极致精度，可以从 train26/weights/best.pt（E23）继续训练到 100 epochs，看能否稳定突破 51%。
