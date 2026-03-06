下面给你一套“先锁定不敏感参数 + 只对关键参数做benchmark调参”的建议，并同时回答你文档里提到的三个关键决策点。整体原则是：把“会显著改变最优点/结论”的参数留下来调，把“主要影响训练效率/可替代”的参数锁死，避免把网格扩大到不可控。 ￼

⸻

1) 你提的三个关键问题：建议怎么定

Q1. CPU / GPU 是否应分别锁定默认参数？

建议：分轨（CPU轨、GPU轨）分别锁定默认参数，但用同一“模型/损失定义与数据划分”。原因是 CPU/GPU 的最优点常被 batch size、吞吐、数值精度、线程 等因素牵引，强行共享一套默认值会导致：CPU轨被迫选小batch/小模型，GPU轨则被迫选不经济配置，最终两条曲线都不“代表性”。（工程上也更容易做公平对照：CPU-only vs CUDA-only 两套report。）

Q2. num_threads 固定单值还是随机器核数比例？

建议：主benchmark固定单值（例如 8 或 16），并把“max/按核数比例”放到补充材料做 scaling 曲线。
原因：如果主表允许按核数比例跑，你的 runtime 结果会不可比（不同机器/容器/BLAS后端差异巨大）。你文档里已计划记录 OMP/MKL/OPENBLAS/NUMEXPR 环境变量，这是正确方向；主表仍建议固定线程数以减少自由度。 ￼

Q3. 主表只放核心参数锁定结果，增强参数放补充材料？

**建议：是的。**主表只呈现“无争议的核心机制 + 公平可复现设置下的结果”，增强项（domain/topology/calibration 等）作为 ablation / optional knobs 放补充材料更稳妥。理由：domain/topology 权重通常是“数据集依赖”的（domain shift 结构不同，最优权重不同），把它们塞进主benchmark容易被质疑“你是在为某个数据集调参”。DANN类方法本身也强调训练过程/权重策略会影响效果与稳定性。 ￼

⸻

2) 哪些参数“影响不大”可以直接固定（结合文献/经验）

这里按“对最终结论敏感度”分层。你的目标是 reference mapping / label transfer + 置信度，这和很多单细胞DL/预训练工作里的经验一致：学习率、模型容量、正则化强度往往决定主性能；其余多是效率/稳定性层面的二阶因素。 ￼

建议固定为默认值（不进主调参）的参数

(A) val_fraction：固定 0.1
0.2 通常只是在你担心过拟合/数据极少时才需要；对大规模细胞数来说，0.1 已足够稳定估计早停信号。更重要的是：val_fraction 会改变有效训练数据量，影响可比性。

(B) num_epochs：固定一个“上限”，靠 early stopping 决定实际轮数
建议：max_epochs = 50（或40也行），不要把 epochs 当作主调参维度。你已经有 early stopping 机制，epochs 网格往往只是在改变“是否早停能触发”，属于低价值自由度。很多单细胞方法报告里也是固定epoch区间/上限再训练。比如一个多对抗域适配的单细胞分配方法设置 encoder training 20 epochs、其它阶段 50 epochs，并未把epoch作为主要搜索维度。 ￼

(C) early_stopping_patience：固定 5（或 10），不做网格
patience=3 往往过激，容易在噪声/类不平衡下提前停止；5 或 10 更稳。建议默认 5，若你发现loss曲线抖动很大再用10。

(D) dropout_rate：固定 0.1（默认），只在你看到明显过拟合时再开一档 0.2
大量深度学习/预训练工作把 dropout 0.1 当常用默认（例如某些大规模单细胞预训练设置 dropout=0.1、LR=1e-4）。 ￼
对你这种“共享编码器 + 分类头”的监督任务，dropout 的边际收益通常不如“合适的容量 + LR”。所以把 dropout 从主调参里拿掉，可以显著缩小搜索空间。

(E) calibration_max_iter / calibration_lr：固定，不要调
你在文档里考虑 temperature scaling。这个方法本质是单参数（温度 T）优化，通常不需要把优化器超参当成研究点；把 max_iter=100~200、lr=0.05 固定即可。温度缩放被广泛认为是“简单但有效”的后处理校准方案。 ￼

(F) reference_storage：固定 external（主默认），full 只做一次资源开销对照
这属于工程/内存策略，通常不该作为“性能调参”变量，只要保证两者输出一致（或误差可解释）即可。 ￼

⸻

3) 哪些参数必须保留做benchmark调参（关键参数）

必须保留（主benchmark的核心调参维度）

(1) learning_rate（最关键）
建议保留 3点：1e-3, 3e-4, 1e-4（你原先的 5e-4 可被 3e-4 替代，覆盖更均匀）。LR 往往比dropout/epoch更决定最终点。

(2) hidden_sizes / model width（第二关键）
建议保留 2–3 档容量（尤其要考虑 CPU 轨的可承受性）：
	•	CPU轨：[256,128]、[512,256]
	•	GPU轨：再加一档 [1024,512]（如果显存允许）
容量会显著改变 fine-grained label 的上限与泛化/校准。

(3) batch_size（第三关键，但要“与设备绑定”）
batch size 同时影响收敛、泛化与吞吐；并且与设备强耦合。泛化层面 batch size 的影响在多领域都有讨论（尤其医学数据/小样本时）。 ￼
建议你把 batch size 作为“每条轨道的少量离散点”：
	•	CPU轨：128, 256（512在CPU上经常是内存/吞吐不经济）
	•	GPU轨：256, 512

建议保留但“只做一次对照/少量点”（不扩展网格）

(4) input_transform：binary vs float
你文档建议默认 binary，并至少做一次对照。这个决策本质上是“建模假设”而不是超参；我同意：做一次严格对照后就锁定默认，不要带入多维网格。 ￼

⸻

4) 增强参数（domain/topology/calibration）：如何处理才不被质疑“过度调参”

主建议：主表默认全关（权重=0），补充材料做 ablation
	•	domain_loss_weight: 主表 0；补充 0.05/0.1
	•	topology_loss_weight: 主表 0；补充 0.01/0.02
	•	topology_k: 固定 20（或10），不要再扩展
原因：DANN类/拓扑约束类方法对数据集差异非常敏感，且训练稳定性依赖权重策略；作为 optional knob + ablation 更符合预期。 ￼

calibration（temperature scaling）：主表建议默认“开”

如果你把“置信度/不确定性”作为核心卖点之一，那么：
	•	主表建议报告 未校准 vs temperature scaling 两列（或两条曲线），但不把它当调参；
	•	温度缩放是后处理，通常不会改变 top-1 label，但显著影响 ECE / reliability；这与 Guo et al. 的结论一致。 ￼

⸻

5) 把搜索空间从“上千组合”压到“可控的几十个run”：一个可执行的调参方案

Step 0：固定所有“非关键参数”

建议默认（可直接写入manifest）：
	•	dropout=0.1
	•	val_fraction=0.1
	•	max_epochs=50（或40）
	•	early_stopping_patience=5
	•	reference_storage=external
	•	calibration = temperature_scaling（并固定 max_iter/lr）
	•	CPU轨固定 num_threads=8（或16），并记录线程环境变量 ￼

Step 1：核心三参小网格（每条轨道 12–18 次以内）

每条轨道只调：learning_rate × hidden_sizes × batch_size
例如：
	•	LR：3个
	•	hidden：2个（CPU轨）/3个（GPU轨）
	•	batch：2个

CPU轨：3×2×2 = 12
GPU轨：3×3×2 = 18

这一步只看“核心指标”（你后续定义的 accuracy / macro-F1 / unknown处理 + calibration指标）。

Step 2：input_transform 一次对照（只在最佳核心配置上）

binary vs float：2次

Step 3：增强项 ablation（只在最佳配置上）
	•	domain_loss_weight：0 / 0.05 / 0.1（3次）
	•	topology_loss_weight：0 / 0.01 / 0.02（3次）
	•	topology_k 固定 20（或10）
合计 6 次（也可只挑一个最典型任务/数据集做）

这样总体控制在：CPU 12 + GPU 18 + 对照/ablation 10–15 ≈ 40–50 次 run，可写进论文级benchmark流程。

⸻

6) 你可以直接“锁定下来”的参数清单（建议版）

锁定（默认，不进调参）
	•	dropout_rate = 0.1（必要时仅加测0.2）
	•	val_fraction = 0.1
	•	max_epochs = 50（配合早停）
	•	early_stopping_patience = 5
	•	reference_storage = external
	•	calibration_method = temperature_scaling（max_iter=100/200固定；lr=0.05固定） ￼
	•	topology_k = 20（若做拓扑loss）

只调关键三参（主benchmark）
	•	learning_rate ∈ {1e-3, 3e-4, 1e-4}
	•	hidden_sizes ∈ {CPU:[256,128],[512,256]; GPU:+[1024,512]}
	•	batch_size ∈ {CPU:128,256; GPU:256,512}

只做一次对照（决策型，而非调参）
	•	input_transform ∈ {binary, float}（在最佳配置上对照一次）

只做补充材料 ablation
	•	domain_loss_weight（DANN-style） ￼
	•	topology_loss_weight


