# mmdetection到mmrotate 迁移 - 文件索引与导航

**迁移完成日期**: 2026-03-25  
**迁移状态**: ✅ 第一阶段完成 (90%)

---

## 🗺️ 文件导航地图

### 📖 文档文件 (按阅读顺序)

1. **QUICK_REFERENCE.md** ⭐ **推荐首先阅读**
   - 快速概览: 5分钟了解全貌
   - 模型列表、文件位置、关键改动
   - 故障排除快速指南
   - 位置: `/projects/u6ae/Yong/u5db/mmrotate/QUICK_REFERENCE.md`

2. **MIGRATION_SUMMARY.txt**
   - 工作摘要: 10分钟深入了解
   - 完成情况、统计数据、后续计划
   - 文档质量检查清单
   - 位置: `/projects/u6ae/Yong/u5db/mmrotate/MIGRATION_SUMMARY.txt`

3. **MIGRATION_CHANGELOG.md** (详细版本) 🔍
   - 完整参考: 1小时详细学习
   - 每个模型的详细改动
   - 框架差异分析
   - 验证清单和注意事项
   - 位置: `/projects/u6ae/Yong/u5db/mmrotate/MIGRATION_CHANGELOG.md`

### 💾 代码文件

#### Backbone 模块 (模型特征提取)
```
mmrotate/models/backbones/
├── msfa.py                          # MSFA backbone (16K)
│   └── 特点: SAR + Wavelet + HOG + Canny处理
│
├── convnext_moe.py                  # ConvNeXt with MoE (39K)
│   └── 特点: Mixture of Experts动态网络
│
└── hivit.py                         # HiViT (12K)
    └── 特点: Hierarchical Vision Transformer
```

#### Neck 模块 (特征融合)
```
mmrotate/models/necks/
└── frequency_spatial_fpn.py         # 频域-空间FPN (27K)
    └── 特点: 结合频域和空间特征处理
```

#### Head 模块 (目标检测)
```
mmrotate/models/dense_heads/
└── gfl_head.py                      # GFL检测头 (29K)
    └── 特点: Generalized Focal Loss
```

#### 注册表文件 (已修改)
```
mmrotate/models/
├── backbones/__init__.py            # [修改] 新增导入注释
└── necks/__init__.py                # [修改] 新增导入注释
```

### 🔧 配置文件 (模型配置)

```
mmrotate/configs/sardet/
├── 01_msfa_r50_frcnn_dota.py        # Model 1: MSFA + Faster-RCNN
│   └── 基于: fg_frcnn_dota_pretrain_sar_wavelet_r50.py
│
├── 02_hivit_frcnn_dota.py           # Model 2: HiViT + Faster-RCNN  
│   └── 基于: hivit_base_SARDet.py
│
├── 03_convnext_moe_gfl_dota.py      # Model 3: ConvNeXt-MoE + GFL
│   └── 基于: SM3Det.py
│
└── 04_r50_freqfpn_gfl_dota.py       # Model 4: ResNet50 + FreqFPN + GFL
    └── 基于: gfl_r50_denodet_sardet.py
```

---

## 📊 详细内容导航

### 快速问答

**Q: 迁移了哪些模型？**  
A: 4个SARDet模型。详见 QUICK_REFERENCE.md 第一部分

**Q: 源文件在哪里找？**  
A: 所有源文件信息在MIGRATION_CHANGELOG.md中的"迁移的代码文件"部分

**Q: 如何激活这些模型？**  
A: 见MIGRATION_SUMMARY.txt的"短期任务"部分

**Q: 有什么关键改动？**  
A: 详见QUICK_REFERENCE.md的"关键改动要点"部分

### 按任务搜索

| 任务 | 查看文件 | 位置 |
|-----|--------|------|
| 快速了解迁移内容 | QUICK_REFERENCE.md | L1-50 |
| 查看4个模型详情 | QUICK_REFERENCE.md | L60-80 |
| 理解框架转变 | MIGRATION_CHANGELOG.md | "关键的框架差异和适配" |
| 了解文件位置 | QUICK_REFERENCE.md | L85-120 |
| 验证文件完整性 | MIGRATION_SUMMARY.txt | L1-80 |
| 查看待处理项目 | MIGRATION_CHANGELOG.md | "待处理的任务" |
| 学习改动标注规则 | QUICK_REFERENCE.md | L140-160 |

---

## 🎯 使用场景导读

### 场景 1: "我想快速了解迁移内容"
👉 **阅读顺序**: QUICK_REFERENCE.md → 配置文件示例  
⏱️ **耗时**: 10-15分钟

**关键步骤**:
1. 打开 QUICK_REFERENCE.md
2. 阅读"迁移完成概要"部分
3. 浏览"文件位置速查表"
4. 查看一个配置文件示例 (如 01_msfa_r50_frcnn_dota.py)

### 场景 2: "我需要激活和测试这些模型"
👉 **阅读顺序**: MIGRATION_CHANGELOG.md → MIGRATION_SUMMARY.txt  
⏱️ **耗时**: 30-45分钟

**关键步骤**:
1. 查看"验证清单"部分
2. 按照"待处理的任务"逐步操作
3. 参考"故障排除快速指南"
4. 查看配置文件中的[待处理]标记

### 场景 3: "我想理解框架转变细节"
👉 **阅读顺序**: MIGRATION_CHANGELOG.md (详细版) → 配置代码对比  
⏱️ **耗时**: 1-2小时

**关键步骤**:
1. 阅读"关键的框架差异和适配"部分
2. 对比配置文件的[原有]和[新增]代码块
3. 查看"Anchor配置变化"和"BBox编码维度"部分
4. 研究具体的配置文件实现

### 场景 4: "我碰到了问题"
👉 **查看**: QUICK_REFERENCE.md 的"故障排除"部分  
⏱️ **耗时**: 5-10分钟

**常见问题映射**:
- 找不到模块 → L165-170
- 数据维度错误 → L172-176  
- 数据加载失败 → L178-182
- 权重不兼容 → L184-188

---

## 📋 核心改动要点 (一页纸总结)

### 框架转变

| 项目 | mmdetection | mmrotate | 为什么改 |
|-----|-----------|---------|--------|
| BBox格式 | 4D (x,y,w,h) | 5D (x,y,w,h,θ) | 支持旋转 |
| Detector | FasterRCNN | RotatedFasterRCNN | 旋转检测 |
| Detector | GFL | RotatedGFL | 旋转检测 |
| Anchor生成 | AnchorGenerator | RotatedAnchorGenerator | 旋转角度 |
| NMS | 标准NMS | 旋转不变NMS | 旋转检测 |

### 代码风格

- ✓ 原有代码: **保留为注释**
- ✓ 改动标记: **[迁移] [改动] [新增] [待处理]**
- ✓ 易追溯: **完整的改动历史**

### 关键文件

| 类型 | 数量 | 行数 |
|-----|------|------|
| 复制的代码文件 | 5 | ~3,500 |
| __init__.py修改 | 2 | ~10 |
| 配置文件 | 4 | ~400 |
| 文档文件 | 3 | ~1,500 |

---

## ✅ 验证清单

使用本文档验证迁移完整性:

- [ ] 5个代码文件已在正确位置
- [ ] 2个__init__.py已修改
- [ ] 4个配置文件已创建
- [ ] 3个文档文件已生成
- [ ] 配置文件注释正确
- [ ] 标记清晰可读
- [ ] 没有误删代码
- [ ] 导入正确注释

---

## 🚀 后续建议

### 立即可做 (今天)
✓ 阅读 QUICK_REFERENCE.md  
✓ 浏览配置文件示例  
✓ 理解框架转变

### 为期1周
- [ ] 解注释导入，进行模型加载测试
- [ ] 验证预训练权重兼容性
- [ ] 准备数据集转换脚本

### 为期2-3周
- [ ] 数据集格式转换完成
- [ ] 单GPU训练测试
- [ ] 基准性能评估

---

## 📞 参考资源

| 资源 | 说明 |
|-----|------|
| QUICK_REFERENCE.md | 快速参考 (5分钟) |
| MIGRATION_SUMMARY.txt | 工作总结 (10分钟) |
| MIGRATION_CHANGELOG.md | 详细文档 (1小时) |
| 配置文件 | 实现细节 |
| mmrotate docs | 框架参考 |

---

**文档创建**: 2026-03-25  
**版本**: 1.0  
**状态**: 完成 ✅

---

*提示: 建议先阅读 QUICK_REFERENCE.md，然后根据需要查看相应部分*
