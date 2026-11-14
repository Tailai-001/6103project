# 英文报告（E）

后续放 LaTeX 源文件、参考文献与最终 PDF。原始项目说明要求 AAAI 双栏、正文最多 6 页（不含 References）；最终以课程官方要求为准。

建议结构：Abstract、Introduction、Related Work、Method、Experiments、Discussion、Conclusion。每项结果应能定位到 run_id、配置和 commit；A–D 审阅各自技术内容。

报告主线更新为 Seam-Aware CutMix：原始方法 → 内容引导 E2 → 接缝矩形选择 E3 → 局部边界优化 E4。区分已有思想、候选设计、已实现行为与实测结论，使用 E4-R 排查不规则形状/面积的作用。

`references.bib` 保留基础论文条目；根 README 提供 SaliencyMix、Puzzle Mix、SmoothMix（CVPR Workshops 2020）与 Seam Carving（2007）的原文入口，精读后补充引用。不得混用同名的其他 SmoothMix 论文。解释本组硬边界选区/路径优化与软边缘混合的区别；查新尚未完成，不保证绝对首创。当前没有成稿或结果。
