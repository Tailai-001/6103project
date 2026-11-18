# 正确性检查

运行 `python -m unittest discover -s tests -v`。`test_member_b.py` 覆盖已知 ramp 的 Sobel、常量图、integral image 对照暴力区域均值、E1/K=1 位级一致、gamma=0 退化、像素/面积/标签、候选同面积及边界、K 前缀和 RNG、不修改原图、同分首选、p=0、空/满/极小框、batch=1、四种浮点 dtype、错误参数及 CUDA。CUDA 不可用时仅跳过 CUDA 测试。

C 尚需添加 seam/DP 的独立正确性测试，不能用 B 测试通过代替其验收。

`test_checkpoint.py` 验证存储中断时旧 checkpoint 仍可读取，后续保存可正常完成；所有 GPU 测试的临时文件使用项目内 TMPDIR。
