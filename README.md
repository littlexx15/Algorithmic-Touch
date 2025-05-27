# Algorithmic Touch

> 基于深度学习的皮肤病图像识别与简笔画演示 Web 应用

## 项目简介

Algorithmic Touch 是一个基于 Flask 的 Web 应用，能够：

1. 接收用户上传的皮肤病照片  
2. 调用已微调的 ResNet34 模型进行分类预测，返回 7 类皮肤病的概率分布  
3. 将原图转换为简笔画风格，并在结果页面展示处理后的图像和疾病信息  

本项目将模型权重托管在 Hugging Face Hub，通过 `hf_hub_download` 动态获取；前端使用 p5.js 绘制柔和的背景动画，提升用户体验。

## 功能特性

- **模型推理**：7 类皮肤病（痤疮/玫瑰痤疮、基底细胞癌等恶性病变、湿疹、药疹、性病、黑色素瘤和痣、指甲真菌等）的概率输出  
- **简笔画效果**：基于 OpenCV 和噪声渲染，实现可配置的白线手绘效果  
- **背景动画**：使用 p5.js 生成低分辨率噪声着色并高斯模糊，适配移动端刘海屏全屏显示  
- **无大文件泄露**：模型权重通过 HF Hub 获取，不随代码仓库分发  

## 目录结构

\`\`\`plaintext
├── app.py                  # Flask 应用入口
├── requirements.txt       # Python 依赖
├── models/
│   ├── model_utils.py     # 模型下载与加载
│   └── predict.py         # 推理接口
├── static/
│   └── js/bg.js           # p5.js 背景动画代码
├── templates/
│   ├── base.html
│   └── pages/
│       ├── index.html     # 上传页面
│       └── result.html    # 结果展示页面
└── utils/
    ├── sketch_effects.py  # 简笔画生成
    ├── image_helpers.py   # PIL DataURL 和裁剪缩放
    └── disease_info.py    # 各类型皮肤病文案数据
\`\`\`

## 安装与运行

### 前置要求

- Python ≥ 3.8  
- Git  
- 可访问 Hugging Face Hub（网络环境）

### 克隆仓库

\`\`\`bash
git clone https://github.com/littlexx15/Algorithmic-Touch.git
cd AlgorithmicTouch
\`\`\`

### 安装依赖

\`\`\`bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate       # Windows
pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

### 配置环境变量

本项目使用 \`huggingface_hub\` 拉取模型权重，需要提供 HF_TOKEN：

\`\`\`bash
export HF_TOKEN="你的 Hugging Face 访问令牌"
\`\`\`

Windows PowerShell：

\`\`\`powershell
$env:HF_TOKEN = "你的 Hugging Face 访问令牌"
\`\`\`

### 启动服务

\`\`\`bash
python app.py
\`\`\`

默认监听 \`0.0.0.0:8000\`，打开浏览器访问：http://localhost:8000

## 配置项集中管理

可在 \`config.py\`（无则新建）中统一定义常量：

\`\`\`python
# config.py
HF_REPO_ID    = "abdlh/ResNet34_finetuned_for_skin_diseases_by-abdlh"
HF_MODEL_FILE = "skin_model2.pth"

# 简笔画参数
SKETCH_PARAMS = {
    "low_thresh":   80,
    "high_thresh": 150,
    "thick_k":       5,
    "thin_k":        2,
    "noise_sigma": 0.005,
    "bin_thresh":   0.5,
    "r_ratio":     0.5,
    "sigma_ratio": 1.0
}

# p5.js 调色板
P5_PALETTE = [
    "#F5E4D7",
    "#E07A5F",
    "#F4B0A8",
    "#392E2B",
    "#F2CC8F"
]
\`\`\`

然后在各模块中：

\`\`\`python
from config import HF_REPO_ID, HF_MODEL_FILE, SKETCH_PARAMS
\`\`\`

## 单元测试（待补充）

建议使用 \`pytest\` 为核心模块编写单元测试：

1. \`models/predict.py\` 的 \`predict()\` 函数输入示例图像，断言输出字典包含 7 个键且概率总和约为 1  
2. \`utils/image_helpers.crop_and_resize\` 在不同宽高比下测试输出尺寸  
3. \`utils/sketch_effects.to_sketch\` 在空白图像和极端参数下验证不抛异常  

运行（项目根目录）：

\`\`\`bash
pytest tests/
\`\`\`

## 异常处理

- \`app.py\` 中对上传文件缺失做了 400 返回  
- 建议对模型加载、图像处理等环节添加 \`try/except\` 并记录日志  
- 可引入 \`logging\` 模块，将错误写入日志文件

## 贡献指南

欢迎提交 Issue 和 PR：

1. Fork 本仓库  
2. 创建新分支 \`feature/xxx\`  
3. 提交代码并发起 PR

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。
