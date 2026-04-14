# DIP Exp1

**拉普拉斯金字塔重建**、**金字塔融合**、**接缝缩放**、**基于接缝的物体移除**。图像读写使用 **Pillow**，数值计算使用 **numpy**，不依赖 OpenCV。

---

## 依赖说明（与 `requirements.txt` 对应）

主程序与 `src/` 中实际用到的模块如下，请区分 **需要 pip 安装的包** 与 **Python 自带标准库**。

| 类型 | 模块 / 包 | 说明 |
|------|-----------|------|
| 需安装 | **numpy** | 数组与数值运算（见 `requirements.txt`） |
| 需安装 | **Pillow**（pip 包名 `pillow`，代码里 `import PIL` / `from PIL import Image`） | 图像读写 |
| 标准库 | **os** | 路径、目录、环境路径 |
| 标准库 | **argparse** | `run_experiments.py` 命令行参数 |
| 标准库 | **sys** | 退出码、`sys.path` 插入 `src`、`stderr` 输出 |

**结论**：只需执行 `pip install -r requirements.txt` 安装 **numpy** 与 **pillow**；`os`、`argparse`、`sys` 随 Python 自带，**不要**写进 `requirements.txt`。

---

## 环境要求

| 项目     | 说明                                           |
| ------ | -------------------------------------------- |
| 操作系统   | macOS / Linux / Windows 均可                   |
| Python | 建议 **3.10 及以上**（与当前 `numpy`、`Pillow` 版本兼容即可） |
| 第三方依赖 | 见 **`requirements.txt`**：仅 **`numpy`**、**`pillow`** |


---

## 1. 获取代码

将仓库克隆或解压到本地目录，下文以 `Exp1` 表示项目根目录（包含 `run_experiments.py` 的文件夹）。

---

## 2. 创建虚拟环境（推荐）

在终端中进入项目根目录后执行：

```bash
cd /path/to/Exp1
python3 -m venv .venv
```

- **macOS / Linux**：用 `python3`；若已安装 `python` 且指向 3.x，也可使用 `python -m venv .venv`。
- **Windows（PowerShell）**：`python -m venv .venv` 或 `py -3 -m venv .venv`。

激活虚拟环境：

- **macOS / Linux（bash/zsh）**：
  ```bash
  source .venv/bin/activate
  ```
- **Windows CMD**：
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Windows PowerShell**：
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

激活后，命令行前缀通常会出现 `(.venv)`。

---

## 3. 安装依赖

在**已激活**虚拟环境的终端中，于项目根目录执行：

```bash
pip install -U pip
pip install -r requirements.txt
```

应安装 **`numpy`** 与 **`pillow`** 两个包（版本满足 `requirements.txt` 中的下限即可）。不会通过 pip 安装 `os` / `argparse` / `sys`。

**可选自检**：

```bash
python -c "import numpy, PIL; print('numpy', numpy.__version__); print('Pillow', PIL.__version__)"
```

---

## 4. 数据与目录约定

实验脚本会读取 `pictures/` 下固定相对路径的图片；请保持目录结构如下（若缺文件，对应任务会报错）：

- **金字塔重建**：`pictures/Image_Pyramid/Image_Reconstruction/building.jpg`
- **金字塔融合**：`pictures/Image_Pyramid/Image_Blending/apple.jpg`、`orange.jpg`、`mask.png`
- **接缝缩放**：`pictures/Seam_Carving/Image_Resizing/Rider.png`
- **物体移除**：`pictures/Seam_Carving/Object_Removal/Couple.png`，以及掩膜（默认见下节）

所有结果默认写入 **`output/`** 目录（脚本会自动创建）。

---

## 5. 运行方式

始终在**项目根目录**执行（且建议已 `source .venv/bin/activate`）。

### 5.1 单次运行某个任务

使用项目内的解释器（未激活 venv 时也可）：

```bash
.venv/bin/python run_experiments.py <task> [选项...]
```

`<task>` 取值：


| 任务名             | 含义          |
| --------------- | ----------- |
| `pyramid_recon` | 金字塔重建       |
| `pyramid_blend` | 金字塔融合       |
| `seam_resize`   | 接缝缩放        |
| `seam_remove`   | 物体移除        |
| `all`           | 按顺序执行以上四个任务 |


### 5.2 一键运行四个任务

```bash
chmod +x run_all.sh   # 仅首次需要
./run_all.sh
```

等价于：

```bash
.venv/bin/python run_experiments.py all
```

### 5.3 常用命令行参数（默认值以 `run_experiments.py` 为准）


| 参数                   | 适用任务                                    | 说明                                                          |
| -------------------- | --------------------------------------- | ----------------------------------------------------------- |
| `--levels`           | `pyramid_recon`, `pyramid_blend`, `all` | 金字塔层数，默认 `6`                                                |
| `--vertical_seams`   | `seam_resize`, `all`                    | 删除的竖直接缝条数，默认 `50`                                           |
| `--horizontal_seams` | `seam_resize`, `all`                    | 删除的水平接缝条数，默认 `0`                                            |
| `--energy`           | `seam_resize`, `seam_remove`, `all`     | `gradient` 或 `saliency`，默认 `saliency`                         |
| `--mask`             | `seam_remove`, `all`                    | 物体移除掩膜路径，默认 `pictures/Seam_Carving/Object_Removal/mask.png` |
| `--max_seams`        | `seam_remove`, `all`                    | 最多删除接缝条数，默认 `69`                                            |
| `--removal_vh_ratio` | `seam_remove`, `all`                    | 每轮先删 N 条竖缝再删 1 条横缝，默认 `40`                                  |


若物体移除提示找不到掩膜，请确认 `--mask` 路径存在，或将掩膜放到默认路径。掩膜语义：**黑（低亮度）≈ 删除区域，白 ≈ 保留**；RGBA 时透明区域按代码约定视为保留（详见 `REPORT.md`）。

---

## 6. 输出文件一览


| 任务    | 主要输出                                                                                                  |
| ----- | ----------------------------------------------------------------------------------------------------- |
| 金字塔重建 | `output/pyramid_reconstruction.png`，以及 `output/pyramid_gaussian/`、`output/pyramid_laplacian/` 下各层 PNG |
| 金字塔融合 | `output/pyramid_blend.png`                                                                            |
| 接缝缩放  | `output/seam_resize.png`                                                                              |
| 物体移除  | `output/seam_object_removal.png`                                                                      |


成功时终端会打印带 **✅** 的分块报告；掩膜缺失时为 **❌** 提示。

---

## 7. 辅助脚本

在已配置 `PYTHONPATH` 或使用仓库内路径的前提下，可预览物体移除掩膜与原图对齐情况：

```bash
.venv/bin/python tools/overlay_couple_mask.py --help
```

默认读取 `Couple.png` 与 `only_mask.png`（可通过参数修改），输出到 `output/couple_mask_overlay.png`。

## 8. 默认配置运行结果

默认配置运行结果已经保存在 `default_results/` 文件夹下，可以直接查看。

