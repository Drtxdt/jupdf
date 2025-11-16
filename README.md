
# JUPDF  
批量把 Jupyter Notebook 变成 PDF 的工具  

（因为一个一个点转 PDF 会让人怀疑人生）

## 这是啥

JUPDF 是一个专门处理 Jupyter Notebook 的批量 PDF 转换器。  

它的主要目标很简单：  

你只管把一堆 `.ipynb` 扔给它，它就会尽力把它们规规矩矩地变成 PDF。

为了保证你的笔记本顺利毕业成为 PDF，本项目做了不少幕后操作，例如：

- 自动修复各种稀奇古怪的相对路径图片  
- 笔记本里的附件图片会被挖出来安置妥当  
- 多套转换方案智能降级（WebPDF → LaTeX → HTML），绝不轻易放弃  
- 日志有颜色，转换有进度条，看起来明明白白  

## 它都可以干嘛

- 批量转换 Jupyter Notebook 为 PDF  
- 自动修复 Markdown/HTML 图片路径  
- 自动提取 `.ipynb` 内部 base64 附件图片  
- 自动根据环境智能选择最可行的转换方式  
- 详细日志输出，可彩色可文件  
- 失败了也不丢人，会告诉你为什么  

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```
你最好也确认自己安装了：  
- **Jupyter nbconvert**  
- **Chromium**（用于 WebPDF）  
- **Pandoc**（最好有）  
- **LaTeX 环境**（若你打算走 LaTeX 路线）

不装也可以，它会自动降级，就是文件可能看起来不太“理想状态”。

### 2. 最简单的用法
```bash
python main.py <input_dir> <output_dir>
```
它会在输出目录中生成你想要的 PDF 或 HTML（如果 PDF 实在搞不定）。

### 3. 一些可选参数
```
--verbose        打印更多过程细节
--log-file xxx   输出日志到文件
--no-preprocess  跳过图片路径修复（不推荐）
--timeout N      设置单个文件转换超时时间
```
如果你是大佬（大概率是），可以执行

```bash
python main.py -h
```

来查看所有用法

## 核心设计亮点

### 自动修复图片路径  
Notebook 有各种奇怪的图片引用方式，比如：

- 相对路径瞎写  
- HTML 内嵌标签 `src` 不对  
- 附件藏在 `.ipynb` 里装死  

JUPDF 会一一把它们揪出来，让它们乖乖可以加载。

### 三段式智能降级  
当转换 PDF 时，它会按照优先级这样尝试：

1. WebPDF（基于 Chromium，最稳定）  
2. LaTeX（如果你安装了 TeX 环境）  
3. HTML（最后的底线，总不能啥都不给你）  

不管怎样，我们总会有大保底的。

## 为什么写这个项目

因为写作业写报告交实验的人太懂这种痛苦了：  

不是每个人都喜欢点开 Notebook、点 File、点 Export、点 PDF、点 Save、然后重复几十次。

有了 JUPDF，你只需要努力写代码，让我们来处理文书工作。

## 许可证

本项目采用MIT许可证

通俗来说，就是你可以随意使用，但是必须注明出处

