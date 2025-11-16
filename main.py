#!/usr/bin/env python3
"""
Jupyter 笔记本批量 PDF 转换工具
支持自动修复相对路径图片、智能转换方式降级
"""

import os
import sys
import json
import shutil
import base64
import logging
import argparse
import subprocess
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ============================================================================
# 颜色和样式定义
# ============================================================================

class Colors:
    """ANSI 颜色代码"""
    # 基础颜色
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # 亮色
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # 背景色
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    
    # 样式
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    HIDDEN = '\033[8m'
    STRIKETHROUGH = '\033[9m'
    
    # 重置
    RESET = '\033[0m'
    
    # 便利方法
    @staticmethod
    def success(text: str) -> str:
        """成功消息（绿色）"""
        return f"{Colors.BRIGHT_GREEN}{text}{Colors.RESET}"
    
    @staticmethod
    def error(text: str) -> str:
        """错误消息（红色）"""
        return f"{Colors.BRIGHT_RED}{text}{Colors.RESET}"
    
    @staticmethod
    def warning(text: str) -> str:
        """警告消息（黄色）"""
        return f"{Colors.BRIGHT_YELLOW}{text}{Colors.RESET}"
    
    @staticmethod
    def info(text: str) -> str:
        """信息消息（青色）"""
        return f"{Colors.BRIGHT_CYAN}{text}{Colors.RESET}"
    
    @staticmethod
    def bold(text: str) -> str:
        """加粗"""
        return f"{Colors.BOLD}{text}{Colors.RESET}"
    
    @staticmethod
    def dim(text: str) -> str:
        """暗淡"""
        return f"{Colors.DIM}{text}{Colors.RESET}"


class AsciiArt:
    """ASCII 艺术字"""
    
    JUPDF = """
     _____  _    _ ____  ____  ______
    |_   _|| |  | |  _ \|  _ \|  ____|
      | |  | |  | | |_) | | | | |__
      | |  | |  | |  __/| | | |  __|
     _| |  | |__| | |   | |_| | |
    |___|   \____/|_|   |____/|_|
    """
    
    @staticmethod
    def print_header():
        """打印头部"""
        print(f"{Colors.BRIGHT_CYAN}{AsciiArt.JUPDF}{Colors.RESET}")


# ============================================================================
# 自定义日志格式化器
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.BRIGHT_BLACK,
        logging.INFO: Colors.BRIGHT_CYAN,
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.BRIGHT_RED,
        logging.CRITICAL: Colors.BG_RED + Colors.WHITE,
    }
    
    
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        # 获取时间
        timestamp = self.formatTime(record, '%H:%M:%S')
        
        # 获取级别颜色
        level_color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
        
        
        # 获取消息
        message = record.getMessage()
        
        # 根据内容选择消息颜色
        if '✓' in message or 'success' in message.lower():
            message_color = Colors.BRIGHT_GREEN
        elif '✗' in message or 'failed' in message.lower() or 'error' in message.lower():
            message_color = Colors.BRIGHT_RED
        elif '⚠' in message or 'warning' in message.lower():
            message_color = Colors.BRIGHT_YELLOW
        else:
            message_color = Colors.WHITE
        
        # 构建日志字符串
        log_string = (
            f"{Colors.DIM}[{timestamp}]{Colors.RESET} "
            f"{level_color}{record.levelname:<8}{Colors.RESET} "
            f"{message_color}{message}{Colors.RESET}"
        )
        
        return log_string


class ProgressBarManager:
    """进度条管理器"""
    
    @staticmethod
    def create_bar(items: List, desc: str = "处理中", unit: str = "it") -> tqdm:
        """创建进度条"""
        if not TQDM_AVAILABLE:
            return iter(items)
        
        return tqdm(
            items,
            desc=desc,
            bar_format=(
                f'{Colors.BRIGHT_CYAN}{Colors.BOLD}'
                f'{{desc}}{Colors.RESET} '
                f'{Colors.BRIGHT_BLUE}|{{bar}}|{Colors.RESET} '
                f'{Colors.BRIGHT_YELLOW}{{percentage:3.0f}}%{Colors.RESET} '
                f'{Colors.DIM}[{{elapsed}}<{{remaining}}]{Colors.RESET}'
            ),
            ncols=100,
            colour='cyan',
            unit=unit,
            unit_scale=True,
        )


# ============================================================================
# 数据类和枚举
# ============================================================================

class ConversionMethod(Enum):
    """转换方法枚举"""
    WEBPDF = "webpdf"      # HTML + Chromium → PDF（最稳定）
    LATEX = "latex"        # LaTeX → PDF
    HTML = "html"          # HTML 格式


@dataclass
class ImageInfo:
    """图片信息数据类"""
    original_path: str
    absolute_path: str
    cell_index: int
    exists: bool
    img_type: str = "markdown"  # markdown, html, attachment


@dataclass
class ConversionResult:
    """转换结果数据类"""
    notebook_path: Path
    success: bool
    output_path: Optional[Path] = None
    method: Optional[ConversionMethod] = None
    error_message: Optional[str] = None


# ============================================================================
# 日志管理
# ============================================================================

class LogManager:
    """日志管理器"""
    
    @staticmethod
    def setup(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
        """配置日志"""
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # 清除已有处理器
        logger.handlers.clear()
        
        # 控制台处理器
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # 使用自定义格式化器
        formatter = ColoredFormatter()
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        # 文件处理器（无颜色）
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            plain_formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(plain_formatter)
            logger.addHandler(file_handler)
        
        return logger


# ============================================================================
# 环境管理
# ============================================================================

class EnvironmentManager:
    """环境检查和配置管理"""
    
    @staticmethod
    def check_nbconvert() -> bool:
        """检查 nbconvert 是否已安装"""
        logging.info("检查依赖包...")
        try:
            import nbconvert
            logging.info(Colors.success("nbconvert 已安装"))
            return True
        except ImportError:
            logging.error(Colors.error("nbconvert 未安装"))
            return False
    
    @staticmethod
    def install_nbconvert() -> bool:
        """安装 nbconvert"""
        logging.info("正在安装依赖...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "nbconvert", "pandas", "matplotlib", "lxml", "tqdm", "pillow"
            ])
            logging.info(Colors.success("依赖安装完成"))
            return True
        except subprocess.CalledProcessError as e:
            logging.error(Colors.error(f"依赖安装失败: {e}"))
            return False
    
    @staticmethod
    def find_pandoc() -> Optional[Path]:
        """查找 Pandoc"""
        logging.debug("查找 Pandoc...")
        
        # 检查 PATH
        pandoc_path = shutil.which("pandoc")
        if pandoc_path:
            logging.info(Colors.success(f"在 PATH 中找到 Pandoc"))
            return Path(pandoc_path)
        
        # 常见位置
        common_paths = [
            r"C:\Program Files\Pandoc\pandoc.exe",
            r"D:\pandoc-3.8.2.1-windows-x86_64\pandoc-3.8.2.1\pandoc.exe",
            "/usr/local/bin/pandoc",
            "/opt/homebrew/bin/pandoc",
            "/usr/bin/pandoc"
        ]
        
        for path in common_paths:
            if Path(path).exists():
                logging.info(Colors.success(f"在常见路径找到 Pandoc"))
                return Path(path)
        
        logging.warning(Colors.warning("未找到 Pandoc"))
        return None
    
    @staticmethod
    def setup_paths(pandoc_path: Optional[Path] = None) -> Dict[str, str]:
        """设置环境变量"""
        logging.debug("配置环境...")
        env = os.environ.copy()
        
        if pandoc_path and pandoc_path.exists():
            pandoc_dir = pandoc_path.parent
            env['PATH'] = f"{pandoc_dir}{os.pathsep}{env['PATH']}"
            logging.info(Colors.success("已配置 Pandoc 路径"))
        
        return env


# ============================================================================
# 笔记本处理
# ============================================================================

class ImagePathFixer:
    """处理笔记本中的图片路径"""
    
    def __init__(self, notebook_path: Path):
        self.notebook_path = notebook_path
        self.notebook_dir = notebook_path.parent.resolve()
        self.processed_images: List[ImageInfo] = []
        self.missing_images: List[ImageInfo] = []
    
    def fix(self, verbose: bool = False) -> bool:
        """修复笔记本中的所有图片路径"""
        try:
            with open(self.notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            modified = self._process_cells(notebook, verbose)
            modified |= self._extract_attachments(notebook, verbose)
            
            if modified:
                self._backup_and_save(notebook)
                return True
            
            return False
        
        except Exception as e:
            logging.error(Colors.error(f"笔记本预处理出错: {e}"))
            return False
    
    def _process_cells(self, notebook: dict, verbose: bool) -> bool:
        """处理笔记本单元格中的图片引用"""
        modified = False
        
        for cell_idx, cell in enumerate(notebook.get('cells', [])):
            if cell.get('cell_type') != 'markdown':
                continue
            
            source = self._get_source_text(cell)
            original_source = source
            
            # 处理 Markdown 图片
            source = self._fix_markdown_images(source, cell_idx, verbose)
            
            # 处理 HTML 图片
            source = self._fix_html_images(source, cell_idx, verbose)
            
            if source != original_source:
                cell['source'] = source
                modified = True
        
        return modified
    
    def _fix_markdown_images(self, source: str, cell_idx: int, verbose: bool) -> str:
        """修复 Markdown 格式的图片 ![alt](path)"""
        
        def replace_image(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            
            # 跳过网络路径和附件
            if img_path.startswith(('http://', 'https://', 'ftp://', 'attachment:')):
                return match.group(0)
            
            abs_path = self._resolve_image_path(img_path, cell_idx, verbose)
            return f'![{alt_text}]({abs_path})'
        
        pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        return re.sub(pattern, replace_image, source)
    
    def _fix_html_images(self, source: str, cell_idx: int, verbose: bool) -> str:
        """修复 HTML 格式的图片 <img src="path">"""
        
        def replace_image(match):
            img_tag = match.group(0)
            src = match.group(1)
            
            if src.startswith(('http://', 'https://', 'data:')):
                return img_tag
            
            abs_path = self._resolve_image_path(src, cell_idx, verbose)
            return img_tag.replace(src, abs_path)
        
        pattern = r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>'
        return re.sub(pattern, replace_image, source)
    
    def _resolve_image_path(self, img_path: str, cell_idx: int, verbose: bool) -> str:
        """解析图片路径"""
        abs_path = (self.notebook_dir / img_path).resolve()
        
        image_info = ImageInfo(
            original_path=img_path,
            absolute_path=str(abs_path),
            cell_index=cell_idx,
            exists=abs_path.exists()
        )
        
        if abs_path.exists():
            self.processed_images.append(image_info)
            if verbose:
                logging.debug(f"修复图片: {img_path}")
        else:
            self.missing_images.append(image_info)
            logging.warning(Colors.warning(f"[单元格 {cell_idx}] 图片不存在: {img_path}"))
        
        return str(abs_path)
    
    def _extract_attachments(self, notebook: dict, verbose: bool) -> bool:
        """提取附件图片"""
        if 'attachments' not in notebook:
            return False
        
        modified = False
        attachments_dir = self.notebook_dir / 'attachments'
        attachments_dir.mkdir(exist_ok=True)
        
        for cell_idx, cell in enumerate(notebook.get('cells', [])):
            if 'attachments' not in cell:
                continue
            
            for att_name, att_data in cell['attachments'].items():
                for mime_type, base64_data in att_data.items():
                    if not mime_type.startswith('image/'):
                        continue
                    
                    if self._extract_single_attachment(
                        att_name, base64_data, mime_type, 
                        cell_idx, attachments_dir, verbose
                    ):
                        modified = True
        
        return modified
    
    def _extract_single_attachment(
        self, att_name: str, base64_data: str, mime_type: str,
        cell_idx: int, attachments_dir: Path, verbose: bool
    ) -> bool:
        """提取单个附件"""
        try:
            ext = mime_type.split('/')[-1]
            ext = 'jpg' if ext == 'jpeg' else ext
            
            filename = f"attachment_{cell_idx}_{att_name}.{ext}"
            filepath = attachments_dir / filename
            
            # 保存文件
            img_data = base64.b64decode(base64_data)
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            if verbose:
                logging.debug(f"提取附件: {filename}")
            
            return True
        
        except Exception as e:
            logging.warning(Colors.warning(f"提取附件失败 {att_name}"))
            return False
    
    def _backup_and_save(self, notebook: dict):
        """备份并保存笔记本"""
        backup_path = self.notebook_path.with_suffix('.ipynb.backup')
        if not backup_path.exists():
            shutil.copy2(self.notebook_path, backup_path)
            logging.debug(f"已创建备份")
        
        with open(self.notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        logging.info(Colors.success(f"处理了 {len(self.processed_images)} 个图片"))
    
    @staticmethod
    def _get_source_text(cell: dict) -> str:
        """获取单元格源代码文本"""
        source = cell.get('source', [])
        return ''.join(source) if isinstance(source, list) else source


# ============================================================================
# 转换执行
# ============================================================================

class ConversionStrategy(ABC):
    """转换策略基类"""
    
    @abstractmethod
    def can_execute(self) -> bool:
        """检查是否可以执行此策略"""
        pass
    
    @abstractmethod
    def convert(self, notebook_path: Path, output_dir: Path, 
                env: Dict[str, str], timeout: int) -> bool:
        """执行转换"""
        pass
    
    def _build_command(self, notebook_path: Path, output_dir: Path,
                      to_format: str) -> List[str]:
        """构建 nbconvert 命令"""
        return [
            "jupyter", "nbconvert",
            "--to", to_format,
            str(notebook_path.name),
            "--output-dir", str(output_dir),
        ]


class WebPdfStrategy(ConversionStrategy):
    """WebPDF 转换策略（基于 Chromium）"""
    
    def can_execute(self) -> bool:
        return True  # 总是可用
    
    def convert(self, notebook_path: Path, output_dir: Path,
                env: Dict[str, str], timeout: int) -> bool:
        cmd = self._build_command(notebook_path, output_dir, "webpdf")
        return self._execute_conversion(cmd, env, timeout)
    
    @staticmethod
    def _execute_conversion(cmd: List[str], env: Dict[str, str], 
                           timeout: int) -> bool:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=timeout, env=env)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


class LatexStrategy(ConversionStrategy):
    """LaTeX 转换策略"""
    
    def can_execute(self) -> bool:
        try:
            subprocess.run(["xelatex", "--version"], capture_output=True,
                         timeout=5)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def convert(self, notebook_path: Path, output_dir: Path,
                env: Dict[str, str], timeout: int) -> bool:
        cmd = self._build_command(notebook_path, output_dir, "pdf")
        cmd.extend(["--LatexExporter.latex_command", "xelatex"])
        return self._execute_conversion(cmd, env, timeout)
    
    @staticmethod
    def _execute_conversion(cmd: List[str], env: Dict[str, str],
                           timeout: int) -> bool:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=timeout, env=env)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


class HtmlStrategy(ConversionStrategy):
    """HTML 转换策略（备选方案）"""
    
    def can_execute(self) -> bool:
        return True
    
    def convert(self, notebook_path: Path, output_dir: Path,
                env: Dict[str, str], timeout: int) -> bool:
        cmd = self._build_command(notebook_path, output_dir, "html")
        return self._execute_conversion(cmd, env, timeout)
    
    @staticmethod
    def _execute_conversion(cmd: List[str], env: Dict[str, str],
                           timeout: int) -> bool:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=timeout, env=env)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


class NotebookConverter:
    """笔记本转换器"""
    
    # 转换策略优先级
    STRATEGIES = [
        (ConversionMethod.WEBPDF, WebPdfStrategy()),
        (ConversionMethod.LATEX, LatexStrategy()),
        (ConversionMethod.HTML, HtmlStrategy()),
    ]
    
    TIMEOUT = 300
    
    def __init__(self, pandoc_path: Optional[Path] = None):
        self.env = EnvironmentManager.setup_paths(pandoc_path)
    
    def convert(self, notebook_path: Path, output_dir: Path,
               preprocess: bool = True, verbose: bool = False) -> ConversionResult:
        """转换单个笔记本"""
        
        logging.info(f"{Colors.BOLD}转换: {notebook_path.name}{Colors.RESET}")
        
        # 预处理阶段
        if preprocess and not self._preprocess(notebook_path, verbose):
            return ConversionResult(notebook_path, False)
        
        # 转换阶段
        original_cwd = os.getcwd()
        notebook_dir = notebook_path.parent.resolve()
        os.chdir(notebook_dir)
        
        try:
            # 设置 LaTeX 搜索路径
            env = self.env.copy()
            env['TEXINPUTS'] = f"{notebook_dir}{os.pathsep}{env.get('TEXINPUTS', '')}"
            
            # 尝试不同的转换策略
            for method, strategy in self.STRATEGIES:
                if not strategy.can_execute():
                    continue
                
                method_name = {
                    ConversionMethod.WEBPDF: "WebPDF",
                    ConversionMethod.LATEX: "LaTeX",
                    ConversionMethod.HTML: "HTML",
                }.get(method, method.value)
                
                logging.info(f"尝试: {Colors.BRIGHT_BLUE}{method_name}{Colors.RESET}...")
                
                if strategy.convert(notebook_path, output_dir, env, self.TIMEOUT):
                    output_file = self._find_output_file(notebook_path, output_dir, method)
                    
                    if output_file and output_file.exists():
                        size_kb = output_file.stat().st_size / 1024
                        logging.info(
                            Colors.success(
                                f"{output_file.name} ({Colors.DIM}{size_kb:.1f} KB{Colors.RESET})"
                            )
                        )
                        
                        return ConversionResult(
                            notebook_path, True, output_file, method
                        )
            
            logging.error(Colors.error("所有转换方式均失败"))
            return ConversionResult(notebook_path, False)
        
        finally:
            os.chdir(original_cwd)
    
    @staticmethod
    def _preprocess(notebook_path: Path, verbose: bool) -> bool:
        """预处理笔记本"""
        logging.info("  预处理笔记本...")
        
        fixer = ImagePathFixer(notebook_path)
        if not fixer.fix(verbose):
            return True  # 没有修改也继续
        
        if fixer.missing_images:
            logging.warning(
                Colors.warning(f"发现 {len(fixer.missing_images)} 个缺失的图片")
            )
        
        return True
    
    @staticmethod
    def _find_output_file(notebook_path: Path, output_dir: Path,
                         method: ConversionMethod) -> Optional[Path]:
        """查找输出文件"""
        if method == ConversionMethod.HTML:
            return output_dir / f"{notebook_path.stem}.html"
        else:
            return output_dir / f"{notebook_path.stem}.pdf"


# ============================================================================
# 批量处理
# ============================================================================

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, converter: NotebookConverter):
        self.converter = converter
    
    def process(self, input_path: Path, output_dir: Optional[Path],
               recursive: bool, preprocess: bool, verbose: bool,
               show_progress: bool = True) -> Tuple[int, int]:
        """处理笔记本文件"""
        
        # 收集笔记本
        notebooks = self._collect_notebooks(input_path, recursive)
        if not notebooks:
            logging.warning(Colors.warning(f"未找到笔记本文件"))
            return 0, 0
        
        # 确定输出目录
        if output_dir is None:
            output_dir = input_path if input_path.is_dir() else input_path.parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info("")
        logging.info(Colors.bold(f"{'═' * 70}"))
        logging.info(
            Colors.BRIGHT_CYAN + 
            f"  批量转换  {len(notebooks)} 个笔记本文件" +
            Colors.RESET
        )
        logging.info(Colors.bold(f"{'═' * 70}"))
        logging.info("")
        
        # 执行转换
        success_count = self._convert_notebooks(
            notebooks, output_dir, preprocess, verbose, show_progress
        )
        
        # 输出统计
        self._print_summary(success_count, len(notebooks))
        
        return success_count, len(notebooks)
    
    @staticmethod
    def _collect_notebooks(input_path: Path, recursive: bool) -> List[Path]:
        """收集笔记本文件"""
        if input_path.is_file() and input_path.suffix == '.ipynb':
            return [input_path]
        
        if input_path.is_dir():
            pattern = "**/*.ipynb" if recursive else "*.ipynb"
            notebooks = sorted(list(input_path.glob(pattern)))
            if notebooks:
                logging.info(Colors.info(f"找到 {len(notebooks)} 个笔记本文件"))
            return notebooks
        
        logging.error(Colors.error(f"路径不存在: {input_path}"))
        return []
    
    def _convert_notebooks(self, notebooks: List[Path], output_dir: Path,
                          preprocess: bool, verbose: bool,
                          show_progress: bool) -> int:
        """转换笔记本列表"""
        success_count = 0
        
        # 创建进度条
        iterator = notebooks
        if show_progress and TQDM_AVAILABLE:
            iterator = ProgressBarManager.create_bar(
                notebooks,
                desc=Colors.BRIGHT_CYAN + "进度" + Colors.RESET,
                unit="个"
            )
        
        for notebook in iterator:
            result = self.converter.convert(
                notebook, output_dir, preprocess, verbose
            )
            
            if result.success:
                success_count += 1
        
        return success_count
    
    @staticmethod
    def _print_summary(success_count: int, total_count: int):
        """输出转换总结"""
        logging.info("")
        logging.info(Colors.bold(f"{'═' * 70}"))
        
        if success_count == total_count:
            logging.info(
                Colors.BRIGHT_GREEN + 
                f"  ✓ 全部转换完成  {success_count}/{total_count}" +
                Colors.RESET
            )
        else:
            logging.info(
                Colors.BRIGHT_YELLOW + 
                f"  ⚠ 部分转换完成  {success_count}/{total_count}" +
                Colors.RESET
            )
        
        logging.info(Colors.bold(f"{'═' * 70}"))
        logging.info("")


# ============================================================================
# 命令行接口
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='Jupyter 笔记本批量 PDF 转换工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
{Colors.BRIGHT_CYAN}使用示例:{Colors.RESET}
  {Colors.BOLD}基本转换:{Colors.RESET}
    python notebook_to_pdf.py notebook.ipynb
    python notebook_to_pdf.py /path/to/notebooks -r
  
  {Colors.BOLD}高级用法:{Colors.RESET}
    python notebook_to_pdf.py notebook.ipynb -v
    python notebook_to_pdf.py notebook.ipynb --log conversion.log
    python notebook_to_pdf.py notebook.ipynb -o /output/dir

{Colors.BRIGHT_CYAN}功能特性:{Colors.RESET}
  {Colors.GREEN}✓{Colors.RESET} 自动转换相对路径图片为绝对路径
  {Colors.GREEN}✓{Colors.RESET} 智能降级：webpdf → latex → html
  {Colors.GREEN}✓{Colors.RESET} 支持 Markdown 和 HTML 格式图片
  {Colors.GREEN}✓{Colors.RESET} 自动提取附件图片
  {Colors.GREEN}✓{Colors.RESET} 批量处理和递归搜索
        '''
    )
    
    parser.add_argument('input', help='输入文件或目录路径')
    parser.add_argument('-o', '--output', help='输出目录')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='递归搜索子目录')
    parser.add_argument('--pandoc-path', type=Path, help='Pandoc 路径')
    parser.add_argument('--no-preprocess', action='store_true',
                       help='跳过预处理步骤')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='详细输出')
    parser.add_argument('--log', help='日志文件路径')
    parser.add_argument('--no-progress', action='store_true',
                       help='禁用进度条')
    
    return parser


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 打印头部
    AsciiArt.print_header()
    
    # 设置日志
    LogManager.setup(args.verbose, args.log)
    
    # 检查依赖
    if not EnvironmentManager.check_nbconvert():
        if input(f"\n{Colors.YELLOW}是否自动安装？(y/n): {Colors.RESET}").lower() == 'y':
            if not EnvironmentManager.install_nbconvert():
                sys.exit(1)
        else:
            sys.exit(1)
    
    # 初始化
    logging.info("")
    pandoc_path = args.pandoc_path or EnvironmentManager.find_pandoc()
    converter = NotebookConverter(pandoc_path)
    processor = BatchProcessor(converter)
    
    # 执行
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else None
    
    success_count, total_count = processor.process(
        input_path, output_dir, args.recursive,
        not args.no_preprocess, args.verbose,
        not args.no_progress
    )
    
    # 返回退出码
    sys.exit(0 if success_count == total_count else 1)


if __name__ == "__main__":
    main()