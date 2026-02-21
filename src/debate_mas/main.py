from __future__ import annotations

import os
import argparse

from dotenv import load_dotenv

from .core.engine import run
from .core.config import CONFIG


def _require_env() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError(
            "缺少环境变量：DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL。请在项目根目录 .env 中配置。"
        )


def main() -> None:
    # 1) 固定从“项目根目录”加载 .env（不依赖当前工作目录 cwd）
    load_dotenv(dotenv_path=os.path.join(CONFIG.BASE_DIR, ".env"))
    _require_env()

    # 2) 定义默认路径
    default_folder = os.path.join(CONFIG.BASE_DIR, "data_test")
    default_output = os.path.join(CONFIG.BASE_DIR, "output_reports")
    default_mission = "审视当前 ETF 池，给出下一周期调仓标的、权重，并附理由与风险提示。"
    default_date = "2025-10-26"

    # 3) 配置 argparse
    parser = argparse.ArgumentParser(description="Debate MAS: 基于多智能体辩论的 ETF 投资决策系统")
    
    parser.add_argument("--mission", type=str, default=default_mission, help="决策任务指令")
    parser.add_argument("--folder", type=str, default=default_folder, help="本地案卷数据文件夹路径")
    parser.add_argument("--date", type=str, default=default_date, help="决策基准日期 (YYYY-MM-DD)")
    parser.add_argument("--output_dir", type=str, default=default_output, help="结果输出目录")

    # 4) 解析参数
    args = parser.parse_args()

    print(f"🚀 Starting Debate MAS...")
    print(f"📂 Data Folder: {args.folder}")
    print(f"📅 Ref Date: {args.date}")
    print(f"🎯 Mission: {args.mission}")

    # 5) 运行引擎
    artifacts = run(
        mission=args.mission,
        ref_date=args.date,
        folder_path=args.folder,
        output_dir=args.output_dir,
        seed_user_message="严格使用案卷证据与工具输出；输出遵守 system prompt 的格式要求。",
    )

    print("✅ 产物已生成：")
    for k, v in (artifacts or {}).items():
        print(f"- {k}: {v}")
