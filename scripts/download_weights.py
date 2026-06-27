#!/usr/bin/env python3
"""
预下载 MedSigLIP 模型权重到本地

用法:
    # 下载到默认位置 ./pretrained/medsiglip-448
    python scripts/download_weights.py

    # 下载到指定目录
    python scripts/download_weights.py --output /data/models/medsiglip-448

    # 使用镜像加速（国内用户）
    HF_ENDPOINT=https://hf-mirror.com python scripts/download_weights.py

首次运行需要:
    1. 访问 https://huggingface.co/google/medsiglip-448 同意使用条款
    2. huggingface-cli login 登录你的 HuggingFace 账号
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download MedSigLIP weights locally")
    parser.add_argument("--output", type=str, default="./pretrained/medsiglip-448",
                        help="Output directory for weights")
    parser.add_argument("--model", type=str, default="google/medsiglip-448",
                        help="HuggingFace model ID")
    parser.add_argument("--force", action="store_true", help="Force re-download even if exists")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        print(f"Weights already exist at: {output_dir}")
        print(f"Size: {_dir_size(output_dir):.1f} MB")
        print(f"\nTo re-download, use --force, or manually delete the directory.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model} to {output_dir} ...")
    print(f"This will download ~1.6 GB of model weights.")
    print(f"If stuck, try setting mirror: HF_ENDPOINT=https://hf-mirror.com python scripts/download_weights.py\n")

    try:
        from huggingface_hub import snapshot_download

        # 检查是否已登录
        from huggingface_hub import whoami
        try:
            user = whoami()
            print(f"Logged in as: {user['name']}")
        except Exception:
            print("[WARN] Not logged in to HuggingFace. Run: huggingface-cli login")
            print("       You also need to accept the license at:")
            print(f"       https://huggingface.co/{args.model}\n")
            print("Proceeding anyway (public repo may work without login for some models)...\n")

        snapshot_download(
            repo_id=args.model,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model.msgpack", "tf_model.h5"],
        )

        print(f"\n✓ Download complete!")
        print(f"  Path: {output_dir}")
        print(f"  Size: {_dir_size(output_dir):.1f} MB")
        print(f"\n  Update your config YAML to use this path:")
        print(f"    model:")
        print(f"      name: \"{output_dir}\"")
        print(f"      local_files_only: true")

    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check network connection")
        print(f"  2. Try mirror: HF_ENDPOINT=https://hf-mirror.com python scripts/download_weights.py")
        print(f"  3. Ensure you accepted the license at https://huggingface.co/{args.model}")
        print(f"  4. Login: huggingface-cli login")
        sys.exit(1)


def _dir_size(path: Path) -> float:
    """目录大小 (MB)"""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


if __name__ == "__main__":
    main()
