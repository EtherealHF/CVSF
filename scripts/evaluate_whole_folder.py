#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def link_or_copy(src, dst):
    try:
        os.symlink(str(src.resolve()), str(dst))
    except Exception:
        shutil.copy2(str(src), str(dst))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("-i", "--in_path", required=True)
    parser.add_argument("-r", "--ref_path", default=None)
    parser.add_argument("--save_vis_dir", default=None)
    parser.add_argument("--record_file", default=None)
    parser.add_argument("--ntest", default=None)
    parser.add_argument("--fr_align", default="resize")
    parser.add_argument("--no_fid", action="store_true")
    args, extra = parser.parse_known_args()

    pred_files = list_images(args.in_path)
    ref_files = list_images(args.ref_path) if args.ref_path else []

    if args.ref_path and len(pred_files) != len(ref_files):
        print(f"[warn] pred/ref count mismatch: {len(pred_files)} vs {len(ref_files)}", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="cvsf_eval_all_") as tmp:
        tmp = Path(tmp)
        pred_tmp = tmp / "pred"
        ref_tmp = tmp / "ref"
        pred_tmp.mkdir()
        ref_tmp.mkdir()

        if args.ref_path:
            ref_by_name = {p.name: p for p in ref_files}
        else:
            ref_by_name = {}

        matched = 0
        for pred in pred_files:
            stem = pred.stem
            name = f"{stem}-all{pred.suffix}"
            link_or_copy(pred, pred_tmp / name)
            if args.ref_path and pred.name in ref_by_name:
                ref = ref_by_name[pred.name]
                link_or_copy(ref, ref_tmp / name)
                matched += 1

        print(f"[info] whole-folder eval files: pred={len(pred_files)}, matched_ref={matched}")

        cmd = [
            sys.executable,
            args.evaluator,
            "-i",
            str(pred_tmp),
            "--groups=-all",
            "--fr_align",
            args.fr_align,
        ]
        if args.ref_path:
            cmd += ["-r", str(ref_tmp)]
        if args.save_vis_dir:
            cmd += ["--save_vis_dir", args.save_vis_dir]
        if args.record_file:
            cmd += ["--record_file", args.record_file]
        if args.ntest:
            cmd += ["--ntest", str(args.ntest)]
        if args.no_fid:
            cmd += ["--no_fid"]
        cmd += extra

        return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
