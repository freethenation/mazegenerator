#!/usr/bin/env python3
"""
Themed Maze Generator CLI

- Normalizes mazegen PNG into crisp black line art on white before sending to OpenAI.
- Builds a binary alpha mask (255=protected maze region, 0=editable padding) with optional dilation.
- OpenAI edits only the unprotected areas around the maze.

Debug outputs (if --debug):
  1_expanded_pre_resize.png
  2_mask_pre_resize.png
  3_expanded_post_resize.png
  4_mask_post_resize.png
  5_themed_from_openai.png
"""

import argparse
import base64
import io
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai")
    sys.exit(1)

try:
    from PIL import Image, ImageFilter
except ImportError:
    print("Error: pillow package not installed. Run: pip install pillow")
    sys.exit(1)

EXPAND_MODES = ["uniform", "scene-bottom", "scene-top"]
TARGET_SIZE = (1024, 1536)  # portrait
TARGET_RATIO = TARGET_SIZE[0] / TARGET_SIZE[1]
BASE_PADDING = 75


def parse_args():
    p = argparse.ArgumentParser(description="Generate themed maze coloring book pages (uploads real maze).")
    p.add_argument("--theme", required=True)
    p.add_argument("--prompt", help="Custom prompt. Use {theme} placeholder.")

    p.add_argument("--model", default="gpt-image-1", help="Recommended: gpt-image-1")
    p.add_argument("--quality", choices=["low", "medium", "high", "auto"], default="high")
    p.add_argument(
        "--input-fidelity",
        choices=["low", "high"],
        default="high",
        help="Only applied when model is gpt-image-1.",
    )
    p.add_argument("--expand", choices=EXPAND_MODES + ["random"], default="random")
    p.add_argument("--dilate", type=int, default=24,
                   help="Pixels to expand protected maze region (prevents edge bleed).")

    # Make uploaded maze darker/crisper
    p.add_argument("--threshold", type=int, default=215,
                   help="0-255. Pixels darker than this become black; others become white.")
    p.add_argument("--line-thicken", type=int, default=3,
                   help="Odd kernel size to thicken maze lines (1=off). e.g. 3,5,7")

    # mazegen passthrough
    p.add_argument(
        "-m", "--maze-type", type=int, choices=range(7), default=0, metavar="TYPE",
        help="""Maze type (random if not specified):
            0: Rectangular
            1: Hexagonal (triangular lattice)
            2: Honeycomb
            3: Circular
            4: Circular (triangular lattice)
            5: User-defined
            6: Triangular (rectangular lattice)""",
    )
    p.add_argument(
        "-a", "--algorithm", type=int, choices=range(5), default=0, metavar="ALGO",
        help="""Generation algorithm:
            0: Kruskal's algorithm (default)
            1: Depth-first search
            2: Breadth-first search
            3: Loop-erased random walk
            4: Prim's algorithm""",
    )
    p.add_argument("-s", "--size", type=int, default=10,
                   help="Size for non-rectangular mazes (types 1-6)")
    p.add_argument("-W", "--width", type=int, default=10,
                   help="Width for rectangular mazes (type 0)")
    p.add_argument("-H", "--height", type=int, default=10,
                   help="Height for rectangular mazes (type 0)")

    p.add_argument("-o", "--output", default=None,
                   help="Output file prefix (default: out/<theme>)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def generate_maze(args, script_dir: Path) -> Path:
    mazegen_path = script_dir / "src" / "mazegen"
    if not mazegen_path.exists():
        print(f"Error: mazegen not found at {mazegen_path}\nPlease build it first: cd src && make")
        sys.exit(1)

    # Random maze type if not specified (exclude 5: User-defined)
    maze_type = args.maze_type if args.maze_type is not None else random.choice([0, 1, 2, 3, 4, 6])

    cmd = [str(mazegen_path), "-m", str(maze_type), "-a", str(args.algorithm), "-t", "1", "-o", args.output]
    if maze_type == 0:
        cmd.extend(["-w", str(args.width), "-h", str(args.height)])
    else:
        cmd.extend(["-s", str(args.size)])

    print(f"Generating maze: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating maze:\n{e.stderr}")
        sys.exit(1)

    png_path = Path(f"{args.output}.png")
    if not png_path.exists():
        print(f"Error: Expected maze file not found: {png_path}\n"
              "Make sure gnuplot is installed with pngcairo support.")
        sys.exit(1)
    return png_path


def normalize_maze_lineart(maze_path: Path, threshold: int, line_thicken: int) -> Image.Image:
    """
    Convert mazegen output into crisp black lines on white (RGB).
    This is what we upload to OpenAI (so it sees YOUR maze).
    """
    img = Image.open(maze_path).convert("L")

    thr = max(0, min(255, int(threshold)))
    bw = img.point(lambda p: 0 if p < thr else 255, mode="L")  # 0=black, 255=white

    k = int(line_thicken)
    if k and k > 1:
        if k % 2 == 0:
            k += 1
        # Thicken black lines: invert -> maxfilter -> invert back
        inv = Image.eval(bw, lambda p: 255 - p)
        inv = inv.filter(ImageFilter.MaxFilter(size=max(1, k)))
        bw = Image.eval(inv, lambda p: 255 - p)

    return Image.merge("RGB", (bw, bw, bw))


def _binary_alpha_mask_rgba(size, protected_rect, dilate_px: int) -> Image.Image:
    """RGBA mask where alpha=255 protects, alpha=0 editable."""
    w, h = size
    alpha = Image.new("L", (w, h), 0)
    alpha.paste(255, protected_rect)

    if dilate_px > 0:
        k = int(dilate_px)
        if k % 2 == 0:
            k += 1
        alpha = alpha.filter(ImageFilter.MaxFilter(size=max(1, k)))

    mask = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    mask.putalpha(alpha)
    return mask


def expand_canvas_and_create_mask(
    maze_lineart_rgb: Image.Image,
    expand_mode: str,
    dilate_px: int,
    debug: bool,
):
    if expand_mode == "random":
        expand_mode = random.choice(EXPAND_MODES)

    orig_w, orig_h = maze_lineart_rgb.size

    pad_top = pad_bottom = pad_left = pad_right = BASE_PADDING
    init_w = orig_w + pad_left + pad_right
    init_h = orig_h + pad_top + pad_bottom

    target_h = int(init_w / TARGET_RATIO)
    extra_h = max(0, target_h - init_h)

    if expand_mode == "uniform":
        pad_top += extra_h // 2
        pad_bottom += extra_h - (extra_h // 2)
    elif expand_mode == "scene-bottom":
        pad_bottom += extra_h
    else:
        pad_top += extra_h

    new_w = orig_w + pad_left + pad_right
    new_h = orig_h + pad_top + pad_bottom

    expanded = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    expanded.paste(maze_lineart_rgb, (pad_left, pad_top))

    protected_rect = (pad_left, pad_top, pad_left + orig_w, pad_top + orig_h)
    mask = _binary_alpha_mask_rgba((new_w, new_h), protected_rect, dilate_px=dilate_px)

    if debug:
        debug_dir = Path("out/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        expanded.save(debug_dir / "1_expanded_pre_resize.png")
        mask.save(debug_dir / "2_mask_pre_resize.png")

    # Keep line art crisp
    expanded = expanded.resize(TARGET_SIZE, Image.Resampling.NEAREST)
    mask = mask.resize(TARGET_SIZE, Image.Resampling.NEAREST)

    if debug:
        debug_dir = Path("out/debug")
        expanded.save(debug_dir / "3_expanded_post_resize.png")
        mask.save(debug_dir / "4_mask_post_resize.png")

    return expanded, mask, expand_mode


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=9)
    return buf.getvalue()


def build_prompt(theme: str, custom: str | None) -> str:
    if custom:
        return custom.format(theme=theme)
    return (
        f"Create a {theme} themed coloring-book scene ONLY in the white padding area OUTSIDE the central maze. "
        "Add themed characters and simple elements in the margins, leaving plenty of blank areas for coloring. "
        "Black line art on white background. "
        "Add 'Start' and 'End' labels near the two gaps/openings in the maze border. "
        "CRITICAL: The maze in the center must remain COMPLETELY UNTOUCHED - do not draw anything on, over, or crossing the maze lines or paths. "
        "Do NOT add any new maze patterns, grids, or labyrinth-like elements anywhere. "
        "Do NOT add any border or frame around the maze."
    )


def transform_maze(maze_path: Path, args) -> Path:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI()

    maze_lineart = normalize_maze_lineart(
        maze_path, threshold=args.threshold, line_thicken=args.line_thicken
    )

    expanded_img, mask_img, _ = expand_canvas_and_create_mask(
        maze_lineart, args.expand, dilate_px=args.dilate, debug=args.debug
    )

    prompt = build_prompt(args.theme, args.prompt)

    expanded_png = _png_bytes(expanded_img)
    mask_png = _png_bytes(mask_img)

    if len(mask_png) >= 4 * 1024 * 1024:
        print(f"Error: mask.png is {len(mask_png)} bytes (>= 4MB). Reduce --dilate.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as td:
        img_path = Path(td) / "expanded.png"
        msk_path = Path(td) / "mask.png"
        img_path.write_bytes(expanded_png)
        msk_path.write_bytes(mask_png)

        kwargs = dict(
            model=args.model,
            image=open(img_path, "rb"),
            mask=open(msk_path, "rb"),
            prompt=prompt,
            size=f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
            quality=args.quality,
            output_format="png",
            background="opaque",
        )
        if args.model == "gpt-image-1":
            kwargs["input_fidelity"] = args.input_fidelity

        try:
            result = client.images.edit(**kwargs)
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            sys.exit(1)

    image_bytes = base64.b64decode(result.data[0].b64_json)
    themed = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if args.debug:
        debug_dir = Path("out/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        themed.save(debug_dir / "5_themed_from_openai.png")

    out_path = Path(f"{args.output}_themed.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    themed.save(out_path)
    print(f"Saved: {out_path}")
    return out_path


def main():
    args = parse_args()
    script_dir = Path(__file__).parent.resolve()

    # Default output based on theme
    if args.output is None:
        args.output = "out/" + args.theme.lower().replace(" ", "_")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    maze_path = generate_maze(args, script_dir)
    print(f"Original maze saved: {maze_path}")

    themed_path = transform_maze(maze_path, args)
    print(f"Themed maze saved: {themed_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
