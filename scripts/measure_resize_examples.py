import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageOps
import numpy as np
import cv2
from src.pipeline import RoadPCIPipeline

def run_comparison():
    pipeline = RoadPCIPipeline(device="cpu")
    examples_dir = Path("app_assets/examples")
    
    files = [
        "crack500_example_01.jpg",
        "crack500_example_02.jpg",
        "road_crack_example_01.jpg",
        "road_crack_example_02.jpg",
    ]
    
    # Header
    print(f"{'Filename':<28} | {'Raw PIL Size (WxH)':<19} | {'Old Size (WxH)':<16} | {'New Size (WxH)':<16} | {'Old sPCI':<8} | {'New sPCI':<8} | {'Diff':<8}")
    print("-" * 115)
    
    for fname in files:
        fpath = examples_dir / fname
        
        # 1. Real raw size from PIL with EXIF orientation
        with Image.open(fpath) as pil_raw:
            pil_trans = ImageOps.exif_transpose(pil_raw)
            raw_w, raw_h = pil_trans.size
            img_rgb = np.array(pil_trans.convert("RGB"))
            
        h, w = img_rgb.shape[:2]
        short_side = min(h, w)
        long_side = max(h, w)
        
        # Old Rule (original upload resize rule applied to uploaded images):
        # scale short side to 720, cap long side at 1280
        scale_old = 720.0 / short_side
        new_short_old = 720
        new_long_old = int(round(long_side * scale_old))
        if new_long_old > 1280:
            scale_cap = 1280.0 / new_long_old
            new_long_old = 1280
            new_short_old = int(round(720.0 * scale_cap))
        target_h_old, target_w_old = (new_short_old, new_long_old) if h <= w else (new_long_old, new_short_old)
        img_old = cv2.resize(img_rgb, (target_w_old, target_h_old), interpolation=cv2.INTER_LINEAR)
        old_size_str = f"{target_w_old}x{target_h_old}"
        
        # New Unified Rule:
        # If short_side > 720, downscale short side to 720, long side capped at 1280.
        # Otherwise keep native resolution (never upscale).
        if short_side > 720:
            scale_new = 720.0 / short_side
            new_short_new = 720
            new_long_new = int(round(long_side * scale_new))
            if new_long_new > 1280:
                scale_cap = 1280.0 / new_long_new
                new_long_new = 1280
                new_short_new = int(round(720.0 * scale_cap))
            target_h_new, target_w_new = (new_short_new, new_long_new) if h <= w else (new_long_new, new_short_new)
            img_new = cv2.resize(img_rgb, (target_w_new, target_h_new), interpolation=cv2.INTER_LINEAR)
        else:
            target_h_new, target_w_new = h, w
            img_new = img_rgb.copy()
        new_size_str = f"{target_w_new}x{target_h_new}"
        
        # Run inference through pipeline
        res_old = pipeline.infer(img_old, use_tta=True)
        res_new = pipeline.infer(img_new, use_tta=True)
        
        spci_old = res_old["spci"]
        spci_new = res_new["spci"]
        diff = abs(spci_new - spci_old)
        
        raw_size_str = f"{raw_w}x{raw_h}"
        print(f"{fname:<28} | {raw_size_str:<19} | {old_size_str:<16} | {new_size_str:<16} | {spci_old:<8.2f} | {spci_new:<8.2f} | {diff:<8.4f}")

if __name__ == "__main__":
    run_comparison()
