"""
Synthetic Road Network Generator (Step 5A).
Generates an illustrative road network with synthetic segment dimensions,
traffic classifications, and image tile samples from results/tile_scores.csv.

LIMITATION: Image tiles are sampled from public vision benchmark datasets
(CRACK500, Road Crack, Pothole) and DO NOT represent real contiguous physical roads.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class RoadSegmentNetwork:
    """
    Encapsulates a synthetic road network of N segments.
    """
    def __init__(self, segments_df: pd.DataFrame, tile_pool_name: str = "all"):
        self.df = segments_df.copy()
        self.tile_pool_name = tile_pool_name
        self.n_segments = len(self.df)
        self.total_area_m2 = float(self.df["area_m2"].sum())

    @property
    def estimated_network_pci(self) -> float:
        """Area-weighted mean estimated sPCI."""
        return float(np.average(self.df["estimated_pci"], weights=self.df["area_m2"]))

    @property
    def true_network_pci(self) -> float:
        """Area-weighted mean true sPCI."""
        return float(np.average(self.df["true_pci"], weights=self.df["area_m2"]))

    def copy(self) -> "RoadSegmentNetwork":
        return RoadSegmentNetwork(self.df.copy(), tile_pool_name=self.tile_pool_name)


def generate_synthetic_network(
    tile_scores_path: str = "results/tile_scores.csv",
    tile_pool: str = "all",
    n_segments: int = 200,
    k_tiles: int = 6,
    seed: int = 42,
) -> RoadSegmentNetwork:
    """
    Generates a synthetic network of N road segments.

    Parameters:
    -----------
    tile_scores_path : Path to tile_scores.csv
    tile_pool : 'all' (all sources) or 'cracks' (crack500 + road_crack only)
    n_segments : Number of segments (default 200)
    k_tiles : Number of tiles sampled per segment (default 6)
    seed : Deterministic random seed

    Returns:
    --------
    RoadSegmentNetwork object containing segment attributes and condition ratings.
    """
    tile_path = Path(tile_scores_path)
    if not tile_path.exists():
        raise FileNotFoundError(
            f"Tile scores file not found at {tile_path}. "
            "Please compute results/tile_scores.csv before initializing network."
        )

    tiles_df = pd.read_csv(tile_path)
    if tile_pool == "cracks":
        filtered_tiles = tiles_df[tiles_df["source"].isin(["crack500", "road_crack"])].copy()
    elif tile_pool == "all":
        filtered_tiles = tiles_df.copy()
    else:
        raise ValueError(f"Unknown tile_pool: '{tile_pool}'. Expected 'all' or 'cracks'.")

    if len(filtered_tiles) == 0:
        raise ValueError(f"No tiles available for tile_pool '{tile_pool}'.")

    rng = np.random.RandomState(seed)

    segments = []
    # Traffic classes: low 50%, medium 35%, high 15% with weights 1, 2, 3
    traffic_classes = ["low", "medium", "high"]
    traffic_probs = [0.50, 0.35, 0.15]
    traffic_weight_map = {"low": 1.0, "medium": 2.0, "high": 3.0}

    tile_pool_indices = np.arange(len(filtered_tiles))

    for i in range(n_segments):
        seg_id = f"seg_{i:03d}"
        length_m = float(rng.uniform(100.0, 600.0))
        width_m = float(rng.uniform(6.0, 14.0))
        area_m2 = length_m * width_m

        t_class = rng.choice(traffic_classes, p=traffic_probs)
        t_weight = traffic_weight_map[t_class]

        # Sample k tiles with replacement
        sampled_indices = rng.choice(tile_pool_indices, size=k_tiles, replace=True)
        sampled_tiles = filtered_tiles.iloc[sampled_indices]

        # ESTIMATED condition = mean predicted sPCI of its k tiles
        est_pci = float(sampled_tiles["predicted_spci"].mean())

        # TRUE condition = mean GT-based sPCI of its k tiles
        true_pci = float(sampled_tiles["gt_spci"].mean())

        segments.append({
            "segment_id": seg_id,
            "length_m": length_m,
            "width_m": width_m,
            "area_m2": area_m2,
            "traffic_class": t_class,
            "traffic_weight": t_weight,
            "estimated_pci": est_pci,
            "true_pci": true_pci,
            "tile_paths": sampled_tiles["image_path"].tolist(),
        })

    df = pd.DataFrame(segments)
    return RoadSegmentNetwork(df, tile_pool_name=tile_pool)
