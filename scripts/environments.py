"""
Synthetic 2D environments for I-MPPI ablation study.

Each environment returns a ground-truth occupancy grid (numpy array)
where 1.0 = occupied, 0.0 = free. The grid is in NED convention
(x = North, y = East).
"""

import numpy as np


def make_corridor(
    width: int = 200,
    height: int = 200,
    resolution: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """
    20x20 m corridor environment with walls and interior obstacles.

    Returns:
        ground_truth: flat float32 array (width * height), 1.0 = occupied
        info: dict with grid metadata
    """
    gt = np.zeros((height, width), dtype=np.float32)

    # Boundary walls (2 cells thick)
    gt[:2, :] = 1.0
    gt[-2:, :] = 1.0
    gt[:, :2] = 1.0
    gt[:, -2:] = 1.0

    # Interior obstacles: three rectangular blocks
    # Block 1: at (5m, 3m), size 1x4 m
    _fill_rect(gt, 5.0, 3.0, 1.0, 4.0, resolution, width, height)
    # Block 2: at (10m, 8m), size 2x2 m
    _fill_rect(gt, 10.0, 8.0, 2.0, 2.0, resolution, width, height)
    # Block 3: at (15m, 5m), size 1x6 m
    _fill_rect(gt, 15.0, 5.0, 1.0, 6.0, resolution, width, height)

    origin_x = 0.0
    origin_y = 0.0
    info = {
        "width": width,
        "height": height,
        "resolution": resolution,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "name": "corridor",
        "extent_m": (width * resolution, height * resolution),
    }
    return gt.flatten(), info


def make_warehouse(
    width: int = 300,
    height: int = 300,
    resolution: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """
    30x30 m warehouse with scattered box obstacles.

    Returns:
        ground_truth: flat float32 array (width * height)
        info: dict with grid metadata
    """
    gt = np.zeros((height, width), dtype=np.float32)

    # Boundary walls
    gt[:2, :] = 1.0
    gt[-2:, :] = 1.0
    gt[:, :2] = 1.0
    gt[:, -2:] = 1.0

    # Scattered boxes
    boxes = [
        (5.0, 5.0, 2.0, 2.0),
        (5.0, 15.0, 2.0, 3.0),
        (12.0, 3.0, 3.0, 1.5),
        (12.0, 12.0, 2.0, 2.0),
        (12.0, 22.0, 1.5, 3.0),
        (20.0, 8.0, 2.5, 2.0),
        (20.0, 18.0, 2.0, 2.0),
        (25.0, 5.0, 1.5, 4.0),
        (25.0, 25.0, 2.0, 2.0),
    ]
    for bx, by, bw, bh in boxes:
        _fill_rect(gt, bx, by, bw, bh, resolution, width, height)

    info = {
        "width": width,
        "height": height,
        "resolution": resolution,
        "origin_x": 0.0,
        "origin_y": 0.0,
        "name": "warehouse",
        "extent_m": (width * resolution, height * resolution),
    }
    return gt.flatten(), info


def _fill_rect(
    grid: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    res: float,
    gw: int,
    gh: int,
) -> None:
    """Fill a rectangle in grid coordinates."""
    r0 = max(0, int(y / res))
    r1 = min(gh, int((y + h) / res))
    c0 = max(0, int(x / res))
    c1 = min(gw, int((x + w) / res))
    grid[r0:r1, c0:c1] = 1.0


def simulate_sensor(
    ground_truth: np.ndarray,
    belief: np.ndarray,
    uav_x: float,
    uav_y: float,
    info: dict,
    fov_rad: float = 1.57,
    max_range: float = 5.0,
    n_rays: int = 64,
) -> None:
    """
    Update belief grid by ray-casting from UAV position into ground truth.

    Cells within FOV: free → 0.1, occupied → 0.9.
    Operates on 2D arrays in-place.

    Args:
        ground_truth: 2D array (height, width), 1.0 = occupied
        belief: 2D array (height, width), 0.5 = unknown, modified in-place
        uav_x, uav_y: UAV position in world coords
        info: grid metadata dict
        fov_rad: total field-of-view angle (radians)
        max_range: sensor max range (meters)
        n_rays: number of rays to cast
    """
    res = info["resolution"]
    ox, oy = info["origin_x"], info["origin_y"]
    w, h = info["width"], info["height"]

    # UAV grid coords
    cx = (uav_x - ox) / res
    cy = (uav_y - oy) / res

    angles = np.linspace(-fov_rad / 2, fov_rad / 2, n_rays)
    step = res  # ray step = 1 cell

    for angle in angles:
        dx = np.cos(angle) * step / res
        dy = np.sin(angle) * step / res
        rx, ry = cx, cy
        dist = 0.0
        while dist < max_range:
            gi = int(round(rx))
            gj = int(round(ry))
            if gi < 0 or gi >= w or gj < 0 or gj >= h:
                break
            if ground_truth[gj, gi] >= 0.9:
                belief[gj, gi] = 0.9  # occupied
                break
            belief[gj, gi] = 0.1  # free
            rx += dx
            ry += dy
            dist += step
