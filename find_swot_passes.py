#!/usr/bin/env python3
"""
Find SWOT passes that intersect a longitude/latitude bounding box, and
optionally keep only same-length along-track segments.

Example
-------
python find_swot_passes.py \
  --data-dir Science_SWOT_L2_SSH_2.0 \
  --cycle 1 \
  --bbox -75 -65 30 40
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
from pathlib import Path

import netCDF4 as nc
import numpy as np


def normalize_lon(lon):
    """Normalize longitude(s) to [-180, 180)."""
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def lon_mask(lon, west, east):
    """
    Return mask for longitudes inside [west, east].

    Supports normal boxes, e.g. [-80, -60], and dateline-crossing boxes,
    e.g. [170, -150] or [170, 210].
    """
    lon = normalize_lon(lon)
    west = float(normalize_lon(west))
    east = float(normalize_lon(east))

    if west <= east:
        return (lon >= west) & (lon <= east)
    return (lon >= west) | (lon <= east)


def pass_intersects_bbox(nc_path, bbox, lon_name="longitude", lat_name="latitude"):
    """
    True if any valid SWOT lon/lat point falls inside bbox.

    bbox = [longitude_west, longitude_east, latitude_south, latitude_north]
    """
    west, east, south, north = bbox

    with nc.Dataset(nc_path, "r") as ds:
        lat = ds[lat_name][:]
        lon = ds[lon_name][:]

    if np.ma.isMaskedArray(lat):
        lat = lat.filled(np.nan)
    if np.ma.isMaskedArray(lon):
        lon = lon.filled(np.nan)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    mask = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= south)
        & (lat <= north)
        & lon_mask(lon, west, east)
    ) # only be true when a point satisfy these five conditions at the same time 
    return bool(np.any(mask)) # return true if there is any point satify these conditions 


def read_lon_lat(nc_path, lon_name="longitude", lat_name="latitude"):
    """Read lon/lat arrays from a SWOT netCDF file as float arrays with NaNs."""
    with nc.Dataset(nc_path, "r") as ds:
        lat = ds[lat_name][:]
        lon = ds[lon_name][:]

    if np.ma.isMaskedArray(lat):
        lat = lat.filled(np.nan)
    if np.ma.isMaskedArray(lon):
        lon = lon.filled(np.nan)
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def bbox_mask(lon, lat, bbox):
    """Boolean mask of points inside bbox."""
    west, east, south, north = bbox
    return (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= south)
        & (lat <= north)
        & lon_mask(lon, west, east)
    )


def along_track_bbox_mask(lon, lat, bbox, along_axis=0):
    """
    Collapse a point mask to an along-track mask.

    For 1-D lon/lat this returns one mask element per point. For 2-D SWOT
    arrays, along_axis=0 means rows are along-track and columns are cross-track.
    A row is kept when any cross-track pixel in that row is inside the bbox.
    (It returns a mask whose shape is along-track length,each point of this mask indicates if there are any track falls in the bbox)
    """
    mask = bbox_mask(lon, lat, bbox)
    if mask.ndim == 1:
        return mask
    if mask.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D lon/lat arrays, got shape {mask.shape}")

    cross_axis = 1 if along_axis == 0 else 0
    return np.any(mask, axis=cross_axis)


def contiguous_true_segments(mask):
    """Return [(start, stop), ...] for contiguous True runs; stop is exclusive."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def longest_segment(mask):
    """Return the longest contiguous True run as (start, stop), or None."""
    segments = contiguous_true_segments(mask)
    if not segments:
        return None
    return max(segments, key=lambda pair: pair[1] - pair[0])


def pass_bbox_segment(nc_path, bbox, along_axis=0, lon_name="longitude", lat_name="latitude"):
    """
    Return metadata for the longest along-track segment inside bbox.

    The returned start/stop indices refer to the along-track dimension and stop
    is exclusive, so data[start:stop] has n_points rows for along_axis=0.
    """
    lon, lat = read_lon_lat(nc_path, lon_name=lon_name, lat_name=lat_name)
    at_mask = along_track_bbox_mask(lon, lat, bbox, along_axis=along_axis)
    segment = longest_segment(at_mask)
    if segment is None:
        return None

    start, stop = segment
    return {
        "start": start,
        "stop": stop,
        "n_points": stop - start,
        "path": str(nc_path),
    }


def find_swot_passes(
    data_dir,
    cycle,
    bbox,
    pass_min=1,
    pass_max=584,
    product_prefix="SWOT_L2_LR_SSH_Basic",
):
    """
    Return pass numbers whose corresponding netCDF file intersects bbox.
    """
    data_dir = Path(data_dir)
    hits = []
    missing = []

    for pass_number in range(pass_min, pass_max + 1):
        pattern = (
            data_dir
            / f"{product_prefix}_{cycle:03d}_{pass_number:03d}_*.nc"
        )
        paths = sorted(glob.glob(str(pattern)))
        if not paths:
            missing.append(pass_number)
            continue

        if pass_intersects_bbox(paths[0], bbox):
            hits.append(pass_number)

    return hits, missing


def find_swot_pass_segments(
    data_dir,
    cycle,
    bbox,
    pass_min=1,
    pass_max=584,
    product_prefix="SWOT_L2_LR_SSH_Basic",
    along_axis=0,
):
    """
    Return dicts for passes intersecting bbox, including along-track segment.
    """
    data_dir = Path(data_dir)
    segments = []
    missing = []

    for pass_number in range(pass_min, pass_max + 1):
        pattern = data_dir / f"{product_prefix}_{cycle:03d}_{pass_number:03d}_*.nc"
        paths = sorted(glob.glob(str(pattern)))
        if not paths:
            missing.append(pass_number)
            continue

        segment = pass_bbox_segment(paths[0], bbox, along_axis=along_axis)
        if segment is not None:
            segment["pass"] = pass_number
            segments.append(segment)

    return segments, missing


def trim_segment_to_n(segment, target_n):
    """Center-trim one segment dict to target_n along-track points."""
    n_points = segment["n_points"]
    if n_points < target_n:
        return None

    extra = n_points - target_n
    start = segment["start"] + extra // 2
    stop = start + target_n
    trimmed = dict(segment)
    trimmed.update({"start": start, "stop": stop, "n_points": target_n})
    return trimmed


def select_same_length_segments(segments, method="exact", target_n=None):
    """
    Keep same-length along-track segments.

    method="exact": keep passes whose original n_points equals target_n. If
    target_n is None, use the most common n_points.

    method="trim": center-trim every segment to target_n. If target_n is None,
    use the minimum available n_points, so all intersecting passes are kept.
    """
    if not segments:
        return [], None

    lengths = [segment["n_points"] for segment in segments]
    if method == "exact":
        if target_n is None:
            target_n = Counter(lengths).most_common(1)[0][0]
        return [s for s in segments if s["n_points"] == target_n], target_n

    if method == "trim":
        if target_n is None:
            target_n = min(lengths)
        selected = [trim_segment_to_n(s, target_n) for s in segments]
        return [s for s in selected if s is not None], target_n

    raise ValueError("method must be 'exact' or 'trim'")


def main():
    parser = argparse.ArgumentParser(
        description="Find SWOT pass numbers intersecting a lon/lat bounding box."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing SWOT .nc files")
    parser.add_argument("--cycle", type=int, required=True, help="SWOT cycle number")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("LON_W", "LON_E", "LAT_S", "LAT_N"),
        help="Bounding box. Dateline crossing is supported.",
    )
    parser.add_argument("--pass-min", type=int, default=1)
    parser.add_argument("--pass-max", type=int, default=584)
    parser.add_argument(
        "--same-n",
        choices=("none", "exact", "trim"),
        default="none",
        help="Require same number of along-track points.",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=None,
        help="Target along-track point count for --same-n exact/trim.",
    )
    parser.add_argument(
        "--along-axis",
        type=int,
        choices=(0, 1),
        default=0,
        help="Along-track axis for 2-D lon/lat arrays.",
    )
    args = parser.parse_args()

    segments, missing = find_swot_pass_segments(
        data_dir=args.data_dir,
        cycle=args.cycle,
        bbox=args.bbox,
        pass_min=args.pass_min,
        pass_max=args.pass_max,
        along_axis=args.along_axis,
    )

    if args.same_n == "none":
        selected = segments
        target_n = None
    else:
        selected, target_n = select_same_length_segments(
            segments, method=args.same_n, target_n=args.target_n
        )

    print("passes:", [segment["pass"] for segment in selected])
    if target_n is not None:
        print("n_points:", target_n)
    print("segments:")
    for segment in selected:
        print(
            f"  pass {segment['pass']:03d}: "
            f"start={segment['start']} stop={segment['stop']} "
            f"n_points={segment['n_points']}"
        )
    if missing:
        print(f"missing files: {len(missing)} pass(es)")


if __name__ == "__main__":
    main()
