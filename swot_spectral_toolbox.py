#!/usr/bin/env python3
"""
Reusable SWOT spectral-analysis helpers.

This module is meant to hold analysis functions that were originally defined
inside seasonality_check_ACC.ipynb, with notebook globals converted to explicit
function arguments.
"""

from __future__ import annotations

import datetime as _datetime
import glob
from pathlib import Path

import netCDF4 as nc
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import pandas as pd
except ImportError:  # pandas is only needed by build_spectral_dataframe.
    pd = None


_LAST_GOOD_STRIPS_LIST = None


def find_swot_file(data_dir, cycle, pass_number, product_prefix="SWOT_L2_LR_SSH_Basic"):
    """Return the first SWOT file matching one cycle/pass, or None."""
    pattern = (
        Path(data_dir)
        / f"{product_prefix}_{int(cycle):03d}_{int(pass_number):03d}_*.nc"
    )
    paths = sorted(glob.glob(str(pattern)))
    return paths[0] if paths else None


def available_cycles(
    data_dir,
    pass_number,
    min_cycle=1,
    max_cycle=49,
    product_prefix="SWOT_L2_LR_SSH_Basic",
):
    """List cycles that have an available netCDF file for this pass."""
    cycles = []
    for cycle in range(min_cycle, max_cycle + 1):
        if find_swot_file(data_dir, cycle, pass_number, product_prefix) is not None:
            cycles.append(cycle)
    return np.asarray(cycles, dtype=int)


def _as_index_array(indx):
    """Convert a range/slice/list of along-track indices to a NumPy index array."""
    if isinstance(indx, slice):
        start = 0 if indx.start is None else indx.start
        if indx.stop is None:
            raise ValueError("Open-ended slices are not supported for indx.")
        step = 1 if indx.step is None else indx.step
        return np.arange(start, indx.stop, step, dtype=int)
    return np.asarray(list(indx), dtype=int)


def _has_masked_values(array):
    """True if a masked array contains any masked value."""
    return np.ma.isMaskedArray(array) and bool(np.any(np.ma.getmaskarray(array)))


def select_strips(
    pass_number,
    indx,
    track_length=None,
    data_dir="/Volumes/My Passport for Mac/version_D",
    min_cycle=1,
    max_cycle=49,
    product_prefix="SWOT_L2_LR_SSH_Basic",
    side_start_indices=(5, 39),
    ssha_var="ssha_karin_2",
    ssha_qual_var="ssha_karin_2_qual",
    xover_var="height_cor_xover",
    xover_qual_var="height_cor_xover_qual",
    tide_var="internal_tide_hret",
    remove_cycle_outliers=True,
    outlier_factor=9.0,
    check_xover_qual=False,
    return_latlon_arrays=False,
):
    """
    Read good left/right SWOT strips for one pass across multiple cycles.

    Parameters
    ----------
    pass_number : int
        SWOT pass number.
    indx : range, list, ndarray, or slice
        Along-track indices to extract, usually range(start, stop) from
        find_swot_pass_segments/select_same_length_segments.
    track_length : int or None
        Number of along-track points. If None, inferred from indx.
    data_dir : str or Path
        Directory containing SWOT netCDF files.

    return_latlon_arrays : bool
        If False, return the notebook-compatible 2-D lat_strip/lon_strip as
        the first two outputs. If True, return 4-D lat_array/lon_array instead.

    Returns
    -------
    lat_strip, lon_strip, ssha_array, tides, time_array, good_strips_list,
    num_useful_strips, num_useful_cycles, cycle_list

    Array shapes are:
    lat_strip/lon_strip : [along_track, full_cross_track]
    ssha/tides : [cycle, side, along_track, selected_cross_track]
    time_array         : [cycle, side, along_track]
    good_strips_list   : [n_good_strips, 2], columns are [cycle_index, side]
    """
    global _LAST_GOOD_STRIPS_LIST

    index_array = _as_index_array(indx)
    if track_length is None:
        track_length = len(index_array)
    if len(index_array) != track_length:
        raise ValueError("track_length must match the number of along-track indices.")

    cycle_list = available_cycles(
        data_dir=data_dir,
        pass_number=pass_number,
        min_cycle=min_cycle,
        max_cycle=max_cycle,
        product_prefix=product_prefix,
    )
    num_useful_cycles = len(cycle_list)

    lat_array = np.full((num_useful_cycles, 2, track_length, swath_width), np.nan)
    lon_array = np.full((num_useful_cycles, 2, track_length, swath_width), np.nan)
    ssha_array = np.full((num_useful_cycles, 2, track_length, swath_width), np.nan)
    tides = np.full((num_useful_cycles, 2, track_length, swath_width), np.nan)
    time_array = np.full((num_useful_cycles, 2, track_length), np.nan)
    lat_strip_return = None
    lon_strip_return = None
    good_strips = []
    num_bad_strips = 0

    for cycle_index, cycle in enumerate(cycle_list):
        path = find_swot_file(data_dir, cycle, pass_number, product_prefix)
        if path is None:
            continue

        with nc.Dataset(path, "r") as data:
            ssha_strip = data[ssha_var][index_array, :]
            ssha_qual = data[ssha_qual_var][index_array, :]
            xover_cor = data[xover_var][index_array, :]
            xover_qual = data[xover_qual_var][index_array, :]
            internal_tide = data[tide_var][index_array, :]
            lat_strip = data["latitude"][index_array, :]
            lon_strip = data["longitude"][index_array, :]
            time_strip = data["time"][index_array]
            lat_strip_return = np.ma.filled(lat_strip, np.nan)
            lon_strip_return = np.ma.filled(lon_strip, np.nan)

            for side, start_index in enumerate(side_start_indices):
                stop_index = start_index + swath_width
                ssha_part = ssha_strip[:, start_index:stop_index]
                qual_part = ssha_qual[:, start_index:stop_index]
                xover_part = xover_cor[:, start_index:stop_index]
                xover_qual_part = xover_qual[:, start_index:stop_index]
                tide_part = internal_tide[:, start_index:stop_index]

                bad = (
                    _has_masked_values(ssha_part)
                    or _has_masked_values(xover_part)
                    or bool(np.any(np.asarray(qual_part) != 0))
                )
                if check_xover_qual:
                    bad = bad or bool(np.any(np.asarray(xover_qual_part) != 0))

                if bad:
                    num_bad_strips += 1
                    continue

                lat_array[cycle_index, side, :, :] = np.ma.filled(
                    lat_strip[:, start_index:stop_index], np.nan
                )
                lon_array[cycle_index, side, :, :] = np.ma.filled(
                    lon_strip[:, start_index:stop_index], np.nan
                )
                ssha_array[cycle_index, side, :, :] = (
                    np.ma.filled(ssha_part, np.nan)
                    + np.ma.filled(xover_part, np.nan)
                    - np.ma.filled(tide_part, np.nan)
                )
                tides[cycle_index, side, :, :] = np.ma.filled(tide_part, np.nan)
                time_array[cycle_index, side, :] = np.ma.filled(time_strip, np.nan)
                good_strips.append((cycle_index, side))
 
    good_strips_list = np.asarray(good_strips, dtype=int).reshape(-1, 2)

    if remove_cycle_outliers and num_useful_cycles > 0 and good_strips_list.size > 0:
        cycle_variances = np.full(num_useful_cycles, np.nan)
        for cycle_index in range(num_useful_cycles):
            cycle_variances[cycle_index] = np.nanvar(ssha_array[cycle_index, :, :, :])

        total_variance = np.nanvar(ssha_array)
        if np.isfinite(total_variance) and total_variance > 0:
            outlier_cycles = np.where(cycle_variances > outlier_factor * total_variance)[0]
            if outlier_cycles.size > 0:
                keep = ~np.isin(good_strips_list[:, 0], outlier_cycles)
                good_strips_list = good_strips_list[keep]

    good_set = set(map(tuple, good_strips_list.tolist()))
    for cycle_index in range(ssha_array.shape[0]):
        for side in range(ssha_array.shape[1]):
            if (cycle_index, side) not in good_set:
                ssha_array[cycle_index, side, :, :] = np.nan

    num_useful_strips = good_strips_list.shape[0]
    _LAST_GOOD_STRIPS_LIST = good_strips_list.copy()
    if return_latlon_arrays:
        lat_out = lat_array
        lon_out = lon_array
    else:
        if lat_strip_return is None:
            lat_strip_return = np.full((track_length, 0), np.nan)
            lon_strip_return = np.full((track_length, 0), np.nan)
        lat_out = lat_strip_return
        lon_out = lon_strip_return
        
    return (
        lat_out,
        lon_out,
        ssha_array,
        tides,
        time_array,
        good_strips_list,
        num_useful_strips,
        num_useful_cycles,
        cycle_list,
    )


def stack_good_strips(data_array, good_strips_list):
    """
    Convert [cycle, side, along_track, cross_track] to [strip, along, cross].
    """
    good_strips_list = np.asarray(good_strips_list, dtype=int)
    out = np.full(
        (good_strips_list.shape[0], data_array.shape[2], data_array.shape[3]),
        np.nan,
    )
    for strip, (cycle_index, side_index) in enumerate(good_strips_list):
        out[strip, :, :] = data_array[cycle_index, side_index, :, :]
    return out


def _anomaly_array(data_array, good_strips_list, anom=True):
    """Build [strip, along_track, cross_track] anomalies or raw strip data."""
    if anom:
        time_mean = np.nanmean(data_array, axis=0)
        anomaly = data_array - time_mean[None, :, :, :]
        spatial_mean = np.nanmean(anomaly, axis=(2, 3))
        anomaly = anomaly - spatial_mean[:, :, None, None]
        return stack_good_strips(anomaly, good_strips_list)
    return stack_good_strips(data_array, good_strips_list)


def wavenumber_1sided(track_length, dx=2e3):
    """Return rFFT one-sided wavenumbers in cycles per meter."""
    return np.fft.rfftfreq(track_length, d=dx)


def spectrum_compute_1sided(
    ssha_array,
    good_strips_list=None,
    track_length=None,
    anom=True,
    dx=2e3,
    k_min=3e-6,
    k_max=2e-5,
):
    """
    Compute one-sided along-track spectra for good SWOT strips.

    Preferred call:
        spectrum_compute_1sided(ssha_array, good_strips_list, anom=True)

    Backward-compatible notebook call:
        spectrum_compute_1sided(ssha_array, num_useful_strips, track_length)

    The backward-compatible form uses the most recent good_strips_list produced
    by select_strips.

    Returns
    -------
    spec_1side : ndarray
        Mean spectrum over strips and cross-track pixels.
    spec_1side_ts : ndarray
        Spectrum for each strip, averaged over cross-track pixels.
    data_array_anom : ndarray
        Data actually passed into the FFT, shape [strip, along, cross].
    spec_1side_ts_bp : ndarray
        Bandpassed spectrum for each strip.
    spec_1side_bp : ndarray
        Mean bandpassed spectrum.
    """
    global _LAST_GOOD_STRIPS_LIST

    if np.isscalar(good_strips_list):
        expected_num_strips = int(good_strips_list)
        if _LAST_GOOD_STRIPS_LIST is None:
            raise ValueError(
                "Old-style call detected, but no cached good_strips_list is "
                "available. Run select_strips first, or call "
                "spectrum_compute_1sided(ssha_array, good_strips_list)."
            )
        good_strips_list = _LAST_GOOD_STRIPS_LIST
        if good_strips_list.shape[0] != expected_num_strips:
            raise ValueError(
                "num_useful_strips does not match the most recent "
                "good_strips_list from select_strips."
            )
    elif good_strips_list is None:
        if _LAST_GOOD_STRIPS_LIST is None:
            raise ValueError(
                "good_strips_list is required unless select_strips was called "
                "immediately before spectrum_compute_1sided."
            )
        good_strips_list = _LAST_GOOD_STRIPS_LIST

    good_strips_list = np.asarray(good_strips_list, dtype=int)
    inferred_track_length = ssha_array.shape[2]
    if track_length is not None and int(track_length) != inferred_track_length:
        raise ValueError(
            f"track_length={track_length} does not match ssha_array.shape[2]="
            f"{inferred_track_length}."
        )
    track_length = inferred_track_length
    num_useful_strips = good_strips_list.shape[0]

    data_array_anom = _anomaly_array(ssha_array, good_strips_list, anom=anom)

    lambda_min = 1.0 / k_max
    lambda_max = 1.0 / k_min
    sigma_min = lambda_min / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_max = lambda_max / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_pixels_min = sigma_min / dx
    sigma_pixels_max = sigma_max / dx

    ssh_smoothed_low = gaussian_filter(
        data_array_anom, sigma=(0, sigma_pixels_max, sigma_pixels_max)
    )
    ssh_smoothed_high = gaussian_filter(
        data_array_anom, sigma=(0, sigma_pixels_min, sigma_pixels_min)
    )
    ssh_bandpassed = ssh_smoothed_high - ssh_smoothed_low

    window = np.sqrt(8.0 / 3.0) * np.sin(
        np.pi * np.arange(track_length) / track_length
    ) ** 2
    window_3d = np.broadcast_to(
        window.reshape(1, track_length, 1),
        (num_useful_strips, track_length, ssha_array.shape[3]),
    )

    ampsq_1side = _fft_power_1sided(data_array_anom * window_3d, dx=dx)
    ampsq_1side_bp = _fft_power_1sided(ssh_bandpassed * window_3d, dx=dx)

    spec_1side_ts = np.nanmean(ampsq_1side, axis=2)
    spec_1side = np.nanmean(ampsq_1side, axis=(0, 2))
    spec_1side_ts_bp = np.nanmean(ampsq_1side_bp, axis=2)
    spec_1side_bp = np.nanmean(ampsq_1side_bp, axis=(0, 2))

    return (
        spec_1side,
        spec_1side_ts,
        data_array_anom,
        spec_1side_ts_bp,
        spec_1side_bp,
    )


def _fft_power_1sided(windowed, dx=2e3):
    """One-sided rFFT power with the notebook's normalization."""
    track_length = windowed.shape[1]
    amp = np.fft.rfft(windowed, axis=1)
    ampsq = dx * (np.abs(amp)) ** 2 / track_length
    ampsq_1side = np.full(ampsq.shape, np.nan)
    ampsq_1side[:, 1:, :] = 2.0 * ampsq[:, 1:, :]
    ampsq_1side[:, 0, :] = ampsq[:, 0, :]
    return ampsq_1side


def output_time(num_useful_cycles, time_array, good_strips_list):
    """
    Convert SWOT seconds since 2000-01-01 to month/date labels for good strips.
    """
    timetot = np.full((num_useful_cycles,), np.nan)
    for cycle_index in range(time_array.shape[0]):
        meantime0 = np.nanmean(np.squeeze(time_array[cycle_index, 0, :]))
        meantime1 = np.nanmean(np.squeeze(time_array[cycle_index, 1, :]))
        if np.isnan(meantime0) and np.isnan(meantime1):
            timetot[cycle_index] = np.nan
        elif np.isnan(meantime0):
            timetot[cycle_index] = meantime1
        else:
            timetot[cycle_index] = meantime0

    epoch_time = _datetime.datetime(1970, 1, 1)
    swot_epoch = _datetime.datetime(2000, 1, 1, 0, 0)
    delta_seconds = (swot_epoch - epoch_time).total_seconds()
    unix_seconds = timetot + delta_seconds

    all_dates = []
    month = np.full((len(unix_seconds),), np.nan)
    for index, seconds in enumerate(unix_seconds):
        if np.isnan(seconds):
            all_dates.append("NaN")
        else:
            date = _datetime.datetime.utcfromtimestamp(seconds)
            all_dates.append(date.strftime("%y/%m/%d"))
            month[index] = date.month

    good_strips_list = np.asarray(good_strips_list, dtype=int)
    monthstrip = month[good_strips_list[:, 0]]
    dates = [all_dates[index] for index in good_strips_list[:, 0]]
    return monthstrip, dates, all_dates


# Backward-compatible alias for the notebook typo.
outputime = output_time

def ssh_field(
    ssha_array,
    track_length=None,
    good_strips_list=None,
    lst_swotime_tot=None,
    cycle_list=None,
    num_useful_cycles=None,
    full_swath_width=69,
    side_start_indices=(5, 39),
):
    """
    Rebuild selected left/right strips into a 69-pixel SWOT swath field.

    Parameters
    ----------
    ssha_array : ndarray
        SSH array from select_strips with shape
        [cycle, side, along_track, selected_cross_track].
    track_length : int or None
        Along-track length. If None, inferred from ssha_array.shape[2].
    good_strips_list : ndarray
        Good strip indices with columns [cycle_index, side].
    lst_swotime_tot : list
        Date labels returned by output_time/outputime. Entries equal to "NaN"
        are removed from the returned time series.
    cycle_list : ndarray or None
        Available cycle numbers. Kept for compatibility with the notebook code.
    num_useful_cycles : int or None
        Number of cycles. If None, inferred from ssha_array.shape[0].

    Returns
    -------
    ssha_mean2side : ndarray
        Time-mean SSH field on the 69-pixel swath grid, shape
        [along_track, full_swath_width].
    ssha_array2side_clean : ndarray
        SSH fields for valid-time cycles, shape
        [valid_cycle, along_track, full_swath_width].
    ssha_array2side_anom_clean : ndarray
        Anomaly SSH fields for valid-time cycles, same shape.
    lst_swotime_tot_clean : list
        Date labels after removing "NaN" entries.
    """
    if good_strips_list is None:
        raise ValueError("good_strips_list is required.")
    if lst_swotime_tot is None:
        raise ValueError("lst_swotime_tot is required.")

    good_strips_list = np.asarray(good_strips_list, dtype=int)
    if track_length is None:
        track_length = ssha_array.shape[2]
    if num_useful_cycles is None:
        num_useful_cycles = ssha_array.shape[0]

    swath_width = ssha_array.shape[3]
    if len(side_start_indices) != ssha_array.shape[1]:
        raise ValueError("side_start_indices must have one start index per side.")

    ssha_mean = np.nanmean(ssha_array, axis=0)
    ssha_mean2side = np.full((track_length, full_swath_width), np.nan)
    for side, start_index in enumerate(side_start_indices):
        stop_index = start_index + swath_width
        ssha_mean2side[:, start_index:stop_index] = ssha_mean[side, :, :]

    ssha_arrayanomly = ssha_array - ssha_mean[None, :, :, :]
    sshaspmean = np.nanmean(ssha_arrayanomly, axis=(2, 3))
    ssha_arrayanomlysp = ssha_arrayanomly - sshaspmean[:, :, None, None]

    ssha_array2side = np.full(
        (num_useful_cycles, track_length, full_swath_width), np.nan
    )
    ssha_array2side_anom = np.full(
        (num_useful_cycles, track_length, full_swath_width), np.nan
    )

    for cycle_index, side in good_strips_list:
        start_index = side_start_indices[side]
        stop_index = start_index + swath_width
        ssha_array2side[cycle_index, :, start_index:stop_index] = (
            ssha_array[cycle_index, side, :, :]
        )
        ssha_array2side_anom[cycle_index, :, start_index:stop_index] = (
            ssha_arrayanomlysp[cycle_index, side, :, :]
        )

    valid_mask = np.asarray([time != "NaN" for time in lst_swotime_tot], dtype=bool)
    if valid_mask.size != num_useful_cycles:
        raise ValueError(
            "lst_swotime_tot length must match num_useful_cycles/ssha_array.shape[0]."
        )

    lst_swotime_tot_clean = [
        lst_swotime_tot[index]
        for index in range(len(lst_swotime_tot))
        if valid_mask[index]
    ]
    ssha_array2side_clean = ssha_array2side[valid_mask]
    ssha_array2side_anom_clean = ssha_array2side_anom[valid_mask]

    # Keep this variable in the function so old notebooks that inspect it while
    # debugging can add it back easily; it is not returned to preserve API.
    if cycle_list is not None:
        _cycle_list_clean = np.asarray(cycle_list)[valid_mask]

    return (
        ssha_mean2side,
        ssha_array2side_clean,
        ssha_array2side_anom_clean,
        lst_swotime_tot_clean,
    )




def build_spectral_dataframe(
    selected,
    data_dir,
    min_cycle=1,
    max_cycle=49,
    dx=2e3,
    target_n=None,
):
    """
    Reproduce the notebook loop that builds one combined spectra DataFrame.
    """
    if pd is None:
        raise ImportError("pandas is required for build_spectral_dataframe.")

    combined = []
    for segment in selected:
        pass_number = int(segment["pass"])
        start = int(segment["start"])
        stop = int(segment["stop"])
        track_length = int(target_n or segment["n_points"])
        indx = range(start, stop)

        (
            _lat_array,
            _lon_array,
            ssha_array,
            tides,
            time_array,
            good_strips_list,
            _num_useful_strips,
            num_useful_cycles,
            _cycle_list,
        ) = select_strips(
            pass_number=pass_number,
            indx=indx,
            track_length=track_length,
            data_dir=data_dir,
            min_cycle=min_cycle,
            max_cycle=max_cycle,
        )

        spec_1side, spec_1side_ts, _data_array_anom, spec_1side_ts_bp, _ = (
            spectrum_compute_1sided(
                ssha_array, good_strips_list=good_strips_list, anom=True, dx=dx
            )
        )
        _spec_tide, spec_1side_tide_ts, _tide_data, _tide_ts_bp, _tide_bp = (
            spectrum_compute_1sided(
                tides, good_strips_list=good_strips_list, anom=False, dx=dx
            )
        )
        monthstrip, dates, _all_dates = output_time(
            num_useful_cycles, time_array, good_strips_list
        )

        combined.append(
            pd.DataFrame(
                {
                    "month": monthstrip,
                    "position": pass_number * np.ones(monthstrip.shape),
                    "dates": dates,
                    "spectrum": list(spec_1side_ts),
                    "spectrum_tides": list(spec_1side_tide_ts),
                    "spectrum_bp": list(spec_1side_ts_bp),
                    "mean_spectrum": [spec_1side] * len(monthstrip),
                }
            )
        )

    if not combined:
        return pd.DataFrame()
    return pd.concat(combined, ignore_index=True)


def balanced_fit(k, A, k0, s0, p):
    """Balanced-flow spectral model component."""
    return (A * (k / k0) ** (-p)) / (1.0 + (k / k0) ** s0)


def matern_spec_new(k, s, A_s):
    """Small-scale Matern spectral model component."""
    return 2.0 * np.pi * A_s / ((1.0 + (100e3 * k) ** 2) ** (s / 2.0))


def F_new(k, A, k0, s0, p, s, A_s):
    """Log of combined balanced + small-scale spectral model."""
    return np.log(balanced_fit(k, A, k0, s0, p) + matern_spec_new(k, s, A_s))
