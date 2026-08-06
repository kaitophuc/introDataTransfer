"""Theory helpers for the OFDM BER simulations.

The default parameters mirror ``main_OFDM.py``:

* 16-QAM with unit average symbol energy
* normalized one-tap Rayleigh block fading, which behaves like AWGN with a
  random phase for this setup
* one LS pilot observation per data subcarrier
* repetition-3 majority decoding

The returned curve is an estimated upper bound, not an exact coded BER. By
default, it matches the normalized one-tap Sionna channel used by
``main_OFDM.py`` and applies a pairwise union bound for the repetition-3
majority vote. A true amplitude-Rayleigh average is still available by setting
``channel_type="rayleigh"``.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping

import numpy as np


def calculate_upper_bound(
    snr_dbs,
    *,
    bits_per_qam_symbol: int = 4,
    repetition_factor: int = 3,
    channel_type: str = "normalized_rayleigh",
    channel_estimation: str = "ls",
    pilot_observations: int = 1,
    pilot_energy: float = 1.0,
    quadrature_order: int = 80,
):
    """Calculate an estimated BER upper bound for the OFDM simulation.

    Parameters
    ----------
    snr_dbs:
        SNR values in dB. These are treated as symbol SNR values, matching
        ``noise_power_target = 1 / snr_linear_target`` in ``main_OFDM.py``.
    bits_per_qam_symbol:
        Number of bits per square-QAM symbol. ``4`` means 16-QAM.
    repetition_factor:
        Odd repetition-code factor used before QAM mapping. ``3`` matches the
        current majority-vote decoder.
    channel_type:
        ``"normalized_rayleigh"`` or ``"awgn"`` for the current
        normalized one-tap Sionna channel. Use ``"rayleigh"`` only for an
        unnormalized amplitude-Rayleigh channel.
    channel_estimation:
        ``"ls"`` applies a simple pilot-estimation penalty. ``"perfect"``
        skips this penalty.
    pilot_observations:
        Number of independent pilot observations used for one channel
        estimate. The current Sionna LS/nearest-neighbor setup effectively uses
        one pilot observation per data subcarrier.
    pilot_energy:
        Average pilot-symbol energy. Sionna's QAM/data normalization is unit
        energy, so the default is ``1.0``.
    quadrature_order:
        Number of Gauss-Laguerre points used for Rayleigh averaging.

    Returns
    -------
    numpy.ndarray
        BER upper-bound values with the same shape as ``snr_dbs``.
    """

    snr_dbs_array = _as_float_array(snr_dbs)
    snr_linear = np.power(10.0, snr_dbs_array / 10.0)
    effective_snr = _effective_symbol_snr(
        snr_linear,
        channel_estimation=channel_estimation,
        pilot_observations=pilot_observations,
        pilot_energy=pilot_energy,
    )

    normalized_channel_type = channel_type.strip().lower()
    awgn_like_types = {"awgn", "normalized", "normalized_rayleigh", "unit_rayleigh"}
    if normalized_channel_type in awgn_like_types:
        return _qam_repetition_pairwise_upper_bound(
            effective_snr,
            bits_per_qam_symbol=bits_per_qam_symbol,
            repetition_factor=repetition_factor,
        )

    if normalized_channel_type != "rayleigh":
        raise ValueError(
            "channel_type must be 'normalized_rayleigh', 'awgn', or 'rayleigh'"
        )

    if quadrature_order < 8:
        raise ValueError("quadrature_order must be at least 8")

    fading_power, weights = np.polynomial.laguerre.laggauss(quadrature_order)
    instantaneous_snr = effective_snr[..., np.newaxis] * fading_power
    info_bit_bound = _qam_repetition_pairwise_upper_bound(
        instantaneous_snr,
        bits_per_qam_symbol=bits_per_qam_symbol,
        repetition_factor=repetition_factor,
    )

    return np.sum(info_bit_bound * weights, axis=-1)


def plot_theory_vs_simulation(
    snr_dbs,
    simulated_bers,
    theory_bers=None,
    *,
    label: str = "Simulated BER",
    theory_label: str = "Theory upper bound",
    title: str = "BER: simulation vs. theory",
    bits_per_qam_symbol: int = 4,
    repetition_factor: int = 3,
    channel_type: str = "normalized_rayleigh",
    channel_estimation: str = "ls",
    pilot_observations: int = 1,
    pilot_energy: float = 1.0,
    quadrature_order: int = 80,
    data_save_path: str | None = None,
    ax=None,
    show: bool = True,
    save_path: str | None = None,
):
    """Visualize simulated BER values against the theory upper bound.

    ``simulated_bers`` may be either one BER sequence or a dictionary such as
    ``{"estimated channel": ber_values, "perfect channel": other_values}``.
    If ``theory_bers`` is omitted, this function calls ``calculate_upper_bound``
    with the supplied simulation parameters.

    ``save_path`` saves the plotted figure. ``data_save_path`` saves the
    numeric SNR/BER data to CSV.

    Returns ``(fig, ax, theory_bers)`` so callers can further edit or save the
    figure.
    """

    import matplotlib.pyplot as plt

    snr_dbs_array = _as_float_array(snr_dbs)
    simulation_series = _normalize_series(simulated_bers, default_label=label)

    if theory_bers is None:
        theory_bers = calculate_upper_bound(
            snr_dbs_array,
            bits_per_qam_symbol=bits_per_qam_symbol,
            repetition_factor=repetition_factor,
            channel_type=channel_type,
            channel_estimation=channel_estimation,
            pilot_observations=pilot_observations,
            pilot_energy=pilot_energy,
            quadrature_order=quadrature_order,
        )

    theory_series = _normalize_theory_series(
        theory_bers,
        default_label=theory_label,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
    else:
        fig = ax.figure

    for series_label, bers in simulation_series.items():
        bers_array = _as_float_array(bers)
        _validate_same_shape(snr_dbs_array, bers_array, name=series_label)
        ax.semilogy(
            snr_dbs_array,
            bers_array,
            marker="o",
            linewidth=1.8,
            label=series_label,
        )

    for series_label, bers in theory_series.items():
        bers_array = _as_float_array(bers)
        _validate_same_shape(snr_dbs_array, bers_array, name=series_label)
        ax.semilogy(
            snr_dbs_array,
            bers_array,
            linestyle="--",
            linewidth=2.0,
            label=series_label,
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if data_save_path is not None:
        save_ber_results(
            data_save_path,
            snr_dbs_array,
            simulated_bers=simulation_series,
            theory_bers=theory_bers,
            theory_label=theory_label,
        )

    if show:
        plt.show()

    return fig, ax, theory_bers


def save_ber_results(
    file_path: str,
    snr_dbs,
    simulated_bers=None,
    theory_bers=None,
    *,
    label: str = "simulated_ber",
    theory_label: str = "theory_upper_bound",
    bits_per_qam_symbol: int = 4,
    repetition_factor: int = 3,
    channel_type: str = "normalized_rayleigh",
    channel_estimation: str = "ls",
    pilot_observations: int = 1,
    pilot_energy: float = 1.0,
    quadrature_order: int = 80,
):
    """Save SNR, theoretical BER, and optional simulated BER values to CSV.

    ``simulated_bers`` can be one BER sequence or a dictionary of labeled BER
    sequences. If ``theory_bers`` is omitted, it is calculated from the supplied
    simulation parameters.

    Returns the path that was written.
    """

    snr_dbs_array = _as_float_array(snr_dbs)

    if theory_bers is None:
        theory_bers = calculate_upper_bound(
            snr_dbs_array,
            bits_per_qam_symbol=bits_per_qam_symbol,
            repetition_factor=repetition_factor,
            channel_type=channel_type,
            channel_estimation=channel_estimation,
            pilot_observations=pilot_observations,
            pilot_energy=pilot_energy,
            quadrature_order=quadrature_order,
        )

    columns = {"snr_db": snr_dbs_array}

    for series_label, bers in _normalize_theory_series(
        theory_bers,
        default_label=theory_label,
    ).items():
        bers_array = _as_float_array(bers)
        _validate_same_shape(snr_dbs_array, bers_array, name=series_label)
        columns[_safe_column_name(series_label)] = bers_array

    if simulated_bers is not None:
        for series_label, bers in _normalize_series(
            simulated_bers,
            default_label=label,
        ).items():
            bers_array = _as_float_array(bers)
            _validate_same_shape(snr_dbs_array, bers_array, name=series_label)
            columns[_safe_column_name(series_label)] = bers_array

    headers = list(columns)
    with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        for row_index in range(snr_dbs_array.size):
            writer.writerow([columns[header][row_index] for header in headers])

    return file_path


def _effective_symbol_snr(
    snr_linear,
    *,
    channel_estimation: str,
    pilot_observations: int,
    pilot_energy: float,
):
    if np.any(snr_linear <= 0):
        raise ValueError("SNR values must be positive after dB conversion")

    normalized_estimation = channel_estimation.strip().lower()
    if normalized_estimation in {"perfect", "none", "ideal"}:
        return snr_linear

    if normalized_estimation != "ls":
        raise ValueError("channel_estimation must be 'ls' or 'perfect'")

    if pilot_observations < 1:
        raise ValueError("pilot_observations must be at least 1")
    if pilot_energy <= 0:
        raise ValueError("pilot_energy must be positive")

    noise_variance = 1.0 / snr_linear
    channel_error_variance = noise_variance / (pilot_observations * pilot_energy)
    effective_noise_variance = noise_variance + channel_error_variance
    return 1.0 / effective_noise_variance


def _qam_repetition_pairwise_upper_bound(
    symbol_snr_linear,
    *,
    bits_per_qam_symbol: int,
    repetition_factor: int,
):
    if bits_per_qam_symbol != 4 or repetition_factor != 3:
        coded_bit_bound = _square_qam_bit_union_bound(
            symbol_snr_linear,
            bits_per_qam_symbol=bits_per_qam_symbol,
        )
        return _repetition_majority_error_bound(
            coded_bit_bound,
            repetition_factor=repetition_factor,
        )

    sign_error, magnitude_error, same_axis_joint_error = _qam16_axis_error_terms(
        symbol_snr_linear
    )

    bound = (
        same_axis_joint_error
        + sign_error * magnitude_error
        + 0.5 * (sign_error**2 + magnitude_error**2)
    )
    return np.clip(bound, 0.0, 0.5)


def _qam16_axis_error_terms(symbol_snr_linear):
    distance_to_nearest_threshold = np.sqrt(symbol_snr_linear / 5.0)
    distance_to_next_threshold = 3.0 * distance_to_nearest_threshold
    distance_to_far_threshold = 5.0 * distance_to_nearest_threshold

    q1 = _q_function(distance_to_nearest_threshold)
    q3 = _q_function(distance_to_next_threshold)
    q5 = _q_function(distance_to_far_threshold)

    sign_error = 0.5 * (q1 + q3)
    magnitude_error = q1 + 0.5 * q3

    # Joint probability that the sign bit and magnitude bit on the same 4-PAM
    # axis are both decoded incorrectly. This keeps the repetition bound aware
    # of the current flat bit layout in main_OFDM.py.
    same_axis_joint_error = np.maximum(q3 - 0.5 * q5, 0.0)

    return sign_error, magnitude_error, same_axis_joint_error


def _square_qam_bit_union_bound(symbol_snr_linear, *, bits_per_qam_symbol: int):
    if bits_per_qam_symbol < 1:
        raise ValueError("bits_per_qam_symbol must be positive")

    constellation_size = 2**bits_per_qam_symbol
    sqrt_constellation_size = int(math.isqrt(constellation_size))
    if sqrt_constellation_size**2 != constellation_size:
        raise ValueError("bits_per_qam_symbol must describe square QAM")

    coefficient = (
        4.0
        / bits_per_qam_symbol
        * (1.0 - 1.0 / sqrt_constellation_size)
    )
    distance_factor = 3.0 / (constellation_size - 1.0)
    q_argument = np.sqrt(distance_factor * symbol_snr_linear)
    bit_bound = coefficient * _q_function(q_argument)

    return np.clip(bit_bound, 0.0, 0.5)


def _repetition_majority_error_bound(coded_bit_error_bound, *, repetition_factor: int):
    if repetition_factor < 1:
        raise ValueError("repetition_factor must be at least 1")
    if repetition_factor % 2 == 0:
        raise ValueError("repetition_factor must be odd for majority decoding")

    p = np.clip(coded_bit_error_bound, 0.0, 0.5)
    needed_errors = repetition_factor // 2 + 1
    total = np.zeros_like(p, dtype=float)

    for error_count in range(needed_errors, repetition_factor + 1):
        total += (
            math.comb(repetition_factor, error_count)
            * np.power(p, error_count)
            * np.power(1.0 - p, repetition_factor - error_count)
        )

    return total


def _q_function(x):
    x_array = np.asarray(x, dtype=float)
    erfc_vectorized = np.vectorize(math.erfc, otypes=[float])
    return 0.5 * erfc_vectorized(x_array / math.sqrt(2.0))


def _as_float_array(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    return array


def _normalize_series(values, *, default_label: str):
    if isinstance(values, Mapping):
        return dict(values)
    return {default_label: values}


def _normalize_theory_series(values, *, default_label: str):
    if isinstance(values, Mapping):
        return dict(values)
    return {default_label: values}


def _validate_same_shape(snr_dbs, bers, *, name: str):
    if snr_dbs.shape != bers.shape:
        raise ValueError(
            f"{name!r} has shape {bers.shape}, but snr_dbs has shape "
            f"{snr_dbs.shape}"
        )


def _safe_column_name(label):
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


__all__ = [
    "calculate_upper_bound",
    "plot_theory_vs_simulation",
    "save_ber_results",
]
