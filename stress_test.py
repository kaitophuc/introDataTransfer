"""Stress-test a frozen neural OFDM receiver across channel environments.

The script strictly loads checkpoints/full_grid_receiver.pt and performs
evaluation only. It never trains the neural receiver and never writes a model
checkpoint.

Default staged experiment
-------------------------
The reference environment is TDL-A, 300 ns RMS delay spread, and 10 m/s fixed
speed. Three focused sweeps independently vary:

* delay spread: 100, 300, and 1000 ns;
* fixed speed: 0, 10, and 30 m/s;
* TDL profile: A, B, C, D, and E.

Every environment is evaluated over the configured SNR points and seeds for
both the frozen neural receiver and the classical LS/LMMSE baseline. Use
--sweep-mode full for the complete Cartesian product.

Channel-window limitation
-------------------------
To match the current project configuration, the discrete time-channel window
is explicitly based on a 3 microsecond maximum delay spread and the default
cyclic prefix remains 16 samples. At large RMS delay spreads, late energy in
some standardized TDL profiles can therefore be truncated. This limitation is
recorded in the run manifest and on every chart.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from sionna.phy import config
from sionna.phy.channel import TimeChannel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.ofdm import OFDMDemodulator

from main_OFDM import build_ofdm_system
from neural_receiver import OFDMNeuralReceiverTrainer


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = "results/stress_test"
CHANNEL_WINDOW_SECONDS = 3e-6
SCHEMA_VERSION = 2

NEURAL_RECEIVER = "neural_checkpoint"
CLASSICAL_RECEIVER = "classical_ls_lmmse"
RECEIVERS = (NEURAL_RECEIVER, CLASSICAL_RECEIVER)

RAW_FILENAME = "stress_test_raw.csv"
SUMMARY_FILENAME = "stress_test_summary.csv"
MANIFEST_FILENAME = "stress_test_manifest.json"
DELAY_CHART_FILENAME = "delay_spread_comparison.png"
MOBILITY_CHART_FILENAME = "mobility_comparison.png"
CHANNEL_MODEL_CHART_FILENAME = "channel_model_comparison.png"

GENERATED_FILENAMES = (
    RAW_FILENAME,
    SUMMARY_FILENAME,
    MANIFEST_FILENAME,
    DELAY_CHART_FILENAME,
    MOBILITY_CHART_FILENAME,
    CHANNEL_MODEL_CHART_FILENAME,
)

RAW_COLUMNS = (
    "receiver",
    "environment_id",
    "sweep_tags",
    "checkpoint_path",
    "checkpoint_sha256",
    "channel_model",
    "delay_spread_ns",
    "speed_mps",
    "snr_db",
    "evaluation_seed",
    "target_coded_bits",
    "actual_coded_bits",
    "actual_info_bits",
    "total_frames",
    "batch_size",
    "ber",
    "fer",
    "completed_at_utc",
)

SUMMARY_COLUMNS = (
    "receiver",
    "channel_model",
    "delay_spread_ns",
    "speed_mps",
    "snr_db",
    "num_seeds",
    "target_coded_bits",
    "actual_coded_bits_per_seed",
    "actual_info_bits_per_seed",
    "mean_ber",
    "std_ber",
    "ci95_half_width_ber",
    "mean_fer",
    "std_fer",
    "ci95_half_width_fer",
)


@dataclass(frozen=True, order=True)
class Environment:
    """One deterministic channel configuration in the stress-test matrix."""

    channel_model: str
    delay_spread_ns: float
    speed_mps: float

    @property
    def key(self) -> tuple[str, float, float]:
        return (
            self.channel_model,
            float(self.delay_spread_ns),
            float(self.speed_mps),
        )

    @property
    def identifier(self) -> str:
        return (
            f"tdl_{self.channel_model.lower()}"
            f"_delay_{number_tag(self.delay_spread_ns)}ns"
            f"_speed_{number_tag(self.speed_mps)}mps"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one frozen neural receiver checkpoint across delay "
            "spread, fixed mobility, SNR, and TDL-profile variations. "
            "No training is performed."
        )
    )

    evaluation = parser.add_argument_group("evaluation matrix")
    evaluation.add_argument(
        "--checkpoint",
        default="checkpoints/full_grid_receiver.pt",
        help="Frozen neural receiver state_dict to evaluate.",
    )
    evaluation.add_argument(
        "--sweep-mode",
        choices=("staged", "full"),
        default="staged",
        help=(
            "staged varies one category around the reference environment; "
            "full evaluates the Cartesian product."
        ),
    )
    evaluation.add_argument(
        "--delay-spreads-ns",
        type=float,
        nargs="+",
        default=[100.0, 300.0, 1000.0],
    )
    evaluation.add_argument(
        "--speeds-mps",
        type=float,
        nargs="+",
        default=[0.0, 10.0, 30.0],
        help="Fixed speeds; each TDL model is built with min_speed=max_speed.",
    )
    evaluation.add_argument(
        "--channel-models",
        choices=list("ABCDE"),
        nargs="+",
        default=list("ABCDE"),
    )
    evaluation.add_argument(
        "--snr-dbs",
        type=float,
        nargs="+",
        default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    )
    evaluation.add_argument(
        "--evaluation-seeds",
        type=int,
        nargs="+",
        default=[1000, 1001, 1002],
    )
    evaluation.add_argument("--target-coded-bits", type=int, default=10_000_000)
    evaluation.add_argument("--batch-size", type=int, default=100)

    reference = parser.add_argument_group("staged-sweep reference")
    reference.add_argument("--reference-delay-spread-ns", type=float, default=300.0)
    reference.add_argument("--reference-speed-mps", type=float, default=10.0)
    reference.add_argument(
        "--reference-channel-model",
        choices=list("ABCDE"),
        default="A",
    )

    system = parser.add_argument_group("OFDM system")
    system.add_argument("--num-subcarriers", type=int, default=64)
    system.add_argument("--num-ofdm-symbols", type=int, default=14)
    system.add_argument("--bits-per-qam-symbol", type=int, default=4)
    system.add_argument("--code-rate", type=float, default=0.5)
    system.add_argument("--cp-len", type=int, default=16)

    output = parser.add_argument_group("execution and output")
    output.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    output.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume compatible per-receiver results already in output-dir.",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete this script's known output files before starting.",
    )
    output.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Evaluation-only plumbing check: 100K coded bits, one seed, and "
            "SNR values 8/12/16 dB."
        ),
    )

    args = parser.parse_args(argv)
    apply_quick_settings(args)
    normalize_and_validate_args(args)
    return args


def apply_quick_settings(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.target_coded_bits = min(args.target_coded_bits, 100_000)
    args.evaluation_seeds = args.evaluation_seeds[:1]
    args.snr_dbs = [8.0, 12.0, 16.0]
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = f"{DEFAULT_OUTPUT_DIR}/quick"


def normalize_and_validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "target_coded_bits",
        "batch_size",
        "num_subcarriers",
        "num_ofdm_symbols",
        "bits_per_qam_symbol",
        "cp_len",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")

    args.delay_spreads_ns = unique_preserving_order(args.delay_spreads_ns)
    args.speeds_mps = unique_preserving_order(args.speeds_mps)
    args.channel_models = unique_preserving_order(args.channel_models)
    args.snr_dbs = unique_preserving_order(args.snr_dbs)
    args.evaluation_seeds = unique_preserving_order(args.evaluation_seeds)

    if not 0.0 < args.code_rate < 1.0:
        raise ValueError("--code-rate must lie strictly between 0 and 1")
    if not args.delay_spreads_ns or any(
        delay <= 0.0 for delay in args.delay_spreads_ns
    ):
        raise ValueError("all delay spreads must be positive")
    if not args.speeds_mps or any(speed < 0.0 for speed in args.speeds_mps):
        raise ValueError("all fixed speeds must be non-negative")
    if not args.channel_models:
        raise ValueError("at least one TDL channel model is required")
    if not args.snr_dbs:
        raise ValueError("at least one SNR value is required")
    if not args.evaluation_seeds:
        raise ValueError("at least one evaluation seed is required")
    if args.reference_delay_spread_ns not in args.delay_spreads_ns:
        raise ValueError(
            "--reference-delay-spread-ns must be included in --delay-spreads-ns"
        )
    if args.reference_speed_mps not in args.speeds_mps:
        raise ValueError("--reference-speed-mps must be included in --speeds-mps")
    if args.reference_channel_model not in args.channel_models:
        raise ValueError(
            "--reference-channel-model must be included in --channel-models"
        )


def unique_preserving_order(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def number_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "minus_").replace(".", "p")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ns_to_seconds(delay_ns: float) -> float:
    return delay_ns * 1e-9


def reference_environment(args: argparse.Namespace) -> Environment:
    return Environment(
        channel_model=args.reference_channel_model,
        delay_spread_ns=args.reference_delay_spread_ns,
        speed_mps=args.reference_speed_mps,
    )


def build_environment_matrix(args: argparse.Namespace) -> list[Environment]:
    if args.sweep_mode == "full":
        return [
            Environment(model, delay, speed)
            for model, delay, speed in itertools.product(
                args.channel_models,
                args.delay_spreads_ns,
                args.speeds_mps,
            )
        ]

    reference = reference_environment(args)
    environments: dict[tuple[str, float, float], Environment] = {}

    for delay in args.delay_spreads_ns:
        environment = Environment(
            reference.channel_model,
            delay,
            reference.speed_mps,
        )
        environments[environment.key] = environment

    for speed in args.speeds_mps:
        environment = Environment(
            reference.channel_model,
            reference.delay_spread_ns,
            speed,
        )
        environments[environment.key] = environment

    for model in args.channel_models:
        environment = Environment(
            model,
            reference.delay_spread_ns,
            reference.speed_mps,
        )
        environments[environment.key] = environment

    return list(environments.values())


def sweep_tags(environment: Environment, args: argparse.Namespace) -> str:
    reference = reference_environment(args)
    tags: list[str] = []
    if (
        environment.channel_model == reference.channel_model
        and environment.speed_mps == reference.speed_mps
    ):
        tags.append("delay_spread")
    if (
        environment.channel_model == reference.channel_model
        and environment.delay_spread_ns == reference.delay_spread_ns
    ):
        tags.append("mobility")
    if (
        environment.delay_spread_ns == reference.delay_spread_ns
        and environment.speed_mps == reference.speed_mps
    ):
        tags.append("channel_model")
    return ";".join(tags) if tags else "full_grid_only"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_weights(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"neural receiver checkpoint not found: {checkpoint_path}"
        )

    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            f"checkpoint does not contain a non-empty state_dict: {checkpoint_path}"
        )
    if not all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise ValueError(f"checkpoint state_dict contains non-tensors: {checkpoint_path}")

    return payload


def build_evaluation_system(
    args: argparse.Namespace,
    environment: Environment,
) -> dict[str, Any]:
    """Build the existing 2-stream system for one fixed environment."""
    delay_spread_seconds = ns_to_seconds(environment.delay_spread_ns)

    system = build_ofdm_system(
        num_subcarriers=args.num_subcarriers,
        num_ofdm_symbols=args.num_ofdm_symbols,
        bits_per_qam_symbol=args.bits_per_qam_symbol,
        code_rate=args.code_rate,
        cp_len=args.cp_len,
        min_speed=environment.speed_mps,
        max_speed=environment.speed_mps,
        delay_spread=delay_spread_seconds,
        num_tx=1,
        num_streams_per_tx=2,
        num_rx_ant=2,
        num_tx_ant=2,
        device=config.device,
    )

    # build_ofdm_system currently creates TDL-A internally. Replace only that
    # channel and its timing-aware demodulator so all profiles A-E are supported
    # without changing any shared project file.
    channel_model = TDL(
        model=environment.channel_model,
        delay_spread=delay_spread_seconds,
        carrier_frequency=3.5e9,
        min_speed=environment.speed_mps,
        max_speed=environment.speed_mps,
        num_rx_ant=2,
        num_tx_ant=2,
        precision="single",
        device=config.device,
    )
    time_channel = TimeChannel(
        channel_model=channel_model,
        bandwidth=15e3 * args.num_subcarriers,
        num_time_samples=args.num_ofdm_symbols
        * (args.num_subcarriers + args.cp_len),
        maximum_delay_spread=CHANNEL_WINDOW_SECONDS,
        normalize_channel=True,
        return_channel=False,
        precision="single",
        device=config.device,
    )
    ofdm_demodulator = OFDMDemodulator(
        fft_size=args.num_subcarriers,
        l_min=time_channel.l_min,
        cyclic_prefix_length=args.cp_len,
        precision="single",
        device=config.device,
    )

    system["sionna_time_channel"] = time_channel
    system["ofdm_demodulator"] = ofdm_demodulator
    return system


def make_evaluation_trainer(
    args: argparse.Namespace,
    environment: Environment,
    frozen_weights: Mapping[str, torch.Tensor],
) -> OFDMNeuralReceiverTrainer:
    system = build_evaluation_system(args, environment)
    trainer = OFDMNeuralReceiverTrainer(
        system=system,
        bits_per_qam_symbol=args.bits_per_qam_symbol,
        device=config.device,
    )
    trainer.full_grid_receiver.load_state_dict(frozen_weights, strict=True)
    trainer.full_grid_receiver.eval()
    return trainer


def assert_weights_unchanged(
    trainer: OFDMNeuralReceiverTrainer,
    frozen_weights: Mapping[str, torch.Tensor],
) -> None:
    actual_state = trainer.full_grid_receiver.state_dict()
    if set(actual_state) != set(frozen_weights):
        raise RuntimeError("neural receiver state_dict keys changed during evaluation")

    for name, expected in frozen_weights.items():
        actual = actual_state[name].detach().cpu()
        if not torch.equal(actual, expected.detach().cpu()):
            raise RuntimeError(
                f"frozen neural receiver parameter changed during evaluation: {name}"
            )


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_run_signature(
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    environments: Sequence[Environment],
) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "sweep_mode": args.sweep_mode,
        "delay_spreads_ns": args.delay_spreads_ns,
        "speeds_mps": args.speeds_mps,
        "channel_models": args.channel_models,
        "snr_dbs": args.snr_dbs,
        "evaluation_seeds": args.evaluation_seeds,
        "target_coded_bits": args.target_coded_bits,
        "batch_size": args.batch_size,
        "reference_environment": asdict(reference_environment(args)),
        "environments": [asdict(environment) for environment in environments],
        "ofdm_system": {
            "num_subcarriers": args.num_subcarriers,
            "num_ofdm_symbols": args.num_ofdm_symbols,
            "bits_per_qam_symbol": args.bits_per_qam_symbol,
            "code_rate": args.code_rate,
            "cp_len": args.cp_len,
            "num_tx": 1,
            "num_streams_per_tx": 2,
            "num_rx_ant": 2,
            "num_tx_ant": 2,
        },
        "channel_window_seconds": CHANNEL_WINDOW_SECONDS,
        "quick": args.quick,
        "torch_version": torch.__version__,
        "sionna_version": package_version("sionna"),
    }


def build_manifest(
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    environments: Sequence[Environment],
    run_signature: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    expected_rows = (
        len(environments)
        * len(args.snr_dbs)
        * len(args.evaluation_seeds)
        * len(RECEIVERS)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "training_performed": False,
        "created_at_utc": utc_now(),
        "last_started_at_utc": utc_now(),
        "complete": False,
        "completed_rows": 0,
        "expected_rows": expected_rows,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "run_signature": dict(run_signature),
        "environment_count": len(environments),
        "environment_matrix": [
            {
                **asdict(environment),
                "environment_id": environment.identifier,
                "sweep_tags": sweep_tags(environment, args),
            }
            for environment in environments
        ],
        "channel_window_limitation": {
            "maximum_delay_spread_seconds": CHANNEL_WINDOW_SECONDS,
            "cyclic_prefix_samples": args.cp_len,
            "note": (
                "Matches the current project setup. At large RMS delay spreads, "
                "late energy in some TDL profiles may be truncated."
            ),
        },
        "outputs": {
            "directory": str(output_dir),
            "raw_csv": RAW_FILENAME,
            "summary_csv": SUMMARY_FILENAME,
            "delay_chart": DELAY_CHART_FILENAME,
            "mobility_chart": MOBILITY_CHART_FILENAME,
            "channel_model_chart": CHANNEL_MODEL_CHART_FILENAME,
        },
    }


def prepare_output_directory(
    args: argparse.Namespace,
    output_dir: Path,
    proposed_manifest: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / filename for filename in GENERATED_FILENAMES]

    if args.overwrite:
        for path in paths:
            if path.exists():
                path.unlink()

    manifest_path = output_dir / MANIFEST_FILENAME
    raw_path = output_dir / RAW_FILENAME

    if not args.resume and any(path.exists() for path in paths):
        raise RuntimeError(
            "stress-test outputs already exist and --no-resume was selected; "
            "use a new --output-dir or explicitly pass --overwrite"
        )

    if raw_path.exists() and not manifest_path.exists():
        raise RuntimeError(
            f"cannot safely resume {raw_path} without {manifest_path}; "
            "use a new --output-dir or explicitly pass --overwrite"
        )

    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as manifest_file:
            existing_manifest = json.load(manifest_file)
        validate_resume_manifest(existing_manifest, proposed_manifest)
        existing_manifest["last_started_at_utc"] = utc_now()
        existing_manifest["complete"] = False
        write_json_atomic(manifest_path, existing_manifest)
        return existing_manifest

    write_json_atomic(manifest_path, proposed_manifest)
    return proposed_manifest


def validate_resume_manifest(
    existing: Mapping[str, Any],
    proposed: Mapping[str, Any],
) -> None:
    if existing.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            "existing stress-test manifest uses an incompatible schema; "
            "use a new --output-dir or explicitly pass --overwrite"
        )
    if existing.get("training_performed") is not False:
        raise RuntimeError("existing manifest is not an evaluation-only run")
    if existing.get("run_signature") != proposed.get("run_signature"):
        raise RuntimeError(
            "existing stress-test outputs do not match the checkpoint or "
            "evaluation configuration; use a new --output-dir or explicitly "
            "pass --overwrite"
        )


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def initialize_raw_csv(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        output_file.flush()
        os.fsync(output_file.fileno())


def append_raw_row(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RAW_COLUMNS)
        writer.writerow(row)
        output_file.flush()
        os.fsync(output_file.fileno())


def load_raw_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise RuntimeError(
                f"existing raw CSV has an incompatible schema: {path}"
            )
        return list(reader)


def result_key(
    receiver: str,
    environment: Environment,
    snr_db: float,
    evaluation_seed: int,
) -> tuple[str, str, float, float, float, int]:
    return (
        receiver,
        environment.channel_model,
        float(environment.delay_spread_ns),
        float(environment.speed_mps),
        float(snr_db),
        int(evaluation_seed),
    )


def raw_row_key(row: Mapping[str, Any]) -> tuple[str, str, float, float, float, int]:
    channel_model = str(row["channel_model"])
    if channel_model.startswith("TDL-"):
        channel_model = channel_model.removeprefix("TDL-")
    return (
        str(row["receiver"]),
        channel_model,
        float(row["delay_spread_ns"]),
        float(row["speed_mps"]),
        float(row["snr_db"]),
        int(row["evaluation_seed"]),
    )


def expected_result_keys(
    args: argparse.Namespace,
    environments: Sequence[Environment],
) -> set[tuple[str, str, float, float, float, int]]:
    return {
        result_key(receiver, environment, snr_db, seed)
        for environment in environments
        for snr_db in args.snr_dbs
        for seed in args.evaluation_seeds
        for receiver in RECEIVERS
    }


def validate_existing_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_keys: set[tuple[str, str, float, float, float, int]],
) -> set[tuple[str, str, float, float, float, int]]:
    completed: set[tuple[str, str, float, float, float, int]] = set()
    for row in rows:
        key = raw_row_key(row)
        if key not in expected_keys:
            raise RuntimeError(
                "raw CSV contains a result outside the current run matrix; "
                "use a new --output-dir or explicitly pass --overwrite"
            )
        if key in completed:
            raise RuntimeError(f"raw CSV contains a duplicate result key: {key}")
        completed.add(key)
    return completed


def result_row(
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    receiver: str,
    environment: Environment,
    snr_db: float,
    evaluation_seed: int,
    trainer: OFDMNeuralReceiverTrainer,
    total_frames: int,
    ber: float,
    fer: float,
) -> dict[str, Any]:
    actual_coded_bits = total_frames * trainer.num_coded_bits_per_frame
    actual_info_bits = total_frames * trainer.num_info_bits_per_frame
    return {
        "receiver": receiver,
        "environment_id": environment.identifier,
        "sweep_tags": sweep_tags(environment, args),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "channel_model": f"TDL-{environment.channel_model}",
        "delay_spread_ns": environment.delay_spread_ns,
        "speed_mps": environment.speed_mps,
        "snr_db": snr_db,
        "evaluation_seed": evaluation_seed,
        "target_coded_bits": args.target_coded_bits,
        "actual_coded_bits": actual_coded_bits,
        "actual_info_bits": actual_info_bits,
        "total_frames": total_frames,
        "batch_size": args.batch_size,
        "ber": ber,
        "fer": fer,
        "completed_at_utc": utc_now(),
    }


def environment_has_missing_results(
    args: argparse.Namespace,
    environment: Environment,
    completed_keys: set[tuple[str, str, float, float, float, int]],
) -> bool:
    return any(
        result_key(receiver, environment, snr_db, seed) not in completed_keys
        for snr_db in args.snr_dbs
        for seed in args.evaluation_seeds
        for receiver in RECEIVERS
    )


def evaluate_matrix(
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    frozen_weights: Mapping[str, torch.Tensor],
    environments: Sequence[Environment],
    raw_path: Path,
    completed_keys: set[tuple[str, str, float, float, float, int]],
) -> None:
    for environment_index, environment in enumerate(environments, start=1):
        if not environment_has_missing_results(args, environment, completed_keys):
            print(
                f"[resume] environment {environment_index}/{len(environments)} "
                f"already complete: {environment.identifier}"
            )
            continue

        print(
            f"[environment {environment_index}/{len(environments)}] "
            f"TDL-{environment.channel_model}, "
            f"delay={environment.delay_spread_ns:g} ns, "
            f"fixed speed={environment.speed_mps:g} m/s"
        )
        config.seed = args.evaluation_seeds[0]
        trainer = make_evaluation_trainer(args, environment, frozen_weights)
        total_frames = math.ceil(
            args.target_coded_bits / trainer.num_coded_bits_per_frame
        )

        for snr_db in args.snr_dbs:
            for evaluation_seed in args.evaluation_seeds:
                for receiver in RECEIVERS:
                    key = result_key(
                        receiver,
                        environment,
                        snr_db,
                        evaluation_seed,
                    )
                    if key in completed_keys:
                        continue

                    # Reusing the same seed for the two receiver paths makes
                    # their bits, channel realizations, and noise matched.
                    config.seed = evaluation_seed
                    if receiver == NEURAL_RECEIVER:
                        ber, fer = trainer.evaluate_neural_snr(
                            snr_db=snr_db,
                            total_frames=total_frames,
                            batch_size=args.batch_size,
                        )
                    else:
                        ber, fer = trainer.evaluate_classical_snr(
                            snr_db=snr_db,
                            total_frames=total_frames,
                            batch_size=args.batch_size,
                        )

                    row = result_row(
                        args,
                        checkpoint_path,
                        checkpoint_sha256,
                        receiver,
                        environment,
                        snr_db,
                        evaluation_seed,
                        trainer,
                        total_frames,
                        ber,
                        fer,
                    )
                    append_raw_row(raw_path, row)
                    completed_keys.add(key)
                    print(
                        f"  snr={snr_db:g} dB seed={evaluation_seed} "
                        f"receiver={receiver} ber={ber:.6g} fer={fer:.6g}"
                    )

        assert_weights_unchanged(trainer, frozen_weights)
        del trainer
        release_device_cache()


def release_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_results(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["receiver"]),
            str(row["channel_model"]),
            float(row["delay_spread_ns"]),
            float(row["speed_mps"]),
            float(row["snr_db"]),
        )
        groups[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group in groups.items():
        ber_values = [float(row["ber"]) for row in group]
        fer_values = [float(row["fer"]) for row in group]
        summaries.append(
            {
                "receiver": key[0],
                "channel_model": key[1],
                "delay_spread_ns": key[2],
                "speed_mps": key[3],
                "snr_db": key[4],
                "num_seeds": len(group),
                "target_coded_bits": int(group[0]["target_coded_bits"]),
                "actual_coded_bits_per_seed": int(group[0]["actual_coded_bits"]),
                "actual_info_bits_per_seed": int(group[0]["actual_info_bits"]),
                "mean_ber": statistics.fmean(ber_values),
                "std_ber": sample_std(ber_values),
                "ci95_half_width_ber": ci95_half_width(ber_values),
                "mean_fer": statistics.fmean(fer_values),
                "std_fer": sample_std(fer_values),
                "ci95_half_width_fer": ci95_half_width(fer_values),
            }
        )

    return sorted(
        summaries,
        key=lambda row: (
            str(row["channel_model"]),
            float(row["delay_spread_ns"]),
            float(row["speed_mps"]),
            float(row["snr_db"]),
            str(row["receiver"]),
        ),
    )


def sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def ci95_half_width(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    from scipy.stats import t as student_t

    critical_value = float(student_t.ppf(0.975, df=len(values) - 1))
    return critical_value * statistics.stdev(values) / math.sqrt(len(values))


def plot_focused_charts(
    output_dir: Path,
    summaries: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> None:
    reference = reference_environment(args)
    common_note = (
        f"Frozen checkpoint; {len(args.evaluation_seeds)} seed(s); "
        f"3 μs channel window; CP={args.cp_len}. "
        "Late TDL energy may be truncated at large delay spreads."
    )

    plot_category(
        output_dir / DELAY_CHART_FILENAME,
        summaries,
        title=(
            "Delay-spread stress test "
            f"(TDL-{reference.channel_model}, {reference.speed_mps:g} m/s)"
        ),
        variants=args.delay_spreads_ns,
        variant_label=lambda value: f"{float(value):g} ns",
        row_matches=lambda row, value: (
            str(row["channel_model"]) == f"TDL-{reference.channel_model}"
            and float(row["speed_mps"]) == reference.speed_mps
            and float(row["delay_spread_ns"]) == float(value)
        ),
        common_note=common_note,
    )
    plot_category(
        output_dir / MOBILITY_CHART_FILENAME,
        summaries,
        title=(
            "Fixed-mobility stress test "
            f"(TDL-{reference.channel_model}, "
            f"{reference.delay_spread_ns:g} ns)"
        ),
        variants=args.speeds_mps,
        variant_label=lambda value: f"{float(value):g} m/s",
        row_matches=lambda row, value: (
            str(row["channel_model"]) == f"TDL-{reference.channel_model}"
            and float(row["delay_spread_ns"]) == reference.delay_spread_ns
            and float(row["speed_mps"]) == float(value)
        ),
        common_note=common_note,
    )
    plot_category(
        output_dir / CHANNEL_MODEL_CHART_FILENAME,
        summaries,
        title=(
            "TDL-profile stress test "
            f"({reference.delay_spread_ns:g} ns, "
            f"{reference.speed_mps:g} m/s)"
        ),
        variants=args.channel_models,
        variant_label=lambda value: f"TDL-{value}",
        row_matches=lambda row, value: (
            str(row["channel_model"]) == f"TDL-{value}"
            and float(row["delay_spread_ns"]) == reference.delay_spread_ns
            and float(row["speed_mps"]) == reference.speed_mps
        ),
        common_note=common_note,
    )


def plot_category(
    path: Path,
    summaries: Sequence[Mapping[str, Any]],
    *,
    title: str,
    variants: Sequence[Any],
    variant_label: Callable[[Any], str],
    row_matches: Callable[[Mapping[str, Any], Any], bool],
    common_note: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
    colors = plt.get_cmap("tab10").colors
    receiver_styles = {
        NEURAL_RECEIVER: ("Frozen neural", "-", "o"),
        CLASSICAL_RECEIVER: ("Classical LS/LMMSE", "--", "s"),
    }
    plot_floor = 1e-8

    for variant_index, variant in enumerate(variants):
        color = colors[variant_index % len(colors)]
        for receiver, (receiver_label, line_style, marker) in receiver_styles.items():
            matching = sorted(
                (
                    row
                    for row in summaries
                    if str(row["receiver"]) == receiver
                    and row_matches(row, variant)
                ),
                key=lambda row: float(row["snr_db"]),
            )
            if not matching:
                raise RuntimeError(
                    f"no summarized results available for {variant_label(variant)} "
                    f"and receiver {receiver}"
                )

            snrs = [float(row["snr_db"]) for row in matching]
            label = f"{variant_label(variant)} — {receiver_label}"
            for axis, metric in zip(axes, ("ber", "fer")):
                means = [
                    max(float(row[f"mean_{metric}"]), plot_floor)
                    for row in matching
                ]
                lower = [
                    max(
                        float(row[f"mean_{metric}"])
                        - float(row[f"ci95_half_width_{metric}"]),
                        plot_floor,
                    )
                    for row in matching
                ]
                upper = [
                    max(
                        float(row[f"mean_{metric}"])
                        + float(row[f"ci95_half_width_{metric}"]),
                        plot_floor,
                    )
                    for row in matching
                ]
                axis.semilogy(
                    snrs,
                    means,
                    color=color,
                    linestyle=line_style,
                    marker=marker,
                    linewidth=1.8,
                    markersize=4.5,
                    label=label,
                )
                axis.fill_between(snrs, lower, upper, color=color, alpha=0.10)

    axes[0].set_title("Bit Error Rate")
    axes[0].set_ylabel("BER")
    axes[1].set_title("Frame Error Rate")
    axes[1].set_ylabel("FER")
    for axis in axes:
        axis.set_xlabel("SNR (dB)")
        axis.grid(True, which="both", linestyle=":", linewidth=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, max(1, len(labels))),
        fontsize=8,
        frameon=True,
    )
    fig.suptitle(title)
    fig.text(0.5, 0.075, common_note, ha="center", va="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    environments = build_environment_matrix(args)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    run_signature = build_run_signature(
        args,
        checkpoint_path,
        checkpoint_sha256,
        environments,
    )
    proposed_manifest = build_manifest(
        args,
        checkpoint_path,
        checkpoint_sha256,
        environments,
        run_signature,
        output_dir,
    )
    manifest = prepare_output_directory(args, output_dir, proposed_manifest)

    raw_path = output_dir / RAW_FILENAME
    initialize_raw_csv(raw_path)
    raw_rows = load_raw_rows(raw_path)
    expected_keys = expected_result_keys(args, environments)
    completed_keys = validate_existing_rows(raw_rows, expected_keys)

    manifest["completed_rows"] = len(completed_keys)
    manifest["complete"] = False
    write_json_atomic(output_dir / MANIFEST_FILENAME, manifest)

    print(f"device: {config.device}")
    print(f"loading frozen checkpoint: {checkpoint_path}")
    print(f"checkpoint SHA-256: {checkpoint_sha256}")
    print("training: disabled")
    print(
        f"sweep mode: {args.sweep_mode}; environments: {len(environments)}; "
        f"expected result rows: {len(expected_keys)}"
    )
    print(
        f"resuming {len(completed_keys)} completed rows; "
        f"{len(expected_keys) - len(completed_keys)} rows remain"
    )
    if args.quick:
        print("WARNING: --quick is a plumbing check, not research evidence.")

    frozen_weights = load_frozen_weights(checkpoint_path)
    evaluate_matrix(
        args,
        checkpoint_path,
        checkpoint_sha256,
        frozen_weights,
        environments,
        raw_path,
        completed_keys,
    )

    if completed_keys != expected_keys:
        missing_count = len(expected_keys - completed_keys)
        raise RuntimeError(f"stress test ended with {missing_count} missing result rows")
    if file_sha256(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("checkpoint file changed during evaluation")

    raw_rows = load_raw_rows(raw_path)
    validate_existing_rows(raw_rows, expected_keys)
    summary_rows = summarize_results(raw_rows)
    write_csv_atomic(
        output_dir / SUMMARY_FILENAME,
        summary_rows,
        SUMMARY_COLUMNS,
    )
    plot_focused_charts(output_dir, summary_rows, args)

    manifest["completed_rows"] = len(completed_keys)
    manifest["complete"] = True
    manifest["completed_at_utc"] = utc_now()
    write_json_atomic(output_dir / MANIFEST_FILENAME, manifest)

    print(f"saved raw per-seed results: {raw_path}")
    print(f"saved aggregated results: {output_dir / SUMMARY_FILENAME}")
    print(f"saved delay-spread chart: {output_dir / DELAY_CHART_FILENAME}")
    print(f"saved mobility chart: {output_dir / MOBILITY_CHART_FILENAME}")
    print(
        "saved channel-model chart: "
        f"{output_dir / CHANNEL_MODEL_CHART_FILENAME}"
    )
    print(f"saved run manifest: {output_dir / MANIFEST_FILENAME}")


if __name__ == "__main__":
    main()
