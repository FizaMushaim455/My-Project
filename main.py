"""
main.py — Unified CLI entry point for the ASL Sign-To-Speech project.

Commands
--------
    python main.py train      Train the ASL classifier from the DATASET/
    python main.py run        Real-time webcam recognition + Urdu speech

Examples
--------
    python main.py train --epochs 20 --plots
    python main.py run --camera 0
    python main.py run --no-speech
    python main.py --help
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config.settings import CAMERA_INDEX, DATASET_PATH, EPOCHS, MODEL_PATH
from src.utils.logging_config import setup_logging


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    """Handle the 'train' sub-command."""
    from src.pipelines.train_pipeline import run_training

    plot_dir: Path | None = Path(".") if args.plots else None
    run_training(
        dataset_path=args.dataset,
        model_output_path=args.model,
        epochs=args.epochs,
        plot_output_dir=plot_dir,
    )


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the 'run' sub-command."""
    from src.core.speech_synthesizer import SpeechSynthesizer
    from src.pipelines.inference_pipeline import run_realtime

    logger = logging.getLogger(__name__)
    logger.info("Starting real-time ASL recognition. Press Esc to stop.")

    recognised_text = run_realtime(
        model_path=args.model,
        camera_index=args.camera,
    )

    logger.info("Recognised text: '%s'", recognised_text)

    if recognised_text and not args.no_speech:
        synth = SpeechSynthesizer()
        urdu_text = synth.speak(recognised_text)
        logger.info("Urdu output: '%s'", urdu_text)
    elif not recognised_text:
        logger.warning("No ASL letters were detected.")
    else:
        logger.info("--no-speech flag set; skipping audio output.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ASL Sign-To-Speech Conversion — production CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    train_parser = subparsers.add_parser(
        "train",
        help="Train the ASL CNN classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the DATASET/ root directory.",
    )
    train_parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Output path for the trained .h5 model.",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs.",
    )
    train_parser.add_argument(
        "--plots",
        action="store_true",
        help="Save training history plots to the current directory.",
    )

    # ---- run ----
    run_parser = subparsers.add_parser(
        "run",
        help="Launch real-time ASL recognition with Urdu speech output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to the pre-trained .h5 model.",
    )
    run_parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="Camera device index (0 = default webcam).",
    )
    run_parser.add_argument(
        "--no-speech",
        action="store_true",
        help="Skip text-to-speech output (recognition only).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    dispatch = {"train": cmd_train, "run": cmd_run}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
