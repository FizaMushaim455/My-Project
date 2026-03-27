"""
main.py — ASL Sign-To-Speech (Developer: Fiza Mushaim | 2023-ag-9944 | Big Data Analysis)

Usage:
------
    python main.py run        Launch the Real-time Geometric Recognition tool.
    python main.py --help     Show this help message.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.utils.logging_config import setup_logging


def cmd_run(args: argparse.Namespace) -> None:
    """Handle the 'run' command for high-accuracy geometric recognition."""
    from src.core.speech_synthesizer import SpeechSynthesizer
    from src.pipelines.landmark_inference import run_landmark_inference
    
    logger = logging.getLogger(__name__)
    logger.info("Starting real-time GEOMETRIC ASL recognition by Fiza Mushaim...")

    recognised_text = run_landmark_inference(
        camera_index=args.camera,
    )

    logger.info("Final recognised text: '%s'", recognised_text)

    if recognised_text and not args.no_speech:
        synth = SpeechSynthesizer()
        urdu_text = synth.speak(recognised_text)
        logger.info("Urdu Speech Logic: '%s'", urdu_text)
    elif not recognised_text:
        logger.warning("No ASL gestures were captured.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ASL Sign-To-Speech Pro Engine by Fiza Mushaim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- run ----
    run_parser = subparsers.add_parser(
        "run",
        help="Launch the high-accuracy ASL recognition (Geometric Engine).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (usually 0).",
    )
    run_parser.add_argument(
        "--no-speech",
        action="store_true",
        help="Disable audio output.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
