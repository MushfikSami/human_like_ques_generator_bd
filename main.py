"""
main.py — CLI Entry Point

Provides a command-line interface for the Bangladeshi Persona & Human-Like
Question Generation Agent.

Usage:
    python main.py --init-db                    # Create database tables
    python main.py --gen-personas 100           # Generate 100 personas
    python main.py --test                       # Micro-batch QA run (100 personas)
    python main.py --generate --batch-size 50   # Full generation run
"""

import argparse
import asyncio
import logging
import sys

from db import init_tables
from persona_generator import generate_personas
from question_generator import run


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("generation.log", encoding="utf-8"),
        ],
    )


def main():
    """Parse CLI arguments and dispatch to the appropriate action."""
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Bangladeshi Persona & Human-Like Question Generation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --init-db                    Create database tables
  python main.py --gen-personas 100           Generate 100 test personas
  python main.py --test                       Micro-batch QA run (batch_size=100)
  python main.py --gen-personas 25000         Generate full persona set
  python main.py --generate --batch-size 50   Full question generation
        """,
    )

    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Create/verify DB tables (personas, generated_questions)",
    )
    parser.add_argument(
        "--gen-personas",
        type=int,
        default=0,
        metavar="N",
        help="Generate N personas and insert into the database",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run question generation for all unprocessed personas",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Number of concurrent LLM calls per batch (default: 50)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Micro-batch QA run with batch_size=100 for testing",
    )

    args = parser.parse_args()

    # Check that at least one action was requested
    if not (args.init_db or args.gen_personas > 0 or args.generate or args.test):
        parser.print_help()
        sys.exit(1)

    # Execute actions in order
    if args.init_db:
        logger.info("Initialising database tables...")
        init_tables()
        logger.info("Database tables ready.")

    if args.gen_personas > 0:
        logger.info("Generating %d personas...", args.gen_personas)
        generate_personas(args.gen_personas)
        logger.info("Persona generation complete.")

    if args.test:
        logger.info("Starting micro-batch QA run (batch_size=100)...")
        asyncio.run(run(batch_size=100))
        logger.info("Micro-batch QA run complete.")
    elif args.generate:
        logger.info("Starting full question generation (batch_size=%d)...", args.batch_size)
        asyncio.run(run(batch_size=args.batch_size))
        logger.info("Question generation complete.")


if __name__ == "__main__":
    main()
