#!/usr/bin/env python3
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a MODENA .inp template with a target structure."
    )
    parser.add_argument("--template", required=True, help="Template .inp file path.")
    parser.add_argument("--output", required=True, help="Rendered .inp output path.")
    parser.add_argument("--structure", required=True, help="Target dot-bracket structure.")
    parser.add_argument(
        "--sequence-constraint",
        default="",
        help="Optional sequence constraint line to inject.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template_path = Path(args.template)
    output_path = Path(args.output)
    text = template_path.read_text(encoding="utf-8")

    rendered = text.replace("{{TARGET_STRUCTURE}}", args.structure)
    rendered = rendered.replace("{{SEQUENCE_CONSTRAINT}}", args.sequence_constraint)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
