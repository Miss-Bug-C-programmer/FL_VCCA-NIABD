"""Estimate NIABD persistent and bounded temporary memory requirements."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-samples", type=int, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--teachers", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=0)
    args = parser.parse_args()
    chunk = args.teachers if args.chunk_size <= 0 else min(args.teachers, args.chunk_size)
    persistent = 2 * args.proxy_samples * args.classes + args.classes
    temporary = chunk * args.proxy_samples * args.classes
    print(json.dumps({
        "persistent_float32_values": persistent,
        "persistent_bytes": persistent * 4,
        "bounded_teacher_chunk_values": temporary,
        "bounded_teacher_chunk_bytes": temporary * 4,
        "proxy_chunk_size": chunk,
    }, indent=2))


if __name__ == "__main__":
    main()
