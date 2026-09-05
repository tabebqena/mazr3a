#!/usr/bin/env python3
"""Read live lm-sensors values on demand and print them as KEY=VALUE lines.

On-demand reference helper for the sensor keys used by the monitoring stack.
Nothing is read from or written to disk: each run calls `sensors` once through
monitor_lib (live, on demand) and prints the parsed values to stdout.

Usage: python3 collect_sensors.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import monitor_lib as lib  # noqa: E402


def main():
    for key, value in lib.collect(lib.run_sensors()):
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
