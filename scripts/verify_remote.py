#!/usr/bin/env python3
"""Run ON the remote Frigate host after deploy: fetch /api/config and print
the effective per-camera object.track lists to confirm cams 4-7 ignore animals.
"""
import json
import urllib.request

with urllib.request.urlopen("http://localhost:5000/api/config", timeout=30) as r:
    d = json.load(r)

print("GLOBAL_TRACK:", d["objects"]["track"])
for c in ["cam01", "cam04", "cam05", "cam06", "cam07"]:
    cam = d["cameras"][c]
    t = cam.get("objects", {}).get("track")
    print(f"{c}_TRACK:", t if t is not None else "<global>")
