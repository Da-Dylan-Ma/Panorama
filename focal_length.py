#!/usr/bin/env python3
"""
Compute focal length in **pixels** for every JPEG in a directory.
Requires: Pillow, piexif (pip install pillow piexif)
Usage:    python focal_px.py /path/to/images
"""
import sys, pathlib, piexif

UNIT_MM = {1:1, 2:25.4, 3:10}          # inch→25.4 mm ; cm→10 mm

def focal_px(jpeg):
    ex = piexif.load(str(jpeg))
    try:
        f_mm = ex["Exif"][piexif.ExifIFD.FocalLength]
        f_mm = f_mm[0] / f_mm[1]        # rational -> float
        res  = ex["Exif"][piexif.ExifIFD.FocalPlaneXResolution]
        res  = res[0] / res[1]
        unit = ex["Exif"][piexif.ExifIFD.FocalPlaneResolutionUnit]
        px_per_mm = res / UNIT_MM.get(unit,25.4)
        return f_mm * px_per_mm
    except KeyError:
        return None

def main(folder):
    vals = []
    for jpg in sorted(pathlib.Path(folder).glob("*.J*G")):
        fpx = focal_px(jpg)
        print(f"{jpg.name:20s}  f_px = {fpx:.1f}" if fpx
              else f"{jpg.name:20s}  EXIF missing → skipped")
        if fpx: vals.append(fpx)
    if vals:
        avg = sum(vals)/len(vals)
        print(f"\nAverage focal_px = {avg:.1f}")
        # write a line you can paste into img_list.txt
        with open("focal_px_out.txt","w") as f:
            f.write(f"{avg:.1f}\n")
        print("Saved average to focal_px_out.txt")
    else:
        print("No usable EXIF found; see fallback below.")

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("Usage: python focal_px.py <image_folder>")
        sys.exit(1)
    main(sys.argv[1])
