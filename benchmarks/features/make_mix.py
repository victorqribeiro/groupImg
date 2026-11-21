"""Build the public test set: photos from Imagenette plus textures from DTD.

  python -I benchmarks/features/make_mix.py IMAGENETTE_DIR DTD_DIR OUT_DIR
         [--photos 150] [--texture-classes 10] [--textures 50] [--seed 0]

IMAGENETTE_DIR is the extracted imagenette2-320 folder (train/ and val/ with
one folder per class). DTD_DIR is the extracted dtd folder (images/ with one
folder per texture). Copies --photos images of each of the 10 Imagenette
classes and --textures images of --texture-classes random DTD textures to
OUT_DIR/photos/<class>/ and OUT_DIR/textures/<texture>/.
"""
import os
import sys
import random
import shutil
import argparse

IMAGENETTE = {
    "n01440764": "tench", "n02102040": "springer", "n02979186": "cassette_player",
    "n03000684": "chain_saw", "n03028079": "church", "n03394916": "french_horn",
    "n03417042": "garbage_truck", "n03425413": "gas_pump", "n03445777": "golf_ball",
    "n03888257": "parachute",
}


def images_in(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg")))


def copy(files, dest):
    os.makedirs(dest)
    for src in files:
        shutil.copyfile(src, os.path.join(dest, os.path.basename(src)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagenette")
    ap.add_argument("dtd")
    ap.add_argument("out")
    ap.add_argument("--photos", type=int, default=150)
    ap.add_argument("--texture-classes", type=int, default=10)
    ap.add_argument("--textures", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if os.path.exists(args.out) and os.listdir(args.out):
        sys.exit(f"{args.out} is not empty")
    rng = random.Random(args.seed)

    for wnid, name in IMAGENETTE.items():
        pool = [os.path.join(args.imagenette, part, wnid, f)
                for part in ("train", "val")
                for f in images_in(os.path.join(args.imagenette, part, wnid))]
        copy(rng.sample(pool, args.photos), os.path.join(args.out, "photos", name))

    root = os.path.join(args.dtd, "images")
    textures = sorted(c for c in os.listdir(root) if os.path.isdir(os.path.join(root, c)))
    for name in sorted(rng.sample(textures, args.texture_classes)):
        pool = [os.path.join(root, name, f) for f in images_in(os.path.join(root, name))]
        copy(rng.sample(pool, args.textures), os.path.join(args.out, "textures", name))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
