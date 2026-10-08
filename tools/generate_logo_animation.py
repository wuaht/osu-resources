#!/usr/bin/env python3
"""
Generates the timing of the intro logo animation textures (Textures/Intro/Triangles/logo-*.png).

The textures are read by sh_LogoAnimation.fs:
- the green channel is the opacity of the logo (its shape),
- the red channel is the animation progress at which a pixel appears (0 = at the start, 255 = never).

This script keeps the green channel and recalculates the red channel, so that each stroke of the logo draws itself
along its path, like the original lazer logo:
- every stroke is a closed outline, which is reduced to its centre line and followed from a start point once around,
- the progress along the stroke is mapped to the time at which it appears, slowing down towards its end,
- the highlight texture runs ahead of the background texture, so that a highlighted tip leads the drawing.

Requires Pillow, numpy, scipy and scikit-image.

Usage:
    python tools/generate_logo_animation.py [--preview <directory>]
"""

import argparse
import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

TEXTURE_DIRECTORY = Path(__file__).resolve().parent.parent / "osu.Game.Resources" / "Textures" / "Intro" / "Triangles"

# The timing of each kind of stroke as (start, end, curve), in red channel values.
# A higher curve makes the stroke appear quickly at first and slow down towards its end.
# The values approximate the original lazer logo textures.
TIMINGS = {
    "logo-highlight": {
        "outer_ring": (10, 216, 4.0),
        "inner_ring": (5, 10, 0.0),
        "letter": (5, 216, 7.0),
        "counter": (41, 41, 0.0),
    },
    "logo-background": {
        "outer_ring": (30, 242, 1.9),
        "inner_ring": (5, 30, 1.5),
        "letter": (5, 237, 2.5),
        "counter": (4, 29, 1.5),
    },
}

# Where each kind of stroke starts and in which direction it is drawn (as seen on screen).
STARTS = {
    "outer_ring": ("top", "counterclockwise"),
    "inner_ring": ("bottom", "clockwise"),
    "letter": ("top_right", "counterclockwise"),
    "counter": ("top_right", "counterclockwise"),
}

# The minimum number of pixels of a stroke. Smaller components are anti-aliasing fragments.
MIN_STROKE_PIXELS = 200

# The radius around the start point which is cut out of a stroke's centre line, separating its start from its end.
CUT_RADIUS = 3

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def bfs(pixels, start):
    """Returns the geodesic distances from start to all pixels connected to it (8-connectivity)."""
    distances = {start: 0}
    queue = deque([start])

    while queue:
        y, x = queue.popleft()
        d = distances[(y, x)]

        for dy, dx in NEIGHBOURS:
            n = (y + dy, x + dx)

            if n in pixels and n not in distances:
                distances[n] = d + (1.4142 if dy and dx else 1)
                queue.append(n)

    return distances


def find_start(points, start):
    ys = np.array([p[0] for p in points])
    xs = np.array([p[1] for p in points])

    if start == "top":
        score = -ys + 1e-3 * xs
    elif start == "bottom":
        score = ys + 1e-3 * xs
    elif start == "top_right":
        score = xs - ys
    else:
        raise ValueError(start)

    return points[int(np.argmax(score))]


def order_stroke(skeleton, start, direction):
    """
    Returns the progress (0 to 1) along a closed stroke for the pixels of its centre line,
    starting at the given start point and going around in the given direction.
    """
    points = list(zip(*np.nonzero(skeleton)))
    start_point = find_start(points, start)

    # cut the loop at the start point, so that it becomes a path from one side of the cut to the other.
    remaining = {p for p in points if math.dist(p, start_point) > CUT_RADIUS}

    # the side of the cut to start from. the path continues from there all the way around to the other side.
    near_cut = sorted((p for p in remaining if math.dist(p, start_point) <= CUT_RADIUS + 2), key=lambda p: math.dist(p, start_point))
    distances = bfs(remaining, near_cut[0])

    # small fragments not connected to the main path (e.g. from skeletonisation artifacts) are ignored.
    length = max(distances.values())
    progress = {p: d / length for p, d in distances.items()}

    if len(progress) < 0.9 * len(points):
        raise RuntimeError(f"Stroke at {start_point} isn't a closed loop ({len(progress)} of {len(points)} centre line pixels connected).")

    # the side of the cut at the end of the path must be close to the start point, otherwise the stroke isn't a loop.
    end_point = max(progress, key=progress.get)

    if math.dist(end_point, start_point) > CUT_RADIUS + 3:
        raise RuntimeError(f"Stroke at {start_point} isn't a closed loop (ends at {end_point}).")

    # the signed area of the path determines its direction. with y pointing down, a positive area is clockwise on screen.
    ordered = sorted(progress, key=progress.get)
    area = sum(a[1] * b[0] - b[1] * a[0] for a, b in zip(ordered, ordered[1:] + ordered[:1])) / 2
    clockwise = area > 0

    if clockwise != (direction == "clockwise"):
        progress = {p: 1 - u for p, u in progress.items()}

    return progress


def classify(components):
    """Determines the kind of each stroke from its bounding box."""
    # (label, y0, x0, y1, x1)
    by_size = sorted(components, key=lambda c: (c[3] - c[1]) * (c[4] - c[2]), reverse=True)
    kinds = {by_size[0][0]: "outer_ring", by_size[1][0]: "inner_ring"}

    letters = by_size[2:]

    for c in letters:
        # a counter (e.g. the hole of an "o") lies inside the bounding box of another letter.
        inside = any(o is not c and o[1] < c[1] and o[2] < c[2] and o[3] > c[3] and o[4] > c[4] for o in letters)
        kinds[c[0]] = "counter" if inside else "letter"

    return kinds


def timing(progress, start, end, curve):
    if curve == 0:
        return start + (end - start) * progress

    return start + (end - start) * (np.exp(curve * progress) - 1) / (math.exp(curve) - 1)


def generate(name):
    path = TEXTURE_DIRECTORY / f"{name}.png"
    image = np.array(Image.open(path).convert("RGB"))
    alpha = image[..., 1]

    labels, count = ndimage.label(alpha >= 128, structure=np.ones((3, 3)))
    components = []

    for label, region in enumerate(ndimage.find_objects(labels), start=1):
        if (labels[region] == label).sum() >= MIN_STROKE_PIXELS:
            components.append((label, region[0].start, region[1].start, region[0].stop, region[1].stop))

    kinds = classify(components)

    centre_lines = np.zeros(alpha.shape, dtype=bool)
    values = np.zeros(alpha.shape, dtype=float)

    for label, kind in sorted(kinds.items()):
        skeleton = skeletonize(labels == label)
        progress = order_stroke(skeleton, *STARTS[kind])
        start, end, curve = TIMINGS[name][kind]

        for (y, x), u in progress.items():
            centre_lines[y, x] = True
            values[y, x] = timing(u, start, end, curve)

        print(f"{name}: {kind:<10} {len(progress):5d} centre line pixels, {start}-{end}")

    # every pixel takes the time of the closest centre line pixel.
    # this includes pixels outside of the logo, so that texture filtering at the edges of strokes doesn't mix in other times.
    _, (nearest_y, nearest_x) = ndimage.distance_transform_edt(~centre_lines, return_indices=True)
    red = np.clip(np.round(values[nearest_y, nearest_x]), 0, 254).astype(np.uint8)

    result = np.stack([red, alpha, np.zeros_like(alpha)], axis=-1)
    Image.fromarray(result, "RGB").save(path, optimize=True)

    return result


def save_preview(textures, directory):
    """
    Renders frames of the animation like sh_LogoAnimation.fs.
    As in IntroTriangles, the grey background is drawn above the white highlight, so the highlight is only visible ahead of the background.
    """
    directory.mkdir(parents=True, exist_ok=True)
    size = 400
    frames = []

    def layer(name, progress):
        texture = textures[name]
        return (texture[..., 0] / 255 < progress) * texture[..., 1] / 255

    for progress in [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]:
        highlight = layer("logo-highlight", progress)
        background = layer("logo-background", progress)
        frame = background * 0.6 + (1 - background) * highlight

        frames.append(Image.fromarray((frame * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS))

    columns = 3
    sheet = Image.new("L", (size * columns, size * math.ceil(len(frames) / columns)))

    for i, frame in enumerate(frames):
        sheet.paste(frame, (size * (i % columns), size * (i // columns)))

    sheet.save(directory / "logo-animation-preview.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=Path, help="directory to render frames of the animation to")
    args = parser.parse_args()

    textures = {name: generate(name) for name in TIMINGS}

    if args.preview:
        save_preview(textures, args.preview)


if __name__ == "__main__":
    main()
