from PIL import Image, ImageSequence
import sys
import os

if len(sys.argv) != 3:
    print("Usage: python halve_gif_fps.py input.gif output.gif")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

print(f"Loading {input_path}...")

with Image.open(input_path) as im:
    frames = []
    durations = []

    for i, frame in enumerate(ImageSequence.Iterator(im)):
        if i % 2 == 0:
            frames.append(frame.copy())
            duration = frame.info.get("duration", 100)
            durations.append(duration * 2)  # stretch timing to simulate original speed

    print(f"Keeping {len(frames)} frames out of {i+1} total.")

    print(f"Saving to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True
    )

print("Done.")

