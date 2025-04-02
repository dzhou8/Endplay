from PIL import Image, ImageSequence
import sys
import os

if len(sys.argv) != 4:
    print("Usage: python sync_gif_durations.py A.gif B.gif output_dir")
    sys.exit(1)

a_path = sys.argv[1]
b_path = sys.argv[2]
output_dir = sys.argv[3]

os.makedirs(output_dir, exist_ok=True)

def load_frames(path):
    with Image.open(path) as im:
        frames = []
        durations = []
        for frame in ImageSequence.Iterator(im):
            frames.append(frame.copy())
            durations.append(frame.info.get("duration", 100))
        return frames, durations

def total_duration(durations):
    return sum(durations)

def pad_to_duration(frames, durations, target):
    total = total_duration(durations)
    while total < target:
        remaining = target - total
        pad_duration = min(remaining, durations[-1])
        frames.append(frames[-1].copy())
        durations.append(pad_duration)
        total += pad_duration
    return frames, durations

def save_gif(frames, durations, output_path):
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True
    )

# Load and align durations
frames_a, durations_a = load_frames(a_path)
frames_b, durations_b = load_frames(b_path)

max_duration = max(total_duration(durations_a), total_duration(durations_b))
frames_a, durations_a = pad_to_duration(frames_a, durations_a, max_duration)
frames_b, durations_b = pad_to_duration(frames_b, durations_b, max_duration)

# Output file names
a_name = os.path.basename(a_path).rsplit(".", 1)[0]
b_name = os.path.basename(b_path).rsplit(".", 1)[0]

a_out = os.path.join(output_dir, f"{a_name}_synced.gif")
b_out = os.path.join(output_dir, f"{b_name}_synced.gif")

# Save synced GIFs
save_gif(frames_a, durations_a, a_out)
save_gif(frames_b, durations_b, b_out)

print(f"Saved:\n  {a_out}\n  {b_out}")

