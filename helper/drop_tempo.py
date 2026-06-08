from pathlib import Path
import subprocess
import shutil

# FFMPEG = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"

# root_folder = Path("/Users/160843/Downloads/tmp")
root_folder = Path("/data/raqchia/audio-assets/")
input_folder = root_folder / "RayAssets" / "man" / "triphone"
output_folder = root_folder / "RayAssets" / "man" / "triphone-slowed"

tempo = 0.60  # -50% tempo = 50% speed

for input_path in sorted(input_folder.rglob("*.wav")):
    rel_path = input_path.relative_to(input_folder)
    output_path = output_folder / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ar", str(48000),
        "-filter", f"atempo={tempo}",
        str(output_path),
    ]

    print("Processing:", input_path)
    subprocess.run(cmd, check=True)

    print("Wrote:", output_path, output_path.exists())

for input_path in sorted(input_folder.rglob("*.json")):
    rel_path = input_path.relative_to(input_folder)
    output_path = output_folder / rel_path
    shutil.copy(input_path, output_path)
