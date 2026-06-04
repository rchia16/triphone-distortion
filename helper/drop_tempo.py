from pathlib import Path
import subprocess

FFMPEG = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"

root_folder = Path("/Users/160843/Downloads/tmp")
input_folder = root_folder / "speech-assets-triphone" / "man" / "triphone"
output_folder = root_folder / "speech-assets-triphone" / "man" / "triphone-slowed"

tempo = 0.50  # -50% tempo = 50% speed

for input_path in sorted(input_folder.rglob("*.wav")):
    rel_path = input_path.relative_to(input_folder)
    output_path = output_folder / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG,
        "-y",
        "-i", str(input_path),
        "-ar", str(48000),
        "-af", f"rubberband=tempo={tempo}",
        str(output_path),
    ]

    print("Processing:", input_path)
    subprocess.run(cmd, check=True)

    print("Wrote:", output_path, output_path.exists())
