from pathlib import Path


directory = "../matplotlib"

directory = Path(directory)

png_dir = directory / "assets"

"""
for png_path in png_dir.iterdir():
    print(png_path)
"""

png_path = directory.glob("**/*.png")
print(*list(png_path), sep="\n")
