import os.path
import random

from src.flow_tint.path import get_data_dir

data_dir = get_data_dir()

with open(data_dir / "samples/stage2_easy_samples.txt", "r", encoding="utf-8") as e:
    samples = e.read().split("\n\n")

with open(data_dir / "samples/stage2_complex_samples.txt", "r", encoding="utf-8") as c:
    samples.extend(c.read().split("\n\n"))

random.shuffle(samples)

if os.path.exists(data_dir / "samples/stage2_samples.txt"):
    os.remove(data_dir / "samples/stage2_samples.txt")

with open(data_dir / "samples/stage2_samples.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(samples))
