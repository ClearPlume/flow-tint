import json
from dataclasses import dataclass
from textwrap import dedent

from src.flow_tint.path import get_data_dir
from src.flow_tint.core.utils import rgb_to_oklch


@dataclass
class ColorStage:
    name: str
    l: int
    c: int
    h: int

    def __str__(self):
        return dedent(f"""\
        {self.name}
        color {self.l} {self.c} {self.h}""")


# 使用示例
if __name__ == "__main__":
    data_dir = get_data_dir()
    color_database = json.load(open(data_dir / "seed/stage1-data.json", encoding="utf-8"))
    print(f"颜色数据库包含 {len(color_database)} 个颜色")

    colors: list[ColorStage] = []

    for i, (name, rgb) in enumerate(color_database.items()):
        lch = rgb_to_oklch(*rgb)
        color = ColorStage(name, *lch)

        if i < 10:
            print(color)

        colors.append(color)

    (data_dir / "samples").mkdir(parents=True, exist_ok=True)

    with open(data_dir / "samples/stage1_samples.txt", "w", encoding="utf-8") as data:
        data.write("\n\n".join(map(str, colors)))
