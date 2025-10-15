import random
from typing import Union, Tuple

from PIL import Image
from imgutils.resource import random_bg_image
from torchvision.transforms import RandomResizedCrop


def create_background(canvas_width: int, canvas_height: int, pure_ratio: float = 0.3,
                      is_check: bool = False) -> Union[Image.Image, Tuple[Image.Image, bool]]:
    if random.random() < pure_ratio:
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        canvas = Image.new('RGB', (canvas_width, canvas_height), color=random_color)
        is_pure = True
    else:
        canvas = RandomResizedCrop(
            size=(canvas_height, canvas_width),
            scale=(0.5, 1.0),
            ratio=(0.8, 1.2),
        )(random_bg_image())
        is_pure = False
    assert canvas.size == (canvas_width, canvas_height)
    if is_check:
        return canvas, is_pure
    else:
        return canvas
