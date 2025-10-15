import math
import random
from typing import List

import numpy as np
from PIL import Image
from imgutils.detect import detection_visualize
from imgutils.resource import random_bg_image
from rectpack import newPacker
from torchvision.transforms import RandomResizedCrop

from generation.trans import create_transform
from ..images import get_random_images


def rad_generation_with_images(images: List[Image.Image], area_expand: float = 1.0):
    rectangles = [x.size for x in images]
    total_area = sum(x.size[0] * x.size[1] for x in images)

    while True:
        expand_ratio = random.random() * area_expand + 1.0
        canvas_ratio = np.random.normal(1.0, 0.2)
        canvas_ratio = max(canvas_ratio, 0.3)
        canvas_ratio = min(canvas_ratio, 3.3)
        canvas_width, canvas_height = (int(math.ceil((total_area * expand_ratio) ** 0.5 * (canvas_ratio ** 0.5))),
                                       int(math.ceil((total_area * expand_ratio) ** 0.5 / (canvas_ratio ** 0.5))))

        packer = newPacker()
        for i, canvas_ratio in enumerate(rectangles):
            packer.add_rect(*canvas_ratio, rid=i)
        packer.add_bin(canvas_width, canvas_height)

        # Start packing
        packer.pack()

        all_rectangles = packer.rect_list()
        if len(all_rectangles) < len(images):
            print(f'Pack failed {len(all_rectangles)}/{len(images)}')
        else:
            break

    # 创建一个新的空白图像
    if random.random() < 0.3:
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        canvas = Image.new('RGB', (canvas_width, canvas_height), color=random_color)
    else:
        canvas = RandomResizedCrop(
            size=(canvas_height, canvas_width),
            scale=(0.5, 1.0),
            ratio=(0.8, 1.2),
        )(random_bg_image())
    assert canvas.size == (canvas_width, canvas_height)
    bboxes = []

    max_x1, max_y1 = 0, 0
    for rect in all_rectangles:
        bid, x0, y0, rect_w, rect_h, rid = rect
        x1, y1 = x0 + rect_w, y0 + rect_h
        max_x1, max_y1 = max(x1, max_x1), max(y1, max_y1)

    offset_x, offset_y = random.randint(0, canvas_width - max_x1), random.randint(0, canvas_height - max_y1)

    # 将所有图片按照rect结果放置到结果图像上
    for rect in all_rectangles:
        bid, x0, y0, rect_w, rect_h, rid = rect
        x0, y0 = x0 + offset_x, y0 + offset_y
        x1, y1 = x0 + rect_w, y0 + rect_h

        # 获取对应的图片
        img = images[rid]
        if img.size == (rect_w, rect_h):
            pass
        elif img.size == (rect_h, rect_w):
            img = img.rotate(90, expand=True)
        else:
            assert False, f'Expected {img.size}, actual: {(rect_w, rect_h)}'
        assert img.size == (rect_w, rect_h), f'Expected {img.size}, actual: {(rect_w, rect_h)}'

        # 将图片粘贴到指定位置
        canvas.paste(img, (x0, y0))
        bboxes.append(((x0, y0, x1, y1), 'box', 1.0))

    return canvas, bboxes


def rad_generation(count: int, area_expand: float = 1.0, max_workers: int = 16):
    images = get_random_images(count, max_workers=max_workers)
    trans = create_transform()
    images = [trans(x) for x in images]
    return rad_generation_with_images(images, area_expand)


def rad(area_expand: float = 1.0, max_workers: int = 16):
    count = int(round(2 ** np.random.normal(3.5, 1.2)))
    return rad_generation(count, area_expand=area_expand, max_workers=max_workers)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    canvas, bboxes = rad()
    plt.imshow(detection_visualize(canvas, bboxes))
    plt.show()
