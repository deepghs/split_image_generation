import random
from typing import Optional

import numpy as np
from PIL import Image

from generation.gene.inpaint import inpaint_unpasted_regions
from .background import create_background
from ..images import get_random_images
from ..trans import create_transform_with_fixed_size


def grid_random(
        nx: int, ny: int,
        width_mean: int = 360, width_std: int = 90, width_min: Optional[int] = 66, width_max: Optional[int] = None,
        height_mean: int = 360, height_std: int = 90, height_min: Optional[int] = 66, height_max: Optional[int] = None,
        int_mean: int = 25, int_std: int = 12, int_min: Optional[int] = 0, int_max: Optional[int] = None,
        prob_last_drop: float = 0.4,
        ld_mean: int = 4, ld_std: int = 2, ld_min: Optional[int] = 1, ld_max: Optional[int] = None,
        use_inpaint: bool = False, inpaint_radius: int = 5,
):
    def _get_width():
        width = np.random.normal(width_mean, width_std)
        if width_min is not None:
            width = max(width, width_min)
        if width_max is not None:
            width = min(width, width_max)
        return int(width)

    def _get_height():
        height = np.random.normal(height_mean, height_std)
        if height_min is not None:
            height = max(height, height_min)
        if height_max is not None:
            height = min(height, height_max)
        return int(height)

    def _get_interval():
        interval = np.random.normal(int_mean, int_std)
        if int_min is not None:
            interval = max(interval, int_min)
        if int_max is not None:
            interval = min(interval, int_max)
        return int(interval)

    if random.random() < prob_last_drop:
        last_drop = np.random.normal(ld_mean, ld_std)
        if ld_min is not None:
            last_drop = max(last_drop, ld_min)
        if ld_max is not None:
            last_drop = min(last_drop, ld_max)
        last_drop = int(round(last_drop))
        last_drop = max(min(last_drop, nx * ny - 1), 0)
    else:
        last_drop = 0

    widths = np.array([_get_width() for _ in range(nx)])
    heights = np.array([_get_height() for _ in range(ny)])
    x_intervals = np.array([_get_interval() for _ in range(nx + 1)])
    y_intervals = np.array([_get_interval() for _ in range(ny + 1)])

    canvas_width = (widths.sum() + x_intervals.sum()).item()
    canvas_height = (heights.sum() + y_intervals.sum()).item()
    if use_inpaint:
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    else:
        canvas = create_background(canvas_width, canvas_height)
    bboxes = []
    pasted_positions = []
    pasted_images = []

    raw_images = get_random_images(count=nx * ny)
    assert len(raw_images) == nx * ny
    raw_images = np.array(raw_images, dtype=object).reshape(nx, ny)
    for i in range(nx):
        for j in range(ny):
            id_ = i * ny + j
            if id_ + last_drop < nx * ny:
                image = raw_images[i, j]
                width, height = widths[i].item(), heights[j].item()
                x0 = x_intervals[:i + 1].sum().item() + widths[:i].sum().item()
                y0 = y_intervals[:j + 1].sum().item() + heights[:j].sum().item()
                x1, y1 = x0 + width, y0 + height
                image = create_transform_with_fixed_size(size=(height, width))(image)
                assert image.size == (width, height)
                canvas.paste(image, (x0, y0))
                pasted_positions.append((x0, y0))
                pasted_images.append(image)
                bboxes.append(((x0, y0, x1, y1), 'box', 1.0))

    canvas = inpaint_unpasted_regions(canvas, pasted_positions, pasted_images, inpaint_radius=inpaint_radius)
    return canvas, bboxes


def grid():
    while True:
        nx, ny = random.randint(1, 10), random.randint(1, 10)
        if nx * ny != 1:
            break
    return grid_random(
        nx, ny,
    )


def grid_avg(
        width_mean: int = 360, width_std: int = 90, width_min: Optional[int] = 66, width_max: Optional[int] = None,
        height_mean: int = 360, height_std: int = 90, height_min: Optional[int] = 66, height_max: Optional[int] = None,
        int_mean: int = 25, int_std: int = 12, int_min: Optional[int] = 0, int_max: Optional[int] = None,
):
    width = np.random.normal(width_mean, width_std)
    if width_min is not None:
        width = max(width, width_min)
    if width_max is not None:
        width = min(width, width_max)
    width = int(width)

    height = np.random.normal(height_mean, height_std)
    if height_min is not None:
        height = max(height, height_min)
    if height_max is not None:
        height = min(height, height_max)
    height = int(height)

    interval = np.random.normal(int_mean, int_std)
    if int_min is not None:
        interval = max(interval, int_min)
    if int_max is not None:
        interval = min(interval, int_max)
    interval = int(interval)

    while True:
        nx, ny = random.randint(1, 10), random.randint(1, 10)
        if nx * ny != 1:
            break
    return grid_random(
        nx, ny,
        width_mean=width, width_std=0, width_min=None, width_max=None,
        height_mean=height, height_std=0, height_min=None, height_max=None,
        int_mean=interval, int_std=0, int_min=None, int_max=None,
    )


def grid_zero():
    return grid_avg(int_mean=0, int_std=0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from imgutils.detect import detection_visualize

    # canvas, bboxes = grid_zero()
    # canvas, bboxes = grid_zero()
    canvas, bboxes = grid_random(3, 3, use_inpaint=True)
    plt.imshow(detection_visualize(canvas, bboxes))
    plt.show()
