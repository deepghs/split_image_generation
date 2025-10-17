import random

import numpy as np
from PIL import Image

from .background import create_background
from .inpaint import inpaint_unpasted_regions
from ..images import get_random_images
from ..trans import create_transform


def simple_generation_with_image(image: Image.Image, size_expand: float = 0.3,
                                 use_inpaint: bool = False, inpaint_radius: int = 5):
    left = random.randint(0, int(image.width * size_expand))
    top = random.randint(0, int(image.height * size_expand))
    right = random.randint(0, int(image.width * size_expand))
    bottom = random.randint(0, int(image.height * size_expand))

    canvas_width = image.width + left + right
    canvas_height = image.height + top + bottom
    if use_inpaint:
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    else:
        canvas = create_background(canvas_width, canvas_height)
    bboxes = []
    pasted_positions = []
    pasted_images = []

    x0, y0 = left, top
    x1, y1 = x0 + image.width, y0 + image.height
    canvas.paste(image, (x0, y0))
    pasted_positions.append((x0, y0))
    pasted_images.append(image)
    bboxes.append(((x0, y0, x1, y1), 'box', 1.0))
    if use_inpaint:
        canvas = inpaint_unpasted_regions(canvas, pasted_positions, pasted_images, inpaint_radius=inpaint_radius)
    return canvas, bboxes


def simple(size_expand: float = 0.3):
    images = get_random_images(1, max_workers=1)
    trans = create_transform()
    images = [trans(x) for x in images]
    return simple_generation_with_image(images[0], size_expand=size_expand)


def simple_zero():
    return simple(size_expand=0.0)


def simple_with_inpaint(size_expand: float = 0.3,
                        inp_e2_mean: float = 2.5, inp_e2_std: float = 1.2, inp_min: int = 1, inp_max: int = 30):
    images = get_random_images(1, max_workers=1)
    trans = create_transform()
    images = [trans(x) for x in images]
    inpaint_radius = int(min(max(2 ** np.random.normal(inp_e2_mean, inp_e2_std), inp_min), inp_max))
    return simple_generation_with_image(
        images[0],
        size_expand=size_expand,
        use_inpaint=True,
        inpaint_radius=inpaint_radius,
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from imgutils.detect import detection_visualize

    # canvas, bboxes = simple()
    # canvas, bboxes = simple_zero()
    canvas, bboxes = simple_with_inpaint()
    plt.imshow(detection_visualize(canvas, bboxes))
    plt.show()
