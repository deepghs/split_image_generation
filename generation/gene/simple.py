import random

from PIL import Image

from .background import create_background
from ..images import get_random_images
from ..trans import create_transform


def simple_generation_with_image(image: Image.Image, size_expand: float = 0.3):
    left = random.randint(0, int(image.width * size_expand))
    top = random.randint(0, int(image.height * size_expand))
    right = random.randint(0, int(image.width * size_expand))
    bottom = random.randint(0, int(image.height * size_expand))

    canvas_width = image.width + left + right
    canvas_height = image.height + top + bottom
    canvas = create_background(canvas_width, canvas_height)
    bboxes = []

    x0, y0 = left, top
    x1, y1 = x0 + image.width, y0 + image.height
    canvas.paste(image, (x0, y0))
    bboxes.append(((x0, y0, x1, y1), 'box', 1.0))
    return canvas, bboxes


def simple(size_expand: float = 0.3):
    images = get_random_images(1, max_workers=1)
    trans = create_transform()
    images = [trans(x) for x in images]
    return simple_generation_with_image(images[0], size_expand=size_expand)


def simple_zero():
    return simple(size_expand=0.0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from imgutils.detect import detection_visualize

    # canvas, bboxes = simple()
    canvas, bboxes = simple_zero()
    plt.imshow(detection_visualize(canvas, bboxes))
    plt.show()
