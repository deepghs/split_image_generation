import math

import numpy as np

from .background import create_background


def zero():
    size = np.random.normal(1024, 256)
    size = min(max(size, 480), 2048)
    canvas_ratio = np.random.normal(1.0, 0.2)
    canvas_ratio = max(canvas_ratio, 0.3)
    canvas_ratio = min(canvas_ratio, 3.3)
    canvas_width, canvas_height = (int(math.ceil(size * (canvas_ratio ** 0.5))),
                                   int(math.ceil(size / (canvas_ratio ** 0.5))))
    canvas, is_pure = create_background(canvas_width, canvas_height, is_check=True)
    bboxes = []
    if not is_pure:
        bboxes.append(((0, 0, canvas.width, canvas.height), 'box', 0.0))

    return canvas, bboxes


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from imgutils.detect import detection_visualize

    # canvas, bboxes = simple()
    canvas, bboxes = zero()
    print(canvas, bboxes)
    plt.imshow(detection_visualize(canvas, bboxes))
    plt.show()
