import numpy as np

from .grid import grid_zero, grid, grid_avg
from .rad import rad
from .simple import simple_zero, simple
from .zero import zero

RS = [10, 15, 10, 40, 5, 10, 10]
MS = [grid_zero, grid, grid_avg, rad, simple_zero, simple, zero]


def random_one():
    # noinspection PyTypeChecker
    selected_method = np.random.choice(MS, p=np.array(RS) / np.sum(RS))
    return selected_method()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from imgutils.detect import detection_visualize

    canvas, bboxes = random_one()
    plt.imshow(detection_visualize(canvas, bboxes))
    plt.show()
