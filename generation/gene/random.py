import numpy as np

from .grid import grid_zero, grid, grid_avg, grid_with_inpaint, grid_avg_with_inpaint
from .rad import rad, rad_with_inpaint
from .simple import simple_zero, simple, simple_with_inpaint
from .zero import zero

_METHODS = {}

RS = [10, 15, 10, 40, 5, 10, 10]
MS = [grid_zero, grid, grid_avg, rad, simple_zero, simple, zero]


def random_one():
    # noinspection PyTypeChecker
    selected_method = np.random.choice(MS, p=np.array(RS) / np.sum(RS))
    return selected_method()


_METHODS['random_one'] = random_one

RSI = [35, 35, 15, 15]
MSI = [simple_with_inpaint, rad_with_inpaint, grid_with_inpaint, grid_avg_with_inpaint]


def random_inp():
    selected_method = np.random.choice(MSI, p=np.array(RSI) / np.sum(RSI))
    return selected_method()


_METHODS['random_inp'] = random_inp


def random_c(method: str = 'one'):
    return _METHODS[f'random_{method}']()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from imgutils.detect import detection_visualize

    canvas, bboxes = random_c('inp')
    plt.imshow(detection_visualize(canvas, bboxes))
    plt.show()
