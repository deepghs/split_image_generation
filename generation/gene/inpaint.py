"""
Image inpainting module for repairing unpasted regions in PIL images.

This module provides functionality to inpaint (repair) regions of an image that have not been
pasted over, using OpenCV's inpainting algorithms. It's particularly useful for filling in
gaps or unwanted areas in composite images where certain regions need to be seamlessly filled.

The module uses OpenCV's TELEA or NS inpainting algorithms to intelligently fill in masked
regions based on surrounding pixel information.
"""

import cv2
import numpy as np
from PIL import Image


def inpaint_unpasted_regions(pil_image, pasted_positions, pasted_images, inpaint_radius=3):
    """
    Perform inpainting on regions of a PIL image that have not been pasted over.

    This function takes a PIL image that has had other images pasted onto it and performs
    inpainting on the remaining unpasted regions. It creates a mask based on the pasted
    positions and images, then uses OpenCV's inpainting algorithm to fill in the gaps.

    :param pil_image: The PIL Image object that has already had images pasted onto it
    :type pil_image: PIL.Image.Image
    :param pasted_positions: List of (x, y) coordinates where images were pasted
    :type pasted_positions: list[tuple[int, int]]
    :param pasted_images: List of PIL Image objects that were pasted onto the base image
    :type pasted_images: list[PIL.Image.Image]
    :param inpaint_radius: Radius parameter for the inpainting algorithm, controls the
                          neighborhood size for inpainting
    :type inpaint_radius: int

    :return: The inpainted PIL Image with unpasted regions filled
    :rtype: PIL.Image.Image

    Example::
        >>> from PIL import Image
        >>> base_img = Image.new('RGB', (200, 200), 'white')
        >>> patch1 = Image.new('RGB', (50, 50), 'red')
        >>> patch2 = Image.new('RGB', (30, 30), 'blue')
        >>> base_img.paste(patch1, (10, 10))
        >>> base_img.paste(patch2, (100, 100))
        >>> positions = [(10, 10), (100, 100)]
        >>> patches = [patch1, patch2]
        >>> result = inpaint_unpasted_regions(base_img, positions, patches)
    """
    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Create mask - mark which regions need to be repaired
    mask = create_inpaint_mask(pil_image, pasted_positions, pasted_images)

    # Execute inpainting
    inpainted = cv2.inpaint(cv_image, mask, inpaint_radius, cv2.INPAINT_TELEA)
    # Alternative: cv2.INPAINT_NS

    # Convert back to PIL format
    result_pil = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))

    return result_pil


def create_inpaint_mask(pil_image, pasted_positions, pasted_images):
    """
    Create an inpainting mask that marks regions needing repair.

    This function generates a binary mask where white pixels (255) indicate regions
    that should be inpainted, and black pixels (0) indicate regions that should be
    preserved (i.e., areas where images have been pasted).

    :param pil_image: The base PIL Image object to create a mask for
    :type pil_image: PIL.Image.Image
    :param pasted_positions: List of (x, y) coordinates where images were pasted
    :type pasted_positions: list[tuple[int, int]]
    :param pasted_images: List of PIL Image objects that were pasted onto the base image
    :type pasted_images: list[PIL.Image.Image]

    :return: Binary mask array where 255 indicates regions to inpaint and 0 indicates
             regions to preserve
    :rtype: numpy.ndarray

    Example::
        >>> from PIL import Image
        >>> import numpy as np
        >>> base_img = Image.new('RGB', (100, 100), 'white')
        >>> patch = Image.new('RGB', (20, 20), 'red')
        >>> positions = [(10, 10)]
        >>> patches = [patch]
        >>> mask = create_inpaint_mask(base_img, positions, patches)
        >>> # mask will be all 255 except for the 20x20 region at (10,10) which will be 0
    """
    width, height = pil_image.size
    mask = np.ones((height, width), dtype=np.uint8) * 255  # Full white mask

    # Mark already pasted regions in mask as black (no need to repair)
    for pos, img in zip(pasted_positions, pasted_images):
        x, y = pos
        img_width, img_height = img.size

        # Ensure boundaries are not exceeded
        x_end = min(x + img_width, width)
        y_end = min(y + img_height, height)

        # Set pasted regions to 0 (black, no repair needed) in mask
        mask[y:y_end, x:x_end] = 0

    return mask
