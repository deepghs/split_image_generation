import os.path
import random
from typing import Union, Tuple

import pandas as pd
from PIL import Image
from hbutils.system import TemporaryDirectory
from hfutils.index import tar_file_download, hf_tar_file_download
from huggingface_hub import hf_hub_download
from imgutils.utils import ts_lru_cache
from torchvision.transforms import RandomResizedCrop

from generation.site.map import ENABLE_LOCAL_MODE, repo_map


@ts_lru_cache()
def _global_df() -> pd.DataFrame:
    """
    Load the global dataframe containing information about background images.

    :return: The global dataframe containing information about background images.
    :rtype: pd.DataFrame
    """
    return pd.read_csv(hf_hub_download(
        repo_id='deepghs/anime-bg',
        repo_type='dataset',
        filename='images.csv'
    ))


def get_bg_image():
    with TemporaryDirectory() as td:
        # logging.info(f'Loading image {name!r} ...')
        df = _global_df()
        row = df.sample(1).to_dict('records')[0]
        image_repo_id = 'deepghs/anime-bg'
        image_archive = f"images/{row['archive']}"
        image_filename = row['filename']
        dst_image_file = os.path.join(td, os.path.basename(row['filename']))
        # logging.info(f'Downloading image to {dst_image_file!r} ...')
        if ENABLE_LOCAL_MODE:
            tar_file_download(
                archive_file=os.path.join(repo_map(image_repo_id), image_archive),
                file_in_archive=image_filename,
                local_file=dst_image_file,
            )
        else:
            hf_tar_file_download(
                repo_id=image_repo_id,
                archive_in_repo=image_archive,
                file_in_archive=image_filename,
                local_file=dst_image_file,
            )

        img = Image.open(dst_image_file)
        img.load()

        return img


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
        )(get_bg_image())
        is_pure = False
    assert canvas.size == (canvas_width, canvas_height)
    if is_check:
        return canvas, is_pure
    else:
        return canvas


if __name__ == '__main__':
    df = _global_df()
    print(df)
    print(df.sample(1).to_dict('records')[0])
    # print(dict(zip(df.columns, df.sample(1))))
