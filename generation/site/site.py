import os
from functools import partial
from typing import List

from PIL import Image
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.index import tar_file_download, hf_tar_file_download

from generation.utils import parallel_call
from .danbooru import _get_danbooru_pos
from .default import _get_default_pos
from .map import ENABLE_LOCAL_MODE, repo_map

_QUERY_DICT = {
    'danbooru': _get_danbooru_pos,
    'nozomi': partial(_get_default_pos, 'deepghs/nozomi_standalone-webp-4Mpixel'),
    'yandere': partial(_get_default_pos, 'deepghs/yande-webp-4Mpixel'),
}


def _get_resource_from_site(site_name: str, src_image_id: int):
    queryer = _QUERY_DICT.get(site_name) or partial(_get_default_pos, f'deepghs/{site_name}-webp-4Mpixel')
    return queryer(src_image_id)


def get_resource_from_site_with_raw_id(id_: str):
    site_name, num_id = id_.rsplit('_', maxsplit=1)
    num_id = int(num_id)
    return _get_resource_from_site(site_name, num_id)


def get_image_from_site(id_: str):
    with TemporaryDirectory() as td:
        # logging.info(f'Loading image {name!r} ...')
        image_repo_id, image_idx_repo_id, image_archive, image_filename = get_resource_from_site_with_raw_id(id_)
        dst_image_file = os.path.join(td, os.path.basename(image_filename))
        # logging.info(f'Downloading image to {dst_image_file!r} ...')
        if ENABLE_LOCAL_MODE:
            tar_file_download(
                archive_file=os.path.join(repo_map(image_repo_id), image_archive),
                idx_file=os.path.join(repo_map(image_idx_repo_id),
                                      os.path.splitext(image_archive)[0] + '.json'),
                file_in_archive=image_filename,
                local_file=dst_image_file,
            )
        else:
            hf_tar_file_download(
                repo_id=image_repo_id,
                idx_repo_id=image_idx_repo_id,
                archive_in_repo=image_archive,
                file_in_archive=image_filename,
                local_file=dst_image_file,
            )

        img = Image.open(dst_image_file)
        img.load()

        return img


def get_images_from_site(ids: List[str], max_workers: int = 16):
    retval = []

    def _fn(id_):
        retval.append(get_image_from_site(id_))

    parallel_call(
        iterable=ids,
        fn=_fn,
        desc=f'Getting {plural_word(len(ids), "image")}',
    )

    return retval


if __name__ == '__main__':
    # print(get_image_from_site('danbooru_5777777'))
    # print(get_image_from_site('gelbooru_7777777'))
    print(get_images_from_site(['danbooru_5777777', 'gelbooru_7777777']))
