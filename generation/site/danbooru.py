import os
from functools import lru_cache
from typing import Dict, Tuple, Optional

from hfutils.index import hf_tar_list_files, tar_list_files

from .map import ENABLE_LOCAL_MODE, repo_map

_DB2023 = 'KBlueLeaf/danbooru2023-webp-4Mpixel'
_DB2023_IDX = 'deepghs/danbooru2023-webp-4Mpixel_index'

_DB_N = 'deepghs/danbooru_newest-webp-4Mpixel-all'


@lru_cache()
def _get_idx_by_block_file_from_2023(block_file: str) -> Dict[int, Tuple[str, str, str, str]]:
    if ENABLE_LOCAL_MODE:
        lst_files = tar_list_files(
            archive_file=os.path.join(repo_map(_DB2023), block_file),
            idx_file=os.path.join(repo_map(_DB2023_IDX), os.path.splitext(block_file)[0] + '.json'),
        )
    else:
        lst_files = hf_tar_list_files(
            repo_id=_DB2023,
            idx_repo_id=_DB2023_IDX,
            repo_type='dataset',
            archive_in_repo=block_file,
        )
    return {
        int(os.path.splitext(os.path.basename(file))[0]): (_DB2023, _DB2023_IDX, block_file, file)
        for file in lst_files
    }


@lru_cache()
def _get_idx_by_block_file_from_newest(block_file: str) -> Dict[int, Tuple[str, str, str, str]]:
    if ENABLE_LOCAL_MODE:
        lst_files = tar_list_files(
            archive_file=os.path.join(repo_map(_DB_N), block_file),
            idx_file=os.path.join(repo_map(_DB_N), os.path.splitext(block_file)[0] + '.json'),
        )
    else:
        lst_files = hf_tar_list_files(
            repo_id=_DB_N,
            idx_repo_id=_DB_N,
            repo_type='dataset',
            archive_in_repo=block_file,
        )
    return {
        int(os.path.splitext(os.path.basename(file))[0]): (_DB_N, _DB_N, block_file, file)
        for file in lst_files
    }


@lru_cache(maxsize=10000)
def _get_idx_by_block_id(src_block_id) -> Dict[int, Tuple[str, str, str, str]]:
    assert 0 <= src_block_id <= 999
    retval = {
        **_get_idx_by_block_file_from_2023(f'images/data-0{src_block_id:03d}.tar'),
        **_get_idx_by_block_file_from_2023(f'images/data-1{src_block_id:03d}.tar'),
        **_get_idx_by_block_file_from_2023(f'images/data-2{src_block_id:03d}.tar'),
        **_get_idx_by_block_file_from_2023(f'updates/20240319/data-{src_block_id % 10}.tar'),
        **_get_idx_by_block_file_from_2023(f'updates/20240522/data-{src_block_id % 10}.tar'),
        **_get_idx_by_block_file_from_newest(f'images/0{src_block_id:03d}.tar'),
    }
    retval = {id_: info for id_, info in retval.items() if id_ % 1000 == src_block_id}
    return retval


def _get_archive_and_filename_by_id(src_image_id) -> Optional[Tuple[str, str, str, str]]:
    val = _get_idx_by_block_id(src_image_id % 1000)
    if src_image_id in val:
        return val[src_image_id]
    else:
        return None


def _get_danbooru_pos(src_image_id) -> Optional[Tuple[str, str, str, str]]:
    return _get_archive_and_filename_by_id(src_image_id)


if __name__ == '__main__':
    print(_get_danbooru_pos(
        src_image_id=5777777
    ))
