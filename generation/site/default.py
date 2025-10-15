import os
from functools import lru_cache
from typing import Tuple, Dict, Optional

from hfutils.index import hf_tar_list_files, tar_list_files

from .map import ENABLE_LOCAL_MODE, repo_map


@lru_cache(maxsize=10000)
def _get_idx_by_block_file_from_site(repo_id: str, block_file: str) -> Dict[int, Tuple[str, str, str, str]]:
    if ENABLE_LOCAL_MODE:
        lst_files = tar_list_files(
            archive_file=os.path.join(repo_map(repo_id), block_file),
            idx_file=os.path.join(repo_map(repo_id), os.path.splitext(block_file)[0] + '.json'),
        )
    else:
        lst_files = hf_tar_list_files(
            repo_id=repo_id,
            idx_repo_id=repo_id,
            repo_type='dataset',
            archive_in_repo=block_file,
        )
    return {
        int(os.path.splitext(os.path.basename(file))[0]): (repo_id, repo_id, block_file, file)
        for file in lst_files
    }


@lru_cache(maxsize=10000)
def _get_idx_by_block_id(repo_id, src_block_id) -> Dict[int, Tuple[str, str, str, str]]:
    assert 0 <= src_block_id <= 999
    retval = {
        **_get_idx_by_block_file_from_site(repo_id, f'images/0{src_block_id:03d}.tar'),
    }
    retval = {id_: info for id_, info in retval.items() if id_ % 1000 == src_block_id}
    return retval


def _get_archive_and_filename_by_id(repo_id, src_image_id) -> Optional[Tuple[str, str, str, str]]:
    val = _get_idx_by_block_id(repo_id, src_image_id % 1000)
    if src_image_id in val:
        return val[src_image_id]
    else:
        return None


def _get_default_pos(repo_id, src_image_id) -> Optional[Tuple[str, str, str, str]]:
    return _get_archive_and_filename_by_id(repo_id, src_image_id)


if __name__ == '__main__':
    print(_get_default_pos(
        repo_id='deepghs/gelbooru-webp-4Mpixel',
        src_image_id=7777777
    ))
