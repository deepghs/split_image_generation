import os

LOCAL_DIR = os.environ.get('LOCAL_DIR')
ENABLE_LOCAL_MODE = bool(LOCAL_DIR)


def repo_map(repo_id: str) -> str:
    return os.path.join(LOCAL_DIR, repo_id)
