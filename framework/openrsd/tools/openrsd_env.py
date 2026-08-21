import importlib
import os
import os.path as osp
import site
import sys


def _without_user_site(paths):
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        user_sites = {osp.abspath(user_site)}
    else:
        user_sites = {osp.abspath(path) for path in user_site}

    cleaned_paths = []
    cwd = osp.abspath(os.getcwd())
    for path_item in paths:
        abs_path = cwd if path_item == '' else osp.abspath(path_item)
        if abs_path in user_sites:
            continue
        cleaned_paths.append(path_item)
    return cleaned_paths


def preload_installed_mmengine():
    """Load site-packages mmengine before local packages are imported.

    The repository contains a partial mmengine checkout that shadows the
    installed package and misses modules such as mmengine.dist. Preloading the
    installed package keeps local mmdet/mmrotate/M_AD available while forcing
    all subsequent mmengine imports to use the complete dependency. User-site
    packages are also removed to avoid mixing ~/.local wheels with conda libs.
    """
    project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    original_path = _without_user_site(sys.path)
    cwd = osp.abspath(os.getcwd())

    filtered_path = []
    for path_item in original_path:
        abs_path = cwd if path_item == '' else osp.abspath(path_item)
        if abs_path == project_root or abs_path.startswith(project_root + osp.sep):
            continue
        filtered_path.append(path_item)

    sys.path = filtered_path
    try:
        importlib.import_module('mmengine')
    finally:
        sys.path = original_path
