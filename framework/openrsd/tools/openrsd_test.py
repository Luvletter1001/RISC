#!/usr/bin/env python3
from openrsd_env import preload_installed_mmengine

preload_installed_mmengine()

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmrotate.utils import register_all_modules as register_all_modules_mmrotate

register_all_modules_mmdet(init_default_scope=False)
register_all_modules_mmrotate(init_default_scope=False)

from test import main


if __name__ == '__main__':
    main()
