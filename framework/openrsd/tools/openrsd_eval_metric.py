#!/usr/bin/env python3
from openrsd_env import preload_installed_mmengine

preload_installed_mmengine()

from analysis_tools.eval_metric import main


if __name__ == '__main__':
    main()
