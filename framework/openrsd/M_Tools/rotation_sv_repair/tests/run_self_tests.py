#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]

TESTS = [
    'test_class_order', 'test_logit_bias', 'test_adapter_identity', 'test_paper_adapter_ckpt',
    'test_orbit_geometry', 'test_pseudo_label_queue', 'test_teacher_fusion',
]

def main():
    failed = 0
    for name in TESTS:
        mod = __import__(f'M_Tools.rotation_sv_repair.tests.{name}', fromlist=['*'])
        for attr in dir(mod):
            if attr.startswith('test_'):
                try:
                    getattr(mod, attr)()
                    print(f'OK {name}.{attr}')
                except Exception as exc:
                    print(f'FAIL {name}.{attr}: {exc}')
                    failed += 1
    return 1 if failed else 0

if __name__ == '__main__':
    sys.exit(main())
