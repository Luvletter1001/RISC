import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_p0148_support_interventions.py")


class P0148SupportInterventionSelectionTest(unittest.TestCase):
    def test_model_code_root_is_prioritized_for_custom_model_imports(self):
        spec = importlib.util.spec_from_file_location("p0148_code_root", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_path = list(sys.path)
        try:
            with tempfile.TemporaryDirectory() as code_root:
                self.assertTrue(
                    hasattr(module, "prioritize_model_code_root"),
                    "runner must expose model-code-root isolation",
                )
                module.prioritize_model_code_root(code_root)
                self.assertEqual(str(Path(code_root).resolve()), sys.path[0])
        finally:
            sys.path[:] = original_path

    def test_direct_script_import_adds_repository_root_to_sys_path(self):
        repo_root = str(SCRIPT.parents[2])
        original_path = list(sys.path)
        sys.path[:] = [
            item
            for item in sys.path
            if os.path.abspath(item or os.getcwd()) != repo_root
        ]
        try:
            spec = importlib.util.spec_from_file_location("p0148_direct", SCRIPT)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertIn(repo_root, sys.path)
        finally:
            sys.path[:] = original_path

    def test_selects_only_requested_original_rotation_views(self):
        self.assertTrue(SCRIPT.exists(), f"missing runner: {SCRIPT}")
        spec = importlib.util.spec_from_file_location("p0148_runner", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertTrue(
            module.is_selected_view(
                "P0148__1024__651___0_rot090_A_original.png",
                angles={0, 90, 180, 270},
                filename_token="_A_original",
            )
        )
        self.assertFalse(
            module.is_selected_view(
                "P0148__1024__651___0_rot090_C_background_only.png",
                angles={0, 90, 180, 270},
                filename_token="_A_original",
            )
        )
        self.assertFalse(
            module.is_selected_view(
                "P0148__1024__651___0_rot030_A_original.png",
                angles={0, 90, 180, 270},
                filename_token="_A_original",
            )
        )


if __name__ == "__main__":
    unittest.main()
