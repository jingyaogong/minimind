import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from scripts.rm_load_check import REPO_ROOT, TRAINER_RM_PATH, main


class PathResolutionTests(unittest.TestCase):
    def test_default_path_is_the_repo_sibling(self):
        # trainer/train_grpo.py 的默认值 ``../../internlm2-1_8b-reward`` 从 trainer/
        # 解析出来就是仓库的同级目录，检查脚本必须指向同一个位置。
        self.assertEqual(TRAINER_RM_PATH.name, "internlm2-1_8b-reward")
        self.assertEqual(TRAINER_RM_PATH.parent, REPO_ROOT.parent)

    def test_missing_directory_fails_before_importing_torch(self):
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            code = main(["--model-path", "/nonexistent/internlm2-1_8b-reward"])
        self.assertEqual(code, 2)
        self.assertIn("/nonexistent/internlm2-1_8b-reward", out.getvalue())
        self.assertIn("目录不存在", err.getvalue())


if __name__ == "__main__":
    unittest.main()
