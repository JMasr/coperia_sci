import shutil
import tempfile
from pathlib import Path

import pytest


class TestExperimentShould:
    @classmethod
    def setup_class(cls):
        cls.str_path_temp_folder = Path(tempfile.mkdtemp())

    @classmethod
    def teardown_class(cls):
        # Remove the temporary files after testing
        shutil.rmtree(cls.str_path_temp_folder)


if __name__ == "__main__":
    # Run all tests in the module
    pytest.main()
