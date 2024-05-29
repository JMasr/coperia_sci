import os
import shutil

import pytest

from models.model_object import ModelBuilder, SUPPORTED_MODELS
from test import ROOT_PATH


class TestModelBuilderShould:
    @classmethod
    def setup_class(cls):
        cls.str_path_temp_folder = os.path.join(ROOT_PATH, "test", "temp_folder")
        os.makedirs(cls.str_path_temp_folder, exist_ok=True)

        cls.default_model = SUPPORTED_MODELS

    @classmethod
    def teardown_class(cls):
        # Remove the temporary files after testing
        shutil.rmtree(cls.str_path_temp_folder)

    @pytest.mark.parametrize("model_name", SUPPORTED_MODELS.keys())
    def test_valid_build_a_model(self, model_name):
        # Arrange
        model_builder = ModelBuilder(
            name=model_name,
            path_to_model=self.str_path_temp_folder,
        )

        # Act
        model_builder.build_model()

        # Assert
        assert model_builder.model is not None
        assert isinstance(model_builder.model, self.default_model[model_name].__class__)

    @pytest.mark.parametrize("model_name", ["Gemini.v50", 1231, {}])
    def test_invalid_build_a_model(self, model_name):
        with pytest.raises(ValueError):
            model_builder = ModelBuilder(
                name=model_name,
                path_to_model=self.str_path_temp_folder,
            )
            model_builder.build_model()

    @pytest.mark.parametrize("model_name", SUPPORTED_MODELS.keys())
    def test_valid_save_model(self, model_name):
        # Arrange
        model_builder = ModelBuilder(
            name=model_name,
            path_to_model=self.str_path_temp_folder,
        )
        model_builder.build_model()
        # Act
        model_builder.save_as_a_serialized_object()
        # Assert
        assert os.path.exists(os.path.join(self.str_path_temp_folder, f"{model_name}.pkl"))

    def test_invalid_save_model(self):
        # Arrange
        model_builder = ModelBuilder(
            name="LogisticRegression",
            path_to_model=self.str_path_temp_folder,
        )

        with pytest.raises(IOError):
            model_builder.save_as_a_serialized_object(path_to_save="invalid_path")

    @pytest.mark.parametrize("model_name", SUPPORTED_MODELS.keys())
    def test_valid_load_model(self, model_name):
        # Arrange
        model_builder = ModelBuilder(
            name=model_name,
            path_to_model=self.str_path_temp_folder,
        )
        model_builder.build_model()
        model_builder.save_as_a_serialized_object()

        # Act
        model_builder.load_model_from_a_serialized_object()
        # Assert
        assert model_builder.model is not None
        assert isinstance(model_builder.model, self.default_model[model_name].__class__)

    def test_invalid_load_model(self):
        # Arrange
        model_builder = ModelBuilder(
            name="LogisticRegression",
            path_to_model=self.str_path_temp_folder,
        )

        with pytest.raises(IOError):
            model_builder.load_model_from_a_serialized_object(path_to_load="invalid_path")


if __name__ == "__main__":
    # Run all tests in the module
    pytest.main()
