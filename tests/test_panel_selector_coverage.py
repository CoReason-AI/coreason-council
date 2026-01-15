# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from unittest.mock import MagicMock, patch

import yaml
from coreason_council.core.panel_selector import PanelSelector


def test_presets_file_not_found() -> None:
    """Test behavior when the presets file does not exist."""
    # Patch settings to point to a non-existent file
    with patch("coreason_council.settings.settings.presets_file", "non_existent.yaml"):
        # We also need to ensure the fallback logic in _load_presets doesn't find it
        # The fallback looks relative to coreason_council module.
        # We can patch Path.exists to return False
        with patch("pathlib.Path.exists", return_value=False):
            selector = PanelSelector()
            # Should have empty presets
            assert selector.presets == {}

            # Select panel should default to fallback
            _, personas = selector.select_panel("any query")
            assert len(personas) == 1
            assert personas[0].name == "Generalist"


def test_presets_file_invalid_yaml() -> None:
    """Test behavior when presets file contains invalid YAML."""
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = "invalid: yaml: ["
        # Mock yaml.safe_load to raise exception
        with patch("yaml.safe_load", side_effect=yaml.YAMLError("Invalid YAML")):
            selector = PanelSelector()
            assert selector.presets == {}


def test_presets_file_fallback_logic() -> None:
    """Test the fallback logic when relative path doesn't exist but module resource does."""
    # If we set settings.presets_file to "garbage", the code attempts fallback.
    # The fallback path is constructed from `coreason_council.__file__`.
    # In the test environment, that file exists (we created it).
    # So `possible_path.exists()` should return True (if we don't mock exists globally).
    # And then it should load successfully.

    with patch("coreason_council.settings.settings.presets_file", "garbage_not_exist.yaml"):
        # We expect it to find the fallback file
        selector = PanelSelector()
        assert "medical" in selector.presets
        assert len(selector.presets["medical"]) > 0


def test_missing_capabilities_in_yaml() -> None:
    """Test that missing capabilities field is handled gracefully."""
    with patch("builtins.open", new_callable=MagicMock):
        pass

    with patch("yaml.safe_load", return_value={"test_cat": [{"name": "TestBot", "system_prompt": "Prompt"}]}):
        # Also need to ensure path exists check passes so it attempts to load
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open"):
                selector = PanelSelector()
                assert len(selector.presets["test_cat"]) == 1
                assert selector.presets["test_cat"][0].capabilities == []
                # Ensure PersonaType enum is not broken
                assert isinstance(selector.presets["test_cat"][0].capabilities, list)
