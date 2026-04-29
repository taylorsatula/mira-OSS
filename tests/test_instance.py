"""Tests for MIRA instance identity derivation."""
import os
import pytest
from unittest.mock import patch


class TestInstanceIdentity:
    """Test instance name derivation from environment."""

    def test_default_instance_name(self):
        with patch.dict(os.environ, {}, clear=True):
            from utils.instance import _read_instance_name
            assert _read_instance_name() == "default"

    def test_custom_instance_name(self):
        with patch.dict(os.environ, {"MIRA_INSTANCE": "kristen"}):
            from utils.instance import _read_instance_name
            assert _read_instance_name() == "kristen"

    def test_instance_name_stripped_and_lowered(self):
        with patch.dict(os.environ, {"MIRA_INSTANCE": "  Kristen  "}):
            from utils.instance import _read_instance_name
            assert _read_instance_name() == "kristen"


class TestVaultPrefix:
    def test_default_vault_prefix(self):
        from utils.instance import vault_prefix
        with patch("utils.instance.INSTANCE_NAME", "default"):
            assert vault_prefix() == "mira"

    def test_custom_vault_prefix(self):
        from utils.instance import vault_prefix
        with patch("utils.instance.INSTANCE_NAME", "kristen"):
            assert vault_prefix() == "mira-kristen"


class TestDatabaseName:
    def test_default_database_name(self):
        from utils.instance import database_name
        with patch("utils.instance.INSTANCE_NAME", "default"):
            assert database_name() == "mira_service"

    def test_custom_database_name(self):
        from utils.instance import database_name
        with patch("utils.instance.INSTANCE_NAME", "kristen"):
            assert database_name() == "mira_service_kristen"


class TestDataDir:
    def test_default_data_dir(self):
        from utils.instance import user_data_base
        with patch("utils.instance.INSTANCE_NAME", "default"):
            from pathlib import Path
            assert user_data_base() == Path("data/users")

    def test_custom_data_dir(self):
        from utils.instance import user_data_base
        with patch("utils.instance.INSTANCE_NAME", "kristen"):
            from pathlib import Path
            assert user_data_base() == Path("data-kristen/users")


class TestValkeyPrefix:
    def test_default_valkey_prefix(self):
        from utils.instance import valkey_prefix
        with patch("utils.instance.INSTANCE_NAME", "default"):
            assert valkey_prefix() == ""

    def test_custom_valkey_prefix(self):
        from utils.instance import valkey_prefix
        with patch("utils.instance.INSTANCE_NAME", "kristen"):
            assert valkey_prefix() == "kristen:"
