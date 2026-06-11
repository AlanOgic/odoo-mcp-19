"""Unit tests for token_crypto — the ported CLORAG crypto contract."""

import pytest

from odoo_mcp.token_crypto import _fernet, decrypt_secret
from tests.conftest import TEST_ENCRYPTION_KEY, encrypt_with_contract


def test_decrypt_round_trip(users_db_seed):
    encrypted = encrypt_with_contract(
        {"api_key": "secret-123"}, users_db_seed.salt, TEST_ENCRYPTION_KEY
    )
    result = decrypt_secret(encrypted, db_path=users_db_seed.db_path)
    assert result == {"api_key": "secret-123"}


def test_decrypt_wrong_password(users_db_seed, monkeypatch):
    encrypted = encrypt_with_contract(
        {"api_key": "x"}, users_db_seed.salt, TEST_ENCRYPTION_KEY
    )
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", "wrong-password")
    _fernet.cache_clear()
    with pytest.raises(RuntimeError, match="mismatch"):
        decrypt_secret(encrypted, db_path=users_db_seed.db_path)
    _fernet.cache_clear()


def test_decrypt_missing_salt(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", TEST_ENCRYPTION_KEY)
    with pytest.raises(RuntimeError, match="salt"):
        decrypt_secret("whatever", db_path=tmp_path / "users.db")


def test_missing_key_env(users_db_seed, monkeypatch):
    encrypted = encrypt_with_contract(
        {"api_key": "x"}, users_db_seed.salt, TEST_ENCRYPTION_KEY
    )
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY")
    with pytest.raises(RuntimeError, match="TOKEN_ENCRYPTION_KEY"):
        decrypt_secret(encrypted, db_path=users_db_seed.db_path)


def test_key_file_fallback(users_db_seed, tmp_path, monkeypatch):
    encrypted = encrypt_with_contract(
        {"api_key": "from-file"}, users_db_seed.salt, TEST_ENCRYPTION_KEY
    )
    key_file = tmp_path / "key_secret"
    key_file.write_text(TEST_ENCRYPTION_KEY + "\n")
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY")
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY_FILE", str(key_file))
    result = decrypt_secret(encrypted, db_path=users_db_seed.db_path)
    assert result["api_key"] == "from-file"
