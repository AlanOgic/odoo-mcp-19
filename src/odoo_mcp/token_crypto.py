"""Minimal decrypt utility for registry secrets.

Ports the CLORAG crypto contract (utils/token_encryption.py) without any
code dependency: Fernet key derived with PBKDF2HMAC-SHA256 (480 000
iterations, 32-byte key) from TOKEN_ENCRYPTION_KEY, salt read from the
``.token_salt`` file next to the registry database. Both containers share
the same secret and salt through the shared data volume.
"""

from __future__ import annotations

import base64
import json
import os
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# MUST match clorag utils/token_encryption.py
_PBKDF2_ITERATIONS = 480_000


def _read_encryption_password() -> str:
    """Read TOKEN_ENCRYPTION_KEY from env or the _FILE Docker-secret path."""
    value = os.environ.get("TOKEN_ENCRYPTION_KEY")
    if value:
        return value.strip()
    file_path = os.environ.get("TOKEN_ENCRYPTION_KEY_FILE")
    if file_path and Path(file_path).is_file():
        return Path(file_path).read_text().strip()
    raise RuntimeError(
        "TOKEN_ENCRYPTION_KEY (or TOKEN_ENCRYPTION_KEY_FILE) is required to"
        " decrypt registry credentials."
    )


@lru_cache(maxsize=4)
def _fernet(salt: bytes, password: str) -> Fernet:
    # PBKDF2 at 480k iterations costs ~0.1-0.3s — derive once per (salt, password).
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=_PBKDF2_ITERATIONS,
    )
    return Fernet(base64.urlsafe_b64encode(kdf.derive(password.encode())))


def decrypt_secret(encrypted: str, db_path: Path) -> dict[str, object]:
    """Decrypt a registry ``encrypted_secret`` value.

    Args:
        encrypted: Fernet ciphertext (as stored by CLORAG).
        db_path: Path to the registry database; the salt lives next to it.

    Returns:
        The decrypted JSON payload (e.g. {"api_key": "..."}).

    Raises:
        RuntimeError: On missing salt/key or key mismatch.
    """
    salt_path = db_path.parent / ".token_salt"
    if not salt_path.is_file():
        raise RuntimeError(f"Registry salt file not found: {salt_path}")
    fernet = _fernet(salt_path.read_bytes(), _read_encryption_password())
    try:
        decrypted = fernet.decrypt(encrypted.encode())
    except InvalidToken as exc:
        raise RuntimeError(
            "TOKEN_ENCRYPTION_KEY mismatch with registry — the secret must"
            " hold the same value as the CLORAG one."
        ) from exc
    result = json.loads(decrypted)
    if not isinstance(result, dict):
        raise RuntimeError("Registry secret payload is not a JSON object")
    return result
