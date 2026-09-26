# auth_config.py
#
# Credential loading priority (highest → lowest):
#   1. Streamlit secrets  (.streamlit/secrets.toml → [auth] section)
#      Expects pre-computed PBKDF2-HMAC-SHA256 hashes in the format:
#        pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
#      Legacy unsalted SHA-256 hashes (64-char hex) are still accepted and
#      are transparently upgraded in memory to PBKDF2-HMAC-SHA256 on first
#      successful login.
#   2. Environment variable *_PASSWORD_HASH  (pre-computed PBKDF2-HMAC-SHA256
#      or legacy SHA-256 hex; legacy hashes are migrated in memory on first
#      successful login)
#   3. Environment variable *_PASSWORD  (plaintext, hashed at runtime via
#      PBKDF2-HMAC-SHA256 with a random 128-bit salt; never stored on disk)
#      — set via .env (gitignored) or the host OS environment
#
# See .streamlit/secrets.toml.example for the recommended approach.
# No plaintext password is ever stored in tracked source files.

import os
import hashlib
import hmac
import secrets
from typing import Optional, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_ALGORITHM = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 100_000
SALT_BYTES = 16  # 16 bytes = 128 bits entropy -> 32 hex chars

# In-memory store for PBKDF2-HMAC-SHA256 hashes that were upgraded from
# legacy SHA-256 during this runtime session. No plaintext password is ever
# stored here — only the newly derived PBKDF2 hash replaces the old one.
_MIGRATED_HASHES: Dict[str, str] = {}


def hash_password(raw_password: str, salt: Optional[str] = None, iterations: int = DEFAULT_ITERATIONS) -> str:
    """
    Derive a secure salted password hash using PBKDF2-HMAC-SHA256.

    Format: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
    """
    if not raw_password:
        return ""
    if salt is None:
        salt = secrets.token_hex(SALT_BYTES)

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        raw_password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    return f"{DEFAULT_ALGORITHM}${iterations}${salt}${derived.hex()}"


def _hash_password(raw_password: str) -> str:
    """Backward-compatible alias for hashing passwords using PBKDF2-HMAC-SHA256."""
    return hash_password(raw_password)


def is_valid_hash_format(hash_str: str) -> bool:
    """
    Check if a string adheres to the pbkdf2_sha256$<iterations>$<salt>$<hash> format.
    """
    if not hash_str or not isinstance(hash_str, str):
        return False
    parts = hash_str.split("$")
    if len(parts) != 4:
        return False
    algo, iter_str, salt, hash_val = parts
    if algo != DEFAULT_ALGORITHM:
        return False
    try:
        iterations = int(iter_str)
        if iterations <= 0:
            return False
    except ValueError:
        return False
    if not salt or not hash_val:
        return False
    try:
        int(salt, 16)
        int(hash_val, 16)
    except ValueError:
        return False
    return True


def is_legacy_hash(hash_str: str) -> bool:
    """
    Check if a string is a legacy unsalted SHA-256 hex string (64 hex characters).
    """
    if not hash_str or not isinstance(hash_str, str):
        return False
    if len(hash_str) != 64 or "$" in hash_str:
        return False
    try:
        int(hash_str, 16)
        return True
    except ValueError:
        return False


def needs_rehash(stored_hash: str, desired_iterations: int = DEFAULT_ITERATIONS) -> bool:
    """
    Check if a stored hash needs to be migrated/upgraded (e.g. legacy SHA-256 or different iteration count).
    """
    if not stored_hash or not isinstance(stored_hash, str):
        return False
    if is_legacy_hash(stored_hash):
        return True
    parts = stored_hash.split("$")
    if len(parts) == 4 and parts[0] == DEFAULT_ALGORITHM:
        try:
            return int(parts[1]) != desired_iterations
        except ValueError:
            return True
    return True


def verify_password(raw_password: str, stored_hash: str) -> bool:
    """
    Verify raw_password against stored_hash safely in constant time.
    Supports both standard PBKDF2-HMAC-SHA256 and legacy SHA-256 (for migration).
    Never logs passwords or hashes.
    """
    if not raw_password or not stored_hash:
        return False
    raw_str = str(raw_password)
    stored_str = str(stored_hash).strip()

    # 1. PBKDF2 format
    if is_valid_hash_format(stored_str):
        parts = stored_str.split("$")
        _, iter_str, salt, expected_hash = parts
        try:
            iterations = int(iter_str)
            derived = hashlib.pbkdf2_hmac(
                "sha256",
                raw_str.encode("utf-8"),
                salt.encode("utf-8"),
                iterations,
            )
            return hmac.compare_digest(derived.hex(), expected_hash)
        except Exception:
            return False

    # 2. Legacy SHA-256 hash
    if is_legacy_hash(stored_str):
        computed = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
        return hmac.compare_digest(computed.lower(), stored_str.lower())

    return False


def _get_hash_from_secrets(key: str) -> str:
    """
    Try to read a pre-hashed password from Streamlit secrets.

    Returns an empty string if:
    - streamlit is not installed
    - secrets.toml is absent or the key is missing
    - we are running outside a Streamlit context (e.g. during pytest)

    Never raises; callers fall back to env-var lookup on empty return.
    """
    try:
        import streamlit as st  # noqa: PLC0415
        # st.secrets raises FileNotFoundError when secrets.toml is missing
        auth_section = st.secrets.get("auth", {})
        return auth_section.get(key, "") or ""
    except Exception:  # noqa: BLE001  (FileNotFoundError, AttributeError, etc.)
        return ""


def _resolve_hash(secrets_key: str, env_hash_key: str, env_plain_key: str, username: str = "") -> str:
    """
    Return the password hash for one user, trying sources in priority order.

    Priority:
      0. In-memory migrated hash (active session upgrade)
      1. Streamlit secrets  [auth] <secrets_key>
      2. env var            <env_hash_key>   (pre-computed hash, PBKDF2 or legacy)
      3. env var            <env_plain_key>  (plaintext, hashed at runtime via PBKDF2)
      4. empty string       → login rejected, no crash
    """
    if username and username in _MIGRATED_HASHES:
        return _MIGRATED_HASHES[username]

    val = _get_hash_from_secrets(secrets_key) or os.getenv(env_hash_key, "")
    if val:
        return val

    plain = os.getenv(env_plain_key, "")
    if plain:
        return hash_password(plain)
    return ""


def get_credentials() -> dict:
    """
    Build and return the credentials dictionary.

    Called fresh on every login attempt so that runtime changes to
    environment variables (e.g. in tests) are always picked up.
    """
    p1_hash    = _resolve_hash(
        "pharmacist1_password_hash",
        "PHARMACIST1_PASSWORD_HASH",
        "PHARMACIST1_PASSWORD",
        username="pharmacist1",
    )
    p2_hash    = _resolve_hash(
        "pharmacist2_password_hash",
        "PHARMACIST2_PASSWORD_HASH",
        "PHARMACIST2_PASSWORD",
        username="pharmacist2",
    )
    admin_hash = _resolve_hash(
        "admin_password_hash",
        "ADMIN_PASSWORD_HASH",
        "ADMIN_PASSWORD",
        username="admin",
    )

    return {
        "usernames": {
            "pharmacist1": {
                "name":     "Sarah Johnson",
                "password": p1_hash,
                "branch":   "Central Pharmacy",
            },
            "pharmacist2": {
                "name":     "James Patel",
                "password": p2_hash,
                "branch":   "North Branch",
            },
            "admin": {
                "name":     "Admin User",
                "password": admin_hash,
                "branch":   "All branches",
            },
        }
    }


class _CredentialsProxy(dict):
    """
    Lazy dict proxy — re-evaluates get_credentials() on every access so
    that environment-variable changes made after import are reflected
    immediately (important for test isolation).
    """

    def __getitem__(self, item):
        return get_credentials()[item]

    def get(self, item, default=None):
        return get_credentials().get(item, default)

    def __contains__(self, item):
        return item in get_credentials()


# Module-level singleton used by app.py and test_edge_cases.py
CREDENTIALS = _CredentialsProxy()


def authenticate_user(username: str, password: str):
    """
    Authenticate a user by verifying their password against configured hashes.

    Returns the user info dictionary if credentials are valid, or None otherwise.
    Safe against None, empty strings, and nonexistent usernames.
    Transparently upgrades legacy SHA-256 hashes in memory upon successful verification.
    """
    if not username or not password:
        return None
    users = get_credentials().get("usernames", {})
    uname = str(username).strip()
    user_info = users.get(uname)
    if not user_info:
        return None
    stored_hash = user_info.get("password", "")
    if not stored_hash:
        return None
    if verify_password(str(password), stored_hash):
        if needs_rehash(stored_hash):
            new_hash = hash_password(str(password))
            _MIGRATED_HASHES[uname] = new_hash
            user_info["password"] = new_hash
        return user_info
    return None
