# auth_config.py
#
# Credential loading priority (highest → lowest):
#   1. Streamlit secrets  (.streamlit/secrets.toml → [auth] section)
#      Expects pre-computed SHA-256 hashes, e.g.:
#        pharmacist1_password_hash = "<hex>"
#   2. Environment variable *_PASSWORD_HASH  (pre-computed SHA-256 hex)
#   3. Environment variable *_PASSWORD  (plaintext, hashed at runtime)
#      — set via .env (gitignored) or the host OS environment
#
# See .streamlit/secrets.toml.example for the recommended approach.
# No plaintext password is ever stored in tracked source files.

import os
import hashlib

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _hash_password(raw_password: str) -> str:
    """Return the SHA-256 hex digest of *raw_password*, or '' if falsy."""
    if not raw_password:
        return ""
    return hashlib.sha256(raw_password.encode()).hexdigest()


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


def _resolve_hash(secrets_key: str, env_hash_key: str, env_plain_key: str) -> str:
    """
    Return the password hash for one user, trying sources in priority order.

    Priority:
      1. Streamlit secrets  [auth] <secrets_key>
      2. env var            <env_hash_key>   (pre-computed hash)
      3. env var            <env_plain_key>  (plaintext, hashed here)
      4. empty string       → login rejected, no crash
    """
    value = (
        _get_hash_from_secrets(secrets_key)
        or os.getenv(env_hash_key, "")
        or _hash_password(os.getenv(env_plain_key, ""))
    )
    return value


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
    )
    p2_hash    = _resolve_hash(
        "pharmacist2_password_hash",
        "PHARMACIST2_PASSWORD_HASH",
        "PHARMACIST2_PASSWORD",
    )
    admin_hash = _resolve_hash(
        "admin_password_hash",
        "ADMIN_PASSWORD_HASH",
        "ADMIN_PASSWORD",
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
    """
    if not username or not password:
        return None
    users = get_credentials().get("usernames", {})
    user_info = users.get(str(username).strip())
    if not user_info:
        return None
    stored_hash = user_info.get("password", "")
    if not stored_hash:
        return None
    if _hash_password(str(password)) == stored_hash:
        return user_info
    return None
