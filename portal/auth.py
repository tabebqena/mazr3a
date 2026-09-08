"""PBKDF2 password hashing + signed HttpOnly session cookies (stdlib only).

Passwords are stored in config/portal.conf as:

    pbkdf2$<iterations>$<salt_hex>$<hash_hex>

Session cookie value: <b64url(username)>.<exp_epoch>.<hmac_sha256_hex>  signed
with the portal SECRET_KEY. Verification is constant-time via
hmac.compare_digest.
"""
import base64
import binascii
import hashlib
import hmac
import secrets
import time

_ALGO = "pbkdf2"
_DEFAULT_ITERS = 600_000


def hash_password(password, iterations=_DEFAULT_ITERS, salt=None):
    """Return a `pbkdf2$<iters>$<salt_hex>$<hash_hex>` string for `password`."""
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "{}${}${}${}".format(_ALGO, iterations, salt.hex(), dk.hex())


def verify_password(password, stored):
    """Constant-time check of `password` against a stored hash string."""
    try:
        algo, iters_s, salt_hex, hash_hex = stored.split("$")
        if algo != _ALGO:
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"),
            binascii.unhexlify(salt_hex), int(iters_s),
        )
        return hmac.compare_digest(dk.hex(), hash_hex.lower())
    except Exception:
        return False


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64u(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_session(secret, username, ttl_s):
    """Create a signed cookie token valid for ttl_s seconds."""
    exp = int(time.time()) + int(ttl_s)
    msg = "{}.{}".format(_b64u(username.encode("utf-8")), exp)
    sig = hmac.new(secret.encode("utf-8"), msg.encode("utf-8"),
                   hashlib.sha256).hexdigest()
    return "{}.{}".format(msg, sig)


def read_session(secret, token):
    """Return the username for a valid, unexpired token, else None."""
    if not token or not secret:
        return None
    try:
        msg, sig = token.rsplit(".", 1)
        expect = hmac.new(secret.encode("utf-8"), msg.encode("utf-8"),
                          hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, sig):
            return None
        b64_name, exp_s = msg.split(".", 1)
        if int(exp_s) < time.time():
            return None
        return _unb64u(b64_name).decode("utf-8")
    except Exception:
        return None
