"""Optional SSL bypass for local corporate-network troubleshooting.

This module no longer disables certificate verification on import. To opt in,
set ``AGENTIC_DISABLE_SSL=1`` or call ``apply_ssl_bypass()`` explicitly.
"""

import os
import ssl
import warnings

SSL_BYPASS_ENV_VAR = "AGENTIC_DISABLE_SSL"
_PATCHED = False


def ssl_bypass_requested(env_var: str = SSL_BYPASS_ENV_VAR) -> bool:
    """Return ``True`` when the SSL bypass was explicitly requested."""
    value = os.getenv(env_var, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def apply_ssl_bypass():
    """
    Apply SSL certificate verification bypass for corporate networks.

    This function:
    1. Disables SSL certificate verification for the default SSL context
    2. Patches httpx Client and AsyncClient to disable verification
    3. Suppresses SSL-related warnings

    This is necessary because corporate SSL inspection creates self-signed
    certificates that Python's SSL verification rejects.
    """
    global _PATCHED
    if _PATCHED:
        return

    def _unverified_context(*args, **kwargs) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    # Disable default SSL verification across the common stdlib entry points.
    ssl._create_default_https_context = ssl._create_unverified_context
    ssl.create_default_context = _unverified_context

    # Suppress SSL warnings
    warnings.filterwarnings("ignore", category=Warning)

    # Patch httpx to disable SSL verification
    # httpx is used by many modern libraries (OpenAI, Hugging Face Hub, etc.)
    try:
        import httpx
        import httpcore

        # Store original __init__ methods
        _original_client_init = httpx.Client.__init__
        _original_async_client_init = httpx.AsyncClient.__init__
        _original_create_ssl_context = httpx._config.create_ssl_context
        _original_default_ssl_context = httpcore._ssl.default_ssl_context

        def _patched_client_init(self, *args, **kwargs):
            kwargs["verify"] = False
            _original_client_init(self, *args, **kwargs)

        def _patched_async_client_init(self, *args, **kwargs):
            kwargs["verify"] = False
            _original_async_client_init(self, *args, **kwargs)

        def _patched_create_ssl_context(*args, **kwargs):
            return _unverified_context()

        def _patched_default_ssl_context():
            return _unverified_context()

        # Apply patches
        httpx.Client.__init__ = _patched_client_init
        httpx.AsyncClient.__init__ = _patched_async_client_init
        httpx._config.create_ssl_context = _patched_create_ssl_context
        httpcore._ssl.default_ssl_context = _patched_default_ssl_context

    except ImportError:
        # httpx not installed, skip patching
        pass
    finally:
        _PATCHED = True


if ssl_bypass_requested():
    apply_ssl_bypass()
