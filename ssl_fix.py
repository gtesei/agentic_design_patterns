"""
SSL Certificate Fix for Corporate Networks

This module provides a workaround for SSL certificate verification issues
in corporate environments with SSL inspection/man-in-the-middle proxies.

Usage:
    Import this module at the top of your script (before any HTTP libraries):

    import ssl_fix  # This automatically applies the SSL bypass

    # Or call it explicitly:
    import ssl_fix
    ssl_fix.apply_ssl_bypass()

Note:
    This disables SSL certificate verification, which is acceptable for
    development in corporate environments but should NOT be used in production.
"""

import ssl
import warnings


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
    # Disable default SSL verification
    ssl._create_default_https_context = ssl._create_unverified_context

    # Suppress SSL warnings
    warnings.filterwarnings('ignore', category=Warning)

    # Patch httpx to disable SSL verification
    # httpx is used by many modern libraries (OpenAI, Hugging Face Hub, etc.)
    try:
        import httpx

        # Store original __init__ methods
        _original_client_init = httpx.Client.__init__
        _original_async_client_init = httpx.AsyncClient.__init__

        def _patched_client_init(self, *args, **kwargs):
            kwargs['verify'] = False
            _original_client_init(self, *args, **kwargs)

        def _patched_async_client_init(self, *args, **kwargs):
            kwargs['verify'] = False
            _original_async_client_init(self, *args, **kwargs)

        # Apply patches
        httpx.Client.__init__ = _patched_client_init
        httpx.AsyncClient.__init__ = _patched_async_client_init

    except ImportError:
        # httpx not installed, skip patching
        pass


# Automatically apply SSL bypass when module is imported
apply_ssl_bypass()
