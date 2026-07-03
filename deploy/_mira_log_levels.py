"""
MIRA custom log level registration.

Installed by deploy/python.sh into site-packages with a .pth auto-import
hook so that TOAST is available on every Logger instance at interpreter
startup before any application code runs. Zero external dependencies.
"""

import logging

TOAST = 60

if not hasattr(logging.Logger, 'toast'):
    logging.addLevelName(TOAST, "TOAST")

    def _toast(self, message, *args, **kwargs):
        if self.isEnabledFor(TOAST):
            self._log(TOAST, message, args, **kwargs)

    logging.Logger.toast = _toast
