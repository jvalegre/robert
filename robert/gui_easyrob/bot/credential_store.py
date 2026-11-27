"""Protect saved cloud API keys with the operating system's credential service."""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
import sys

from PySide6.QtCore import QSettings


class CredentialStoreError(RuntimeError):
    """A credential could not be saved or recovered securely."""

    def __init__(self, message: str, temporary_secret: str = "") -> None:
        super().__init__(message)
        self.temporary_secret = temporary_secret


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("size", wintypes.DWORD),
        ("data", ctypes.POINTER(ctypes.c_byte)),
    ]


def _windows_crypt(data: bytes, *, decrypt: bool) -> bytes:
    source_buffer = ctypes.create_string_buffer(data)
    source = _DataBlob(len(data), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_byte)))
    result = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    operation = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    operation.argtypes = (
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p if decrypt else wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    )
    operation.restype = wintypes.BOOL
    if not operation(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(result.data, ctypes.c_void_p))


class CredentialStore:
    """Store a single robBOT key without keeping plaintext in QSettings."""

    LEGACY_KEY = "bot/api_key"
    PROTECTED_KEY = "bot/api_key_protected"
    SERVICE = "easyROB.robBOT"
    ACCOUNT = "cloud_api_key"

    def __init__(self, settings) -> None:
        self.settings = settings

    def _check_settings_write(self) -> None:
        status = getattr(self.settings, "status", lambda: QSettings.NoError)()
        if status != QSettings.NoError:
            raise CredentialStoreError("The API key settings could not be written.")

    @staticmethod
    def _keyring():
        try:
            import keyring
        except ImportError as exc:
            raise CredentialStoreError(
                "Secure API key storage is unavailable on this system. Install the keyring package."
            ) from exc
        return keyring

    def save(self, secret: str) -> None:
        if not secret:
            raise CredentialStoreError("The API key is empty.")
        try:
            if sys.platform == "win32":
                protected = _windows_crypt(secret.encode("utf-8"), decrypt=False)
                self.settings.setValue(self.PROTECTED_KEY, base64.b64encode(protected).decode("ascii"))
            else:
                self._keyring().set_password(self.SERVICE, self.ACCOUNT, secret)
            self.settings.remove(self.LEGACY_KEY)
            self.settings.sync()
            self._check_settings_write()
        except CredentialStoreError:
            raise
        except Exception as exc:
            raise CredentialStoreError("The API key could not be saved securely.") from exc

    def load(self) -> str:
        protected = str(self.settings.value(self.PROTECTED_KEY, "") or "")
        try:
            if sys.platform == "win32":
                secret = (
                    _windows_crypt(base64.b64decode(protected, validate=True), decrypt=True).decode("utf-8")
                    if protected else ""
                )
            else:
                secret = self._keyring().get_password(self.SERVICE, self.ACCOUNT) or ""
        except CredentialStoreError:
            secret = ""
        except Exception as exc:
            raise CredentialStoreError("The saved API key could not be read securely.") from exc
        if secret:
            if self.settings.value(self.LEGACY_KEY, None) is not None:
                self.settings.remove(self.LEGACY_KEY)
                self.settings.sync()
                self._check_settings_write()
            return secret

        legacy = str(self.settings.value(self.LEGACY_KEY, "") or "")
        if not legacy:
            return ""
        try:
            self.save(legacy)
        except CredentialStoreError as exc:
            self.settings.remove(self.LEGACY_KEY)
            self.settings.sync()
            raise CredentialStoreError(
                "The old API key could not be migrated to secure storage. It is available for this session only.",
                temporary_secret=legacy,
            ) from exc
        return legacy

    def delete(self) -> None:
        try:
            if sys.platform != "win32":
                keyring = self._keyring()
                if keyring.get_password(self.SERVICE, self.ACCOUNT):
                    keyring.delete_password(self.SERVICE, self.ACCOUNT)
            self.settings.remove(self.PROTECTED_KEY)
            self.settings.remove(self.LEGACY_KEY)
            self.settings.sync()
            self._check_settings_write()
        except CredentialStoreError:
            raise
        except Exception as exc:
            raise CredentialStoreError("The saved API key could not be deleted.") from exc
