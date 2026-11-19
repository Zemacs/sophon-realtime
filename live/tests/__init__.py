"""Tests. Installs the fake JEV (see jev_stub.py) so ledger/control tests do not depend on a live Karpo."""
from . import jev_stub as _jev_stub

_jev_stub.install()
