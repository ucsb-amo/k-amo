"""kamo.imaging.cross_sections: the tabulated values, and that the module stays
cheap to import (it is read by analysis workers and the acquisition server)."""
import math
import subprocess
import sys

import pytest

from kamo.imaging import cross_sections as cs


def test_import_is_arc_free():
    code = ("import sys; import kamo.imaging.cross_sections; "
            "bad = [m for m in ('arc', 'scipy', 'pandas', 'torch') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_closed_line_formula():
    lam = 766.701e-9
    assert cs.closed_line_cross_section(lam) == pytest.approx(3 * lam**2 / (2 * math.pi))


def test_legacy_value_is_lambda_squared_not_a_cross_section():
    ratio = cs.K39_LEGACY_LAMBDA_SQUARED_M2 / cs.K39_D2_CLOSED_HIGH_FIELD_M2
    assert ratio == pytest.approx(2 * math.pi / 3, rel=1e-4)


def test_legacy_d1_is_above_d2():
    assert cs.K39_LEGACY_D1_M2 / cs.K39_D2_CLOSED_HIGH_FIELD_M2 == pytest.approx(1.009, abs=2e-3)


def test_tabulated_high_field_value_reproduces():
    pytest.importorskip("arc")
    value = cs.compute_closed_line_cross_section(*cs.K39_HIGH_FIELD_IMAGING_LINE)
    assert value == pytest.approx(cs.K39_D2_CLOSED_HIGH_FIELD_M2, rel=1e-4)
