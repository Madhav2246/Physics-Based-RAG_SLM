import pytest
from physics.slm_extractor import _verify_extractable

def test_verify_extractable():
    # zero value
    assert _verify_extractable(0.0, "Vsb is 0 volts") is True
    assert _verify_extractable(0.0, "Vth is 0.5V") is False

    # SI values directly present
    assert _verify_extractable(0.5, "Vov is 0.5V") is True
    assert _verify_extractable(500, "current is 500 microamps") is True

    # Word-unit values (half, quarter)
    assert _verify_extractable(0.5, "overdrive is half a volt") is True
    assert _verify_extractable(0.25, "quarter") is True

    # Rejection of fabricated values
    assert _verify_extractable(42.0, "Id is 1mA") is False
    assert _verify_extractable(0.5, "Id is 1mA") is False

    # Magnitudes matching with different representations
    assert _verify_extractable(1e-3, "Id is 1mA") is True
    assert _verify_extractable(2e-6, "W is 2 microns") is True
    assert _verify_extractable(1.5e-9, "tox is 1.5 nm") is True
    assert _verify_extractable(2.5, "Id is 2.5 A") is True
