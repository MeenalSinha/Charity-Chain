"""
test_currency_math.py - Tests for Currency-Safe Money Value Object
"""

import pytest
from decimal import Decimal
from backend.app.core.currency import Money


def test_money_creation_from_minor_units():
    m = Money(minor_units=2550, currency="USD")
    assert m.minor_units == 2550
    assert m.currency == "USD"
    assert m.to_decimal() == Decimal("25.50")
    assert "25.50 USD" in m.to_formatted_string()


def test_money_creation_from_major_units():
    m = Money.from_major_units("19.99", "USD")
    assert m.minor_units == 1999
    assert m.to_decimal() == Decimal("19.99")


def test_bankers_rounding_half_even():
    # 2.5 cents rounds to nearest even (2) or 3.5 rounds to 4
    # In currency minor unit conversion with fractional minor units:
    m1 = Money.from_major_units("10.025", "USD")  # 1002.5 cents -> 1002
    assert m1.minor_units == 1002
    
    m2 = Money.from_major_units("10.035", "USD")  # 1003.5 cents -> 1004
    assert m2.minor_units == 1004


def test_money_addition_and_subtraction():
    m1 = Money(minor_units=1500, currency="USD")
    m2 = Money(minor_units=2500, currency="USD")
    
    total = m1 + m2
    assert total.minor_units == 4000
    assert total.currency == "USD"
    
    diff = m2 - m1
    assert diff.minor_units == 1000


def test_money_cross_currency_prevention():
    usd = Money(minor_units=1000, currency="USD")
    eur = Money(minor_units=1000, currency="EUR")
    
    with pytest.raises(ValueError, match="Currency mismatch"):
        _ = usd + eur


def test_money_percentage_multiplication():
    # 10% fee on $50.00 (5000 cents) -> 500 cents ($5.00)
    base = Money(minor_units=5000, currency="USD")
    fee = base.multiply_percentage(10)
    assert fee.minor_units == 500
    assert fee.to_decimal() == Decimal("5.00")

