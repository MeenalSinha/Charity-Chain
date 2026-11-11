"""
currency.py - Currency-Safe Financial Data Contracts & Calculations
===================================================================
Enforces the strict rule:
NEVER USE FLOATING-POINT ARITHMETIC FOR FINANCIAL VALUES.
All monetary amounts are stored in integer minor units (e.g., cents, paise)
or precise Decimals, with explicit ISO 4217 currency tracking.
"""

from typing import Union
from decimal import Decimal, ROUND_HALF_EVEN
from dataclasses import dataclass


# Currency minor unit scale mapping (e.g. USD has 2 decimal places -> factor 100)
CURRENCY_MINOR_UNITS = {
    "USD": 2,
    "EUR": 2,
    "GBP": 2,
    "INR": 2,
    "JPY": 0,
    "ETH": 18,  # Wei representation
}


@dataclass(frozen=True)
class Money:
    """
    Immutable value object representing an exact monetary value.
    Attributes:
        minor_units (int): The amount expressed in integer minor units (e.g., cents).
        currency (str): Uppercase 3-letter ISO 4217 code (or supported asset symbol like ETH).
    """
    minor_units: int
    currency: str

    def __post_init__(self):
        if not isinstance(self.minor_units, int):
            raise TypeError(f"minor_units must be an int, got {type(self.minor_units).__name__}")
        if not isinstance(self.currency, str) or len(self.currency.strip()) < 3:
            raise ValueError(f"Invalid currency code: {self.currency}")
        object.__setattr__(self, "currency", self.currency.strip().upper())

    @property
    def scale(self) -> int:
        return CURRENCY_MINOR_UNITS.get(self.currency, 2)

    @classmethod
    def from_major_units(cls, amount: Union[Decimal, str, int], currency: str = "USD") -> "Money":
        """
        Creates a Money instance from major units (e.g., Decimal('25.50') USD -> 2550 cents).
        Uses Banker's Rounding (ROUND_HALF_EVEN) to prevent accumulated bias.
        """
        curr = currency.strip().upper()
        scale = CURRENCY_MINOR_UNITS.get(curr, 2)
        dec = Decimal(str(amount))
        multiplier = Decimal(10 ** scale)
        minor = int((dec * multiplier).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))
        return cls(minor_units=minor, currency=curr)

    def to_decimal(self) -> Decimal:
        """Converts minor units to exact major Decimal representation."""
        divisor = Decimal(10 ** self.scale)
        return (Decimal(self.minor_units) / divisor).quantize(
            Decimal(f"0.{'0' * self.scale}" if self.scale > 0 else "1")
        )

    def to_formatted_string(self) -> str:
        """Returns human-readable string representation (e.g. '$25.50 USD')."""
        return f"{self.to_decimal()} {self.currency}"

    def __add__(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        return Money(minor_units=self.minor_units + other.minor_units, currency=self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        return Money(minor_units=self.minor_units - other.minor_units, currency=self.currency)

    def multiply_percentage(self, percentage: Union[Decimal, str, int]) -> "Money":
        """Multiplies money by a percentage using Banker's Rounding."""
        pct = Decimal(str(percentage)) / Decimal("100")
        result_minor = int((Decimal(self.minor_units) * pct).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))
        return Money(minor_units=result_minor, currency=self.currency)

    def _assert_same_currency(self, other: "Money"):
        if not isinstance(other, Money):
            raise TypeError(f"Cannot perform financial operations between Money and {type(other).__name__}")
        if self.currency != other.currency:
            raise ValueError(f"Currency mismatch: {self.currency} vs {other.currency}. Explicit conversion required.")

    def __repr__(self) -> str:
        return f"Money({self.minor_units} minor units, {self.currency} [={self.to_decimal()}])"

