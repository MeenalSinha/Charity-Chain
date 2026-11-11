"""
test_milestone_management.py - Unit tests for Project Milestone Management & Validation
======================================================================================
Tests milestone sequence constraints, coordinate validation, budget integrity,
and currency matching.
"""

import pytest
from backend.app.api.schemas import CreateMilestoneRequest, UpdateMilestoneRequest
from pydantic import ValidationError


def test_milestone_valid_coordinates():
    """Verify standard valid geographic coordinates."""
    req = CreateMilestoneRequest(
        sequence_number=1,
        title="Site Preparation",
        objective="Prepare construction foundation",
        funding_target_minor=50000,
        currency="USD",
        target_latitude=34.052235,
        target_longitude=-118.243683
    )
    assert req.target_latitude == 34.052235
    assert req.target_longitude == -118.243683


def test_milestone_invalid_latitude():
    """Verify latitude outside [-90, 90] fails validation."""
    with pytest.raises(ValidationError):
        CreateMilestoneRequest(
            sequence_number=1,
            title="Invalid Latitude",
            objective="Testing coordinate bounds",
            funding_target_minor=10000,
            currency="USD",
            target_latitude=95.0,  # Invalid: > 90
            target_longitude=0.0
        )


def test_milestone_invalid_longitude():
    """Verify longitude outside [-180, 180] fails validation."""
    with pytest.raises(ValidationError):
        CreateMilestoneRequest(
            sequence_number=1,
            title="Invalid Longitude",
            objective="Testing coordinate bounds",
            funding_target_minor=10000,
            currency="USD",
            target_latitude=0.0,
            target_longitude=-190.0  # Invalid: < -180
        )


def test_milestone_negative_or_zero_sequence():
    """Verify milestone sequence number must be positive integer >= 1."""
    with pytest.raises(ValidationError):
        CreateMilestoneRequest(
            sequence_number=0,  # Invalid: sequence must be >= 1
            title="Invalid Sequence",
            objective="Sequence zero test",
            funding_target_minor=10000,
            currency="USD"
        )


def test_milestone_negative_funding_target():
    """Verify negative funding target is rejected."""
    with pytest.raises(ValidationError):
        CreateMilestoneRequest(
            sequence_number=1,
            title="Negative Target",
            objective="Testing negative target",
            funding_target_minor=-500,  # Invalid
            currency="USD"
        )
