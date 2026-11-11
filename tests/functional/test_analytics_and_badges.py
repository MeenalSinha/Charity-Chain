"""
test_analytics_and_badges.py - Analytics and gamification functional tests
"""

import pytest
from charity_tracker import BadgeSystem, VisualizationManager


def test_badge_progression_system(isolated_db):
    user = "Active Donor"
    
    # 0 donations -> no badges
    badges_empty = BadgeSystem.check_badges(user, isolated_db)
    assert 'early_supporter' not in badges_empty
    
    # 1 donation -> early_supporter
    m_id = isolated_db.create_milestone("NGO Alpha", 1000000000000000000, "Clean Energy")
    isolated_db.add_donation(m_id, user, 1000000000000000000)
    
    badges_1 = BadgeSystem.check_badges(user, isolated_db)
    assert 'early_supporter' in badges_1
    assert 'top_donor' not in badges_1
    
    # 5 donations -> top_donor
    for _ in range(4):
        isolated_db.add_donation(m_id, user, 500000000000000000)
    
    badges_5 = BadgeSystem.check_badges(user, isolated_db)
    assert 'top_donor' in badges_5
    
    progress = BadgeSystem.get_badge_progress(user, isolated_db)
    assert progress['top_donor']['earned'] is True
    assert progress['early_supporter']['earned'] is True


def test_visualization_figures():
    gauge = VisualizationManager.create_confidence_gauge(0.85)
    assert gauge is not None
    
    checks = {
        'gps': {'status': 'PASSED', 'passed': True},
        'tamper': {'status': 'PASSED', 'passed': True},
        'object_detection': {'status': 'UNAVAILABLE', 'passed': False}
    }
    panel = VisualizationManager.create_confidence_explanation_panel(checks, 0.85)
    assert panel is not None

