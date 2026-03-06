"""Tests for the theme / design system module."""

import os
os.environ.setdefault("XAI_API_KEY", "fake-test-key-for-unit-tests")

from src.theme import (
    COLORS,
    EXECUTIVE_CSS,
    decision_badge,
    decision_color,
    kpi_card,
    page_header,
    risk_badge,
    risk_color,
    section_header,
)


class TestColors:
    def test_has_required_keys(self):
        required = {"primary", "secondary", "accent", "background", "surface", "text",
                     "success", "danger", "warning", "muted"}
        assert required.issubset(set(COLORS.keys()))

    def test_colors_are_hex(self):
        for key, val in COLORS.items():
            assert val.startswith("#"), f"{key} color should be hex: {val}"

    def test_has_semantic_bg_colors(self):
        assert "success_bg" in COLORS
        assert "danger_bg" in COLORS
        assert "warning_bg" in COLORS


class TestRiskColor:
    def test_high_risk(self):
        assert risk_color(75) == COLORS["danger"]

    def test_medium_risk(self):
        assert risk_color(45) == COLORS["warning"]

    def test_low_risk(self):
        assert risk_color(15) == COLORS["success"]

    def test_boundary_high(self):
        assert risk_color(70) == COLORS["danger"]

    def test_boundary_medium(self):
        assert risk_color(30) == COLORS["warning"]

    def test_boundary_low(self):
        assert risk_color(29) == COLORS["success"]


class TestRiskBadge:
    def test_high_risk_badge(self):
        html = risk_badge(85)
        assert "85" in html
        assert COLORS["danger"] in html

    def test_low_risk_badge(self):
        html = risk_badge(10)
        assert "10" in html
        assert COLORS["success"] in html

    def test_returns_string(self):
        assert isinstance(risk_badge(50), str)

    def test_contains_pill_styling(self):
        html = risk_badge(50)
        assert "border-radius" in html


class TestDecisionColor:
    def test_approved(self):
        assert decision_color("approved") == COLORS["success"]

    def test_rejected(self):
        assert decision_color("rejected") == COLORS["danger"]

    def test_escalated(self):
        assert decision_color("escalated") == COLORS["warning"]

    def test_pending(self):
        assert decision_color("pending_human_review") == COLORS["warning"]

    def test_unknown(self):
        assert decision_color("something_else") == COLORS["muted"]


class TestDecisionBadge:
    def test_approved_badge(self):
        html = decision_badge("approved")
        assert "approved" in html
        assert "uppercase" in html
        assert COLORS["success"] in html
        assert COLORS["success_bg"] in html

    def test_rejected_badge(self):
        html = decision_badge("rejected")
        assert "rejected" in html
        assert COLORS["danger"] in html

    def test_underscore_replaced(self):
        html = decision_badge("pending_human_review")
        assert "pending human review" in html.lower()

    def test_returns_string(self):
        assert isinstance(decision_badge("approved"), str)


class TestKpiCard:
    def test_basic_card(self):
        html = kpi_card("Total", "42")
        assert "Total" in html
        assert "42" in html
        assert COLORS["surface"] in html

    def test_with_subtitle(self):
        html = kpi_card("Rate", "85%", subtitle="17 of 20")
        assert "17 of 20" in html

    def test_with_accent(self):
        html = kpi_card("Risk", "30", accent="#FF0000")
        assert "#FF0000" in html

    def test_no_subtitle_when_empty(self):
        html = kpi_card("Label", "Value")
        # Should not contain the subtitle div wrapper when empty
        assert html.count("<div") < 5


class TestPageHeader:
    def test_contains_title(self):
        html = page_header()
        assert "Invoice Processing AI" in html

    def test_contains_gradient(self):
        html = page_header()
        assert "linear-gradient" in html
        assert COLORS["primary"] in html

    def test_contains_status(self):
        html = page_header()
        assert "System Online" in html


class TestSectionHeader:
    def test_basic(self):
        html = section_header("My Title")
        assert "My Title" in html

    def test_with_subtitle(self):
        html = section_header("Title", "Subtitle text")
        assert "Subtitle text" in html


class TestExecutiveCSS:
    def test_is_nonempty_string(self):
        assert isinstance(EXECUTIVE_CSS, str)
        assert len(EXECUTIVE_CSS) > 100

    def test_contains_metric_styling(self):
        assert "metric" in EXECUTIVE_CSS.lower() or "data-testid" in EXECUTIVE_CSS

    def test_contains_color_variables(self):
        assert COLORS["primary"] in EXECUTIVE_CSS or COLORS["accent"] in EXECUTIVE_CSS

    def test_hides_streamlit_chrome(self):
        assert "#MainMenu" in EXECUTIVE_CSS
        assert "display: none" in EXECUTIVE_CSS

    def test_contains_font_import(self):
        assert "Inter" in EXECUTIVE_CSS
        assert "fonts.googleapis.com" in EXECUTIVE_CSS

    def test_contains_tab_styling(self):
        assert "tab" in EXECUTIVE_CSS.lower()

    def test_contains_sidebar_styling(self):
        assert "stSidebar" in EXECUTIVE_CSS

    def test_chart_background_transparent(self):
        assert "vega-embed" in EXECUTIVE_CSS
        assert "transparent" in EXECUTIVE_CSS

    def test_chart_container_uses_light_bg(self):
        assert "stVegaLiteChart" in EXECUTIVE_CSS
