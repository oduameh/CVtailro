"""Unit tests for email follow-up templates."""

import pytest

from email_templates import format_templates_as_markdown, generate_follow_up_templates


@pytest.mark.unit
class TestEmailTemplates:
    """Email template generation."""

    def test_generates_three_templates(self):
        templates = generate_follow_up_templates(
            candidate_name="Jane Doe",
            job_title="Software Engineer",
            company="Acme",
        )
        assert len(templates) == 3

    def test_templates_contain_job_info(self):
        templates = generate_follow_up_templates(
            candidate_name="Jane Doe",
            job_title="Software Engineer",
            company="Acme",
        )
        for t in templates:
            assert "Software Engineer" in t.subject or "Software Engineer" in t.body
            assert "Acme" in t.body
            assert "Jane Doe" in t.body

    def test_single_skill_mention(self):
        templates = generate_follow_up_templates(
            candidate_name="Jane",
            job_title="Engineer",
            company="Acme",
            key_skills=["Python"],
        )
        assert "particularly in Python" in templates[0].body

    def test_two_skills_mention(self):
        templates = generate_follow_up_templates(
            candidate_name="Jane",
            job_title="Engineer",
            company="Acme",
            key_skills=["Python", "Django"],
        )
        assert "particularly in Python and Django" in templates[0].body

    def test_no_skills(self):
        templates = generate_follow_up_templates(
            candidate_name="Jane",
            job_title="Engineer",
            company="Acme",
        )
        assert "particularly in" not in templates[0].body or "aligns well" in templates[0].body

    def test_use_cases(self):
        templates = generate_follow_up_templates(
            candidate_name="Jane",
            job_title="Engineer",
            company="Acme",
        )
        use_cases = {t.use_case for t in templates}
        assert "Application follow-up" in use_cases
        assert "Post-interview thank you" in use_cases
        assert "Networking outreach" in use_cases


@pytest.mark.unit
class TestFormatTemplatesMarkdown:
    """Markdown formatting."""

    def test_format_templates_as_markdown(self):
        templates = generate_follow_up_templates("Jane", "Engineer", "Acme")
        md = format_templates_as_markdown(templates)
        assert "## Email Follow-Up Templates" in md
        assert "### 1." in md
        assert "**Subject:**" in md
        assert "### 2." in md
        assert "### 3." in md
