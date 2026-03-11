"""Tests for contact info stripping in recruiter resume versions."""

from app.services.pipeline import strip_contact_info


class TestStripContactInfo:
    def test_strips_email_from_contact_line(self):
        md = "# John Doe\nNew York, NY | (555) 123-4567 | john@example.com | linkedin.com/in/johndoe\n\n## Professional Summary"
        result = strip_contact_info(md)
        assert "john@example.com" not in result
        assert "(555) 123-4567" not in result
        assert "linkedin.com" not in result
        assert "New York, NY" in result
        assert "# John Doe" in result
        assert "## Professional Summary" in result

    def test_strips_phone_number_formats(self):
        md = "# Jane\nLondon | +44 7700 900000 | jane@mail.com\n\n## Experience"
        result = strip_contact_info(md)
        assert "+44 7700 900000" not in result
        assert "jane@mail.com" not in result
        assert "London" in result

    def test_keeps_headings_and_content(self):
        md = "# Alice\nSF | alice@test.com | linkedin.com/in/alice\n\n## Summary\nExperienced developer.\n\n## Experience\n**Engineer** | Acme Corp | SF\n2020 - Present\n- Built stuff"
        result = strip_contact_info(md)
        assert "alice@test.com" not in result
        assert "## Summary" in result
        assert "## Experience" in result
        assert "**Engineer** | Acme Corp | SF" in result
        assert "- Built stuff" in result

    def test_does_not_strip_role_pipes(self):
        md = "# Bob\nNYC | bob@x.com\n\n**Software Engineer** | Google | NYC\n2020 - 2023"
        result = strip_contact_info(md)
        assert "**Software Engineer** | Google | NYC" in result

    def test_handles_no_contact_line(self):
        md = "# Resume\n\n## Summary\nSome text."
        result = strip_contact_info(md)
        assert result == md

    def test_strips_standalone_email(self):
        md = "# Name\njohn@example.com\n\n## Summary"
        result = strip_contact_info(md)
        assert "john@example.com" not in result

    def test_strips_standalone_linkedin(self):
        md = "# Name\nhttps://www.linkedin.com/in/johndoe\n\n## Summary"
        result = strip_contact_info(md)
        assert "linkedin.com" not in result

    def test_preserves_empty_lines(self):
        md = "# Name\nCity | phone@email.com\n\n## Summary\n\nContent here."
        result = strip_contact_info(md)
        assert "## Summary" in result
        assert "Content here." in result

    def test_all_parts_contact_drops_line(self):
        md = "# Name\n+1-555-0199 | me@test.com | linkedin.com/in/me\n\n## Summary"
        result = strip_contact_info(md)
        lines = [l for l in result.split("\n") if l.strip()]
        assert lines[0] == "# Name"
        assert lines[1] == "## Summary"
