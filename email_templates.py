"""Email Follow-Up Templates — generates professional email templates for job seekers.

Three templates:
1. Application follow-up (1 week after applying)
2. Post-interview thank you
3. Networking/informational interview request
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EmailTemplate:
    subject: str
    body: str
    use_case: str
    when_to_send: str


def generate_follow_up_templates(
    candidate_name: str,
    job_title: str,
    company: str,
    key_skills: list[str] | None = None,
    talking_points: list[str] | None = None,
) -> list[EmailTemplate]:
    """Generate 3 professional email templates tailored to the job application."""

    skills_mention = ""
    if key_skills and len(key_skills) >= 2:
        skills_mention = f", particularly in {key_skills[0]} and {key_skills[1]}"
    elif key_skills and len(key_skills) >= 1:
        skills_mention = f", particularly in {key_skills[0]}"

    talking_point = ""
    if talking_points and len(talking_points) > 0:
        talking_point = talking_points[0]

    first_name = candidate_name.split()[0] if candidate_name else "there"

    templates = [
        EmailTemplate(
            subject=f"Following Up — {job_title} Application",
            body=f"""Dear Hiring Manager,

I hope this message finds you well. I recently submitted my application for the {job_title} position at {company}, and I wanted to express my continued interest in the opportunity.

I believe my background{skills_mention} aligns well with the requirements of the role. I am confident I can contribute meaningfully to your team and would welcome the chance to discuss how my experience can support {company}'s goals.

I understand you are likely reviewing many applications, so I appreciate your time and consideration. Please don't hesitate to reach out if you need any additional information from me.

Thank you for your time, and I look forward to hearing from you.

Best regards,
{candidate_name}""",
            use_case="Application follow-up",
            when_to_send="5-7 business days after submitting your application",
        ),
        EmailTemplate(
            subject=f"Thank You — {job_title} Interview",
            body=f"""Dear [Interviewer's Name],

Thank you for taking the time to meet with me today to discuss the {job_title} position at {company}. I truly enjoyed learning more about the team and the exciting work you're doing.

Our conversation reinforced my enthusiasm for the role. I was particularly excited to hear about [specific topic discussed]. I believe my experience{skills_mention} would allow me to make a strong impact from day one.

I am very enthusiastic about the opportunity to join {company} and contribute to the team's success. Please don't hesitate to reach out if there is any additional information I can provide.

Thank you again for the opportunity. I look forward to the next steps.

Warm regards,
{candidate_name}""",
            use_case="Post-interview thank you",
            when_to_send="Within 24 hours after your interview",
        ),
        EmailTemplate(
            subject=f"Connecting About Opportunities at {company}",
            body=f"""Dear [Contact Name],

I hope this message finds you well. My name is {candidate_name}, and I am reaching out because I am very interested in the {job_title} role at {company}.

I have been following {company}'s work and am impressed by [specific aspect]. With my background{skills_mention}, I believe I could bring valuable perspective to your team.

I would love the opportunity to have a brief conversation to learn more about the team culture and any advice you might have for someone applying to this role. I am happy to work around your schedule — even a 15-minute chat would be incredibly valuable.

Thank you for considering my request. I look forward to connecting.

Best regards,
{candidate_name}""",
            use_case="Networking outreach",
            when_to_send="Before or shortly after applying, to build connections at the company",
        ),
    ]

    return templates


def format_templates_as_markdown(templates: list[EmailTemplate]) -> str:
    """Format email templates as readable markdown."""
    md_parts = ["## Email Follow-Up Templates\n"]

    for i, t in enumerate(templates, 1):
        md_parts.append(f"### {i}. {t.use_case}")
        md_parts.append(f"**When to send:** {t.when_to_send}\n")
        md_parts.append(f"**Subject:** {t.subject}\n")
        md_parts.append("---")
        md_parts.append(t.body)
        md_parts.append("\n---\n")

    return "\n".join(md_parts)
