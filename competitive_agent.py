from openai import OpenAI

class CompetitiveAgent:
    def __init__(self, project_name: str, model: str = "gpt-4o"):
        self.project_name = project_name
        self.model = "gpt-4o"
        self.client = OpenAI()

    def analyze(self) -> str:
        prompt = (
            f"You are a technical strategist. Based on the open-source Python project called '{self.project_name}', "
            f"identify one or two competing or similar tools or libraries. For each, compare their:\n"
            f"- Core features\n"
            f"- Typical use cases\n"
            f"- Popularity (GitHub stars, PyPI downloads, etc.)\n"
            f"- Strengths and limitations\n\n"
            f"Then summarize whether this project stands out, or how it fits into the ecosystem.\n\n"
            f"Your response should be suitable for a CEO or business decision-maker."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a concise and strategic technical market analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        return response.choices[0].message.content.strip()
