from openai import OpenAI

class CompetitiveAgent:
    def __init__(self, project_name: str, model: str = "gpt-4o"):
        self.project_name = project_name
        self.model = "gpt-4o"
        self.client = OpenAI()

    def analyze(self) -> str:
        prompt = (
            f"You are a technical strategist conducting a competitive analysis for the open-source project '{self.project_name}'.\n\n"
            f"Identify 1–2 competing or similar open-source tools.\n\n"
            f"Return only a well-formatted HTML table comparing all tools, including '{self.project_name}', using the following columns:\n"
            f"Tool, Core Features, Use Case, Performance, Ease of Use, Maintenance, Adoption, License\n\n"
            f"⚠️ Do not include any explanations, headings, bullet points, or markdown formatting.\n"
            f"Only return the <table> element with <tr>, <th>, and <td> tags.\n"
            f"Ensure the table is valid HTML and will render cleanly in a PDF document.\n"
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
