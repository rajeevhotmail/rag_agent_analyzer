from openai import OpenAI

class NarrativeAgent:
    def __init__(self, role: str, repo_name: str, qa_pairs: list, model="gpt-4"):
        self.role = role
        self.repo_name = repo_name
        self.qa_pairs = qa_pairs
        self.model = model

    def _compose_paragraph(self, question, answer):
        prompt = (
            f"You are a technical writer preparing a report for a {self.role}.\n"
            f"The repository is named '{self.repo_name}'.\n\n"
            f"Given the following insight:\n\n"
            f"Q: {question}\nA: {answer}\n\n"
            f"Write a concise, clear paragraph for a report. Avoid repeating the question.\n"
        )
        client = OpenAI()

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful technical writing assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        return response.choices[0].message.content.strip()


    def build_narrative(self):
        print("🤖 Building narrative section-by-section...")
        paragraphs = []
        for i, qa in enumerate(self.qa_pairs):
            print(f"✏️  Processing insight {i+1}/{len(self.qa_pairs)}...")
            para = self._compose_paragraph(qa["question"], qa["answer"])
            paragraphs.append(para)
        return "\n\n".join(paragraphs)
