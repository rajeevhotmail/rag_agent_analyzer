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


    def build_narrative(self, report_data: dict) -> str:
        """
        Combines answers into a stitched narrative suitable for report writing.
        Uses an LLM to rewrite the structured answers as a cohesive report.
        """
        from openai import OpenAI

        role = report_data.get("role", "programmer").title()
        answers = report_data.get("answers", [])

        # Prepare Q&A block
        qa_block = ""
        for i, ans in enumerate(answers, 1):
            qa_block += f"{i}. Q: {ans['question']}\nA: {ans['answer']}\n\n"

        # Role-specific snippet toggle
        if role.lower() in ["ceo", "sales", "sales_manager", "marketing"]:
            snippet_instruction = (
                "Do not include any code snippets. Focus on conceptual, strategic, and architectural explanations. "
                "Avoid technical implementation details and developer-level content."
            )
        else:
            snippet_instruction = (
                "If the answers refer to functions, decorators, test cases, or usage patterns, include code snippets "
                "(3–8 lines) using markdown-style code blocks (```python). Embed them naturally into the prose."
            )

        # Prompt to GPT
        prompt = (
            f"You are a senior technical writer creating a professional report for a software project, "
            f"from the perspective of a {role}. Below are 10 answers about the project.\n\n"
            f"{qa_block}\n\n"
            f"Instructions:\n"
            f"1. Write in flowing prose, not in Q&A or bullet-heavy format.\n"
            f"2. Do not reference 'the context' or that this is AI-generated.\n"
            f"3. Speak with authority and clarity, as if you analyzed the code yourself.\n"
            f"4. {snippet_instruction}\n"
            f"5. Avoid repetition, keep the tone confident and concise.\n"
            f"6. Avoid exaggerating or introducing features such as AI, ML, blockchain, or predictive analytics "
            f"unless they were explicitly mentioned in the answers."
        )
        if role.lower() in ["ceo", "sales", "sales_manager", "marketing"]:
            prompt += (
                "\n7. Begin the report with a clearly marked section titled **Executive Summary**. "
                "Write 3–5 sentences summarizing the project's strategic importance, key strengths, and overall purpose. "
                "After this section, continue the full narrative without repeating the summary."
            )
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional technical writer who crafts software reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=2000
        )

        return response.choices[0].message.content.strip()