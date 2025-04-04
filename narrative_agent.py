from openai import OpenAI
import json

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
                "If the answers mention specific functions, test methods, or class structures, "
                "embed at least one short code snippet (3–8 lines) using markdown-style triple backticks (```). "
                "Use code blocks only for actual code — not for formatting text."
            )

        # Prompt to GPT
        prompt = (
            f"You are a senior technical writer generating a formal report about a software project. "
            f"This report is for the role: {role}.\n\n"
            f"The following are answers derived from actual source code and documentation. "
            f"Some answers already include Python code snippets inside triple backticks. "
            f"Use these code blocks directly in your report wherever they are relevant.\n\n"
            f"{qa_block}\n\n"
            f"Instructions:\n"
            f"1. Write in fluent, third-person prose — no bullet points, markdown headers, or list formatting.\n"
            f"2. Maintain a professional, confident, book-like tone.\n"
            f"3. Do not use first-person language like 'I', 'we', or 'our'.\n"
            f"4. Avoid phrases like 'based on the context' or 'retrieved content'.\n"
            f"5.If any answer includes a code snippet (in triple backticks), embed that code block exactly as-is in the report. You must include at least one code block for each major function, type, or structural element discussed. Do not omit them. \n"
            f"6. Only use triple backticks for code. Do not use markdown formatting (###, **bold**, lists) in regular text.\n"
            f"7. If information is missing, say: 'Not found in retrieved content.'\n"
            f"8. Avoid exaggeration or features like AI, ML, blockchain, etc., unless explicitly mentioned."
        )
        if role.lower() in ["ceo", "sales", "sales_manager", "marketing"]:
            prompt += (
                "\n7. Begin the report with a clearly marked section titled **Executive Summary**. "
                "Write 3–5 sentences summarizing the project's strategic importance, key strengths, and overall purpose. "
                "After this section, continue the full narrative without repeating the summary."
            )
        with open("debug_phase2_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        print("================DEBUG: Number of answers passed to Phase 2:", len(answers))
        with open("debug_phase2_answers_raw.txt", "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=2)
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional technical writer who crafts software reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        return response.choices[0].message.content.strip()