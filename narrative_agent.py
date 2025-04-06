from openai import OpenAI
import json

class NarrativeAgent:
    def __init__(self, role, repo_name, qa_pairs,  model="gpt-4o", syntax_errors=None):
        self.role = role
        self.repo_name = repo_name
        self.qa_pairs = qa_pairs
        self.model = model
        self.syntax_errors = syntax_errors

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
            f"5. If any answer includes a code snippet (in triple backticks), embed that code block exactly as-is in the report. You must include at least one code block for each major function, type, or structural element discussed. Do not omit them. \n"
            f"6. Only use triple backticks for code. Do not use markdown formatting (###, **bold**, lists) in regular text.\n"
            f"7. If information is missing, say: 'Not found in retrieved content.'\n"
            f"8. Avoid exaggeration or features like AI, ML, blockchain, etc., unless explicitly mentioned.\n"
            f"9. {snippet_instruction}"
        )

        # Add syntax error information to the prompt if available
        if hasattr(self, 'syntax_errors') and self.syntax_errors and self.syntax_errors.get('has_syntax_errors', False):
            prompt += (
                f"\n10. Include a 'Code Quality' section in your report that addresses the following syntax errors found in the codebase:\n"
                f"{self.syntax_errors['summary']}\n"
            )

            # Add detailed error information if appropriate for the role
            if role.lower() == "programmer":
                prompt += "\nDetailed syntax errors:\n"
                for error in self.syntax_errors['errors']:
                    file_path = error['file_path']
                    line_info = f" at line {error['line_number']}" if error.get('line_number') else ""
                    prompt += f"- {file_path}{line_info}: {error['error_msg']}\n"

            prompt += "\nAnalyze what these syntax errors might indicate about code quality and development practices."

        if role.lower() in ["ceo", "sales", "sales_manager", "marketing"]:
            prompt += (
                f"\n11. Begin the report with a clearly marked section titled **Executive Summary**. "
                f"Write 3–5 sentences summarizing the project's strategic importance, key strengths, and overall purpose. "
                f"After this section, continue the full narrative without repeating the summary."
            )


        print("================DEBUG: Number of answers passed to Phase 2:", len(answers))

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