import os
import time
import sys
from section_headers import ROLE_SECTION_HEADERS
from weasyprint import HTML, CSS
from datetime import datetime

class WeasyPDFWriter:
    def __init__(self, output_dir="output_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_pdf(self, text: str, repo_name: str, role: str, key_findings: list[str] = None) -> str:
        filename = f"{repo_name}_{role}_report_{int(time.time())}.pdf"
        output_path = os.path.join(self.output_dir, filename)

        # Convert content into HTML
        html_content = self._build_html(text, repo_name, role, key_findings)

        # Write PDF
        HTML(string=html_content).write_pdf(output_path, stylesheets=[self._get_default_css()])
        print(f"✅ PDF saved to: {output_path}")
        return output_path

    def _build_html(self, text: str, repo_name: str, role: str, key_findings: list[str]) -> str:
        from section_headers import ROLE_SECTION_HEADERS

        lines = text.split('\n')
        paragraphs = []
        in_code_block = False
        code_buffer = []

        for line in lines:
            if line.strip().startswith("```"):
                if in_code_block:
                    paragraphs.append("<pre><code class=\"language-python\">" + "\n".join(code_buffer) + "</code></pre>")
                    code_buffer = []
                    in_code_block = False
                else:
                    in_code_block = True
            elif in_code_block:
                code_buffer.append(line)
            elif line.strip():
                paragraphs.append(f"<p>{line.strip()}</p>")

        section_headers = ROLE_SECTION_HEADERS.get(role.lower(), ["Narrative"])
        chunk_size = max(1, len(paragraphs) // len(section_headers))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ✅ Start HTML layout
        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>{repo_name} ({role})</title>
        </head>
        <body>
            <h1>Repository Analysis Report</h1>
            <h2>{repo_name} ({role.title()} Perspective)</h2>
            <p><em>Generated on: {timestamp}</em></p>
        """

        # ✅ Insert Key Findings FIRST
        if key_findings:
            html += "<h2>Key Findings</h2><ul>"
            for point in key_findings:
                if point.strip():
                    html += f"<li>{point.strip()}</li>"
            html += "</ul>"

        # ✅ Table of Contents
        html += "<h2>Table of Contents</h2><ul>"
        for i, header in enumerate(section_headers):
            section_id = f"section{i+1}"
            html += f'<li><a href="#{section_id}">{header}</a></li>'
        html += "</ul>"

        # ✅ Main content
        for i, header in enumerate(section_headers):
            section_id = f"section{i+1}"
            html += f'<h2 id="{section_id}">{header}</h2>'
            for para in paragraphs[i * chunk_size: (i + 1) * chunk_size]:
                if para.startswith("<pre>"):
                    html += para
                else:
                    html += para

        html += "</body></html>"
        return html




    def _get_default_css(self):
        return CSS(string="""
            @page {
                size: A4;
                margin: 1in;
                @bottom-right {
                    content: "Page " counter(page);
                    font-size: 10pt;
                    color: #888;
                }
            }
            body {
                font-family: 'Helvetica', sans-serif;
                line-height: 1.6;
            }
            h1 {
                font-size: 24pt;
                text-align: center;
                margin-bottom: 0.5em;
            }
            h2 {
                font-size: 16pt;
                color: #2a2a2a;
                margin-top: 1.2em;
            }
            p {
                margin: 0.4em 0;
                font-size: 11pt;
            }
            ul {
                margin: 0 0 1em 1.5em;
                font-size: 11pt;
            }
            pre {
                background: #f5f5f5;
                border-left: 3px solid #ccc;
                padding: 10px;
                overflow-x: auto;
                font-size: 10pt;
                font-family: monospace;
            }
            code.language-python {
                color: #2a2a2a;
            }
            code.language-python .keyword { color: #007020; font-weight: bold; }
            code.language-python .name { color: #06287e; }
            code.language-python .string { color: #4070a0; }
            code.language-python .comment { color: #60a0b0; font-style: italic; }
            code.language-python .number { color: #40a070; }
        """)


# --- CLI Runner ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python weasy_pdf_writer.py <path_to_text_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Dummy test values
    repo_name = "test-repo"
    role = "programmer"
    key_findings = ["This is a sample key finding.", "Another key point shown here."]

    writer = WeasyPDFWriter()
    writer.write_pdf(text, repo_name, role, key_findings)
