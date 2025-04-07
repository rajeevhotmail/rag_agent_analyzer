import os
import time
import sys
from section_headers import ROLE_SECTION_HEADERS
from weasyprint import HTML, CSS
from datetime import datetime

def _render_table(table_lines: list[str]) -> str:
    html = "<table border='1' cellspacing='0' cellpadding='5'>"
    for i, row in enumerate(table_lines):
        cells = [c.strip() for c in row.split("|")[1:-1]]
        if i == 0:
            html += "<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>"
        else:
            html += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    html += "</table><br/>"
    return html
class WeasyPDFWriter:
    def __init__(self, output_dir="output_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_pdf(self, text: str, repo_name: str, role: str, key_findings: list[str] = None, competitive_section: str = None) -> str:
        filename = f"{repo_name}_{role}_report_{int(time.time())}.pdf"
        output_path = os.path.join(self.output_dir, filename)

        print("====== DEBUG HTML INPUT START ======")
        print(text)
        print("====== DEBUG HTML INPUT END ======")
        # Convert content into HTML
        html_content = self._build_html(text, repo_name, role, key_findings, competitive_section)
        print("======= HTML Content Length =======")
        print(len(html_content))
        print("===================================")

        # Write PDF
        HTML(string=html_content).write_pdf(output_path, stylesheets=[self._get_default_css()])
        print(f"✅ PDF saved to: {output_path}")
        return output_path
    def _build_html(self, text: str, repo_name: str, role: str, key_findings: list[str] = None, competitive_section: str = None) -> str:
        from section_headers import ROLE_SECTION_HEADERS
        from datetime import datetime
        import re

        lines = text.split('\n')
        body_html_parts = []
        summary_html = ""
        in_code_block = False
        code_buffer = []
        is_collecting_summary = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                in_code_block = not in_code_block
                if not in_code_block:
                    body_html_parts.append("<pre><code>" + "\n".join(code_buffer) + "</code></pre>")
                    code_buffer = []
                continue

            if in_code_block:
                code_buffer.append(line)
                continue

            if "**Executive Summary**" in stripped:
                is_collecting_summary = True
                continue

            if is_collecting_summary:
                if not stripped:
                    is_collecting_summary = False
                    continue
                summary_html += f"<p>{stripped}</p>"
            elif stripped.lstrip().startswith("### "):
                body_html_parts.append(f"<h3>{stripped.lstrip()[4:].strip()}</h3>")
            elif stripped.lstrip().startswith("## "):
                section_title = stripped.lstrip()[3:].strip()
                body_html_parts.append(f"<h2>{section_title}</h2>")
            elif stripped.lstrip().startswith("# "):
                section_title = stripped.lstrip()[2:].strip()
                body_html_parts.append(f"<h2>{section_title}</h2>")
            elif re.match(r"^\*\*(.+?)\*\*$", stripped):
                bold_heading = re.match(r"^\*\*(.+?)\*\*$", stripped).group(1).strip()
                body_html_parts.append(f"<h2>{bold_heading}</h2>")
            elif stripped:
                from narrative_agent import strip_markdown
                clean = strip_markdown(stripped)
                clean = re.sub(r"^#+\s*", "", clean)  # sanitize any stray #
                body_html_parts.append(f"<p>{clean}</p>")


        #section_headers = ROLE_SECTION_HEADERS.get(role.lower(), ["Narrative"])
        #chunk_size = max(1, len(body_html_parts) // len(section_headers))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

        # ✅ Executive Summary
        if summary_html:
            html += "<h2>Executive Summary</h2>"
            html += summary_html

        # ✅ Table of Contents (dynamic)


        # ✅ Build sections dynamically based on <h2> tags in body_html_parts
        rendered_sections = []
        toc_entries = []
        current_section_id = None
        current_section_title = None
        current_paras = []

        for part in body_html_parts:
            if part.startswith("<h2>") and part.endswith("</h2>"):
                # Save previous section
                if current_section_id and current_paras:
                    rendered_sections.append((current_section_id, current_section_title, current_paras))

                # Start new section
                current_section_title = re.sub(r"<\/?h2>", "", part).strip()
                current_section_id = f"section_{len(toc_entries) + 1}"
                toc_entries.append((current_section_id, current_section_title))
                current_paras = []
            else:
                current_paras.append(part)

        # Final section
        if current_section_id and current_paras:
            rendered_sections.append((current_section_id, current_section_title, current_paras))

        # ✅ TOC rendering (same as before)
        html += "<h2>Table of Contents</h2><ul>"
        for section_id, header in toc_entries:
            html += f'<li><a href="#{section_id}">{header}</a></li>'
        if competitive_section:
            html += '<li><a href="#competitive">Competitive Landscape</a></li>'
        html += "</ul>"


        # ✅ Key Findings
        if key_findings:
            html += "<h2>Key Findings</h2><ul>"
            for point in key_findings:
                if point.strip():
                    html += f"<li>{point.strip()}</li>"
            html += "</ul>"

        # ✅ Render narrative sections
        for section_id, header, paras in rendered_sections:
            html += f'<h2 id="{section_id}">{header}</h2>'
            for para in paras:
                html += para

        # ✅ Render competitive section if present
        if competitive_section:
            html += '<h2 id="competitive">Competitive Landscape</h2>'
            html += competitive_section

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
                font-size: 10pt;
                font-family: monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                table-layout: fixed;
                word-wrap: break-word;
            }
            th, td {
                border: 1px solid #ccc;
                padding: 8px;
                font-size: 10pt;
                vertical-align: top;
                word-break: break-all;
            }
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