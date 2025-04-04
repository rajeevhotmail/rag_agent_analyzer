def _build_html( self, text: str, repo_name: str, role: str, key_findings: list[str], competitive_section: str = None ) -> str:
        from section_headers import ROLE_SECTION_HEADERS
        import re
        from datetime import datetime

        lines = text.split('\n')
        in_code_block = False
        code_buffer = []
        summary_html = ""
        body_html_parts = []

        is_summary = False
        collecting_summary = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                if not in_code_block:
                    body_html_parts.append("<pre><code>" + "\n".join(code_buffer) + "</code></pre>")
                    code_buffer = []
                continue

            if in_code_block:
                code_buffer.append(line)
                continue

            if "**Executive Summary**" in line:
                collecting_summary = True
                continue

            if collecting_summary:
                if not line.strip():
                    collecting_summary = False
                    continue
                summary_html += f"<p>{line.strip()}</p>"
            else:
                if line.strip().startswith("- "):
                    body_html_parts.append(f"<p>{line.strip()}</p>")
                elif line.strip().startswith("**"):
                    body_html_parts.append(f"<p><strong>{line.strip('**')}</strong></p>")
                elif line.strip():
                    body_html_parts.append(f"<p>{line.strip()}</p>")

        section_headers = ROLE_SECTION_HEADERS.get(role.lower(), ["Narrative"])
        chunk_size = max(1, len(body_html_parts) // len(section_headers))
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

        if summary_html:
            html += "<h2>Executive Summary</h2>"
            html += summary_html

        # ✅ Table of Contents
        html += "<h2>Table of Contents</h2><ul>"
        for i, header in enumerate(section_headers):
            section_id = f"section{i+1}"
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

        # ✅ Main Narrative Sections
        for i, header in enumerate(section_headers):
            section_id = f"section{i+1}"
            html += f'<h2 id="{section_id}">{header}</h2>'
            for para in body_html_parts[i * chunk_size: (i + 1) * chunk_size]:
                html += para

        # ✅ Competitive Landscape Section with Markdown Table Support
        if competitive_section:
            html += '<h2 id="competitive">Competitive Landscape</h2>'
            lines = competitive_section.strip().split("\n")
            table_lines = []
            in_table = False

            for line in lines:
                if re.match(r"^\|.+\|$", line.strip()):
                    in_table = True
                    table_lines.append(line.strip())
                elif in_table and not line.strip():
                    # End of table block
                    html += "<table border='1' cellspacing='0' cellpadding='5'>"
                    for i, row in enumerate(table_lines):
                        cells = [c.strip() for c in row.split("|")[1:-1]]
                        if i == 0:
                            html += "<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>"
                        else:
                            html += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
                    html += "</table><br/>"
                    table_lines = []
                    in_table = False
                elif in_table:
                    table_lines.append(line.strip())
                elif line.strip().startswith("- "):
                    html += f"<p>{line.strip()}</p>"
                elif line.strip().startswith("**") and line.strip().endswith("**"):
                    html += f"<p><strong>{line.strip('**')}</strong></p>"
                elif line.strip():
                    html += f"<p>{line.strip()}</p>"

            # ⬇️ ✅ This block goes OUTSIDE the loop to catch table at end of file
            if in_table and table_lines:
                html += "<table border='1' cellspacing='0' cellpadding='5'>"
                for i, row in enumerate(table_lines):
                    cells = [c.strip() for c in row.split("|")[1:-1]]
                    if i == 0:
                        html += "<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>"
                    else:
                        html += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
                html += "</table><br/>"
        return html