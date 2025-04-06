class SyntaxErrorTracker:
    """Tracks syntax errors across different file types during repository processing."""

    def __init__(self):
        self.errors = []
        self.error_count = 0

    def add_error(self, file_path, language, error_msg, line_number=None, function_name=None, metadata=None):
        """
        Record a syntax error with enhanced details.

        Args:
            file_path: Path to the file with the error
            language: Programming language of the file
            error_msg: Error message or description
            line_number: Line number where the error occurred (optional)
            function_name: Function or class name containing the error (optional)
            metadata: Additional metadata about the error (optional)
        """
        self.errors.append({
            'file_path': file_path,
            'language': language,
            'error_msg': error_msg,
            'line_number': line_number,
            'function_name': function_name,
            'metadata': metadata or {}
        })
        self.error_count += 1

    def has_errors(self):
        """Check if any syntax errors were found."""
        return self.error_count > 0

    def get_errors(self):
        """Get all recorded syntax errors."""
        return self.errors

    def get_error_count(self):
        """Get the total number of syntax errors."""
        return self.error_count

    def generate_report(self):
        """Generate a formatted report of all syntax errors."""
        if not self.has_errors():
            return {
                "has_syntax_errors": False,
                "error_count": 0,
                "summary": "No syntax errors were detected in the codebase.",
                "errors": []
            }

        # Group errors by language
        errors_by_language = {}
        for error in self.errors:
            lang = error['language']
            if lang not in errors_by_language:
                errors_by_language[lang] = []
            errors_by_language[lang].append(error)

        # Create summary
        languages_with_errors = list(errors_by_language.keys())
        error_summary = f"Found {self.error_count} syntax errors across {len(languages_with_errors)} languages: {', '.join(languages_with_errors)}"

        return {
            "has_syntax_errors": True,
            "error_count": self.error_count,
            "summary": error_summary,
            "errors": self.errors,
            "errors_by_language": errors_by_language
        }