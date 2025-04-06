def _process_python_code(self, file_path: str, content: str) -> List[ContentChunk]:
        """
        Process Python code file using AST.

        Args:
            file_path: Path to the file
            content: File content

        Returns:
            List of ContentChunk objects
        """
        chunks = []

        try:
            # Parse the Python code into an AST
            tree = ast.parse(content)

            # Track line numbers for AST nodes
            line_numbers = {}
            for node in ast.walk(tree):
                if hasattr(node, 'lineno'):
                    line_numbers[node] = (
                        getattr(node, 'lineno', None),
                        getattr(node, 'end_lineno', None)
                    )

            # Process classes
            for cls_node in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
                cls_name = cls_node.name
                cls_start, cls_end = line_numbers.get(cls_node, (None, None))
                if cls_start and cls_end:
                    # Extract class definition and docstring
                    cls_lines = content.splitlines()[cls_start-1:cls_end]
                    cls_content = '\n'.join(cls_lines)

                    # Create chunk for the class
                    cls_chunk = ContentChunk(
                        content=cls_content,
                        file_path=file_path,
                        chunk_type=FILE_TYPE_CODE,
                        start_line=cls_start,
                        end_line=cls_end,
                        language=LANG_PYTHON,
                        name=cls_name,
                        metadata={"type": "class"}
                    )
                    chunks.append(cls_chunk)

                    # Process methods within the class
                    for method_node in [n for n in cls_node.body if isinstance(n, ast.FunctionDef)]:
                        method_name = method_node.name
                        method_start, method_end = line_numbers.get(method_node, (None, None))
                        if method_start and method_end:
                            # Extract method definition
                            method_lines = content.splitlines()[method_start-1:method_end]
                            method_content = '\n'.join(method_lines)

                            # Create chunk for the method
                            method_chunk = ContentChunk(
                                content=method_content,
                                file_path=file_path,
                                chunk_type=FILE_TYPE_CODE,
                                start_line=method_start,
                                end_line=method_end,
                                language=LANG_PYTHON,
                                parent=cls_name,
                                name=method_name,
                                metadata={"type": "method"}
                            )
                            chunks.append(method_chunk)

            # Process standalone functions
            for func_node in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
                func_name = func_node.name
                func_start, func_end = line_numbers.get(func_node, (None, None))
                if func_start and func_end:
                    # Extract function definition
                    func_lines = content.splitlines()[func_start-1:func_end]
                    func_content = '\n'.join(func_lines)

                    # Create chunk for the function
                    func_chunk = ContentChunk(
                        content=func_content,
                        file_path=file_path,
                        chunk_type=FILE_TYPE_CODE,
                        start_line=func_start,
                        end_line=func_end,
                        language=LANG_PYTHON,
                        name=func_name,
                        metadata={"type": "function"}
                    )
                    chunks.append(func_chunk)

            # If no chunks were created (e.g., file with only imports or constants),
            # create a single chunk for the entire file
            if not chunks:
                self.logger.debug(f"No classes or functions found in {file_path}, using whole file")
                chunks = [ContentChunk(
                    content=content,
                    file_path=file_path,
                    chunk_type=FILE_TYPE_CODE,
                    language=LANG_PYTHON,
                    metadata={"type": "whole_file"}
                )]

            return chunks

        except SyntaxError as e:
            # Handle Python syntax errors
            self.logger.warning(f"Syntax error in Python file {file_path}: {e}")
            # Track the error
            self.error_tracker.add_error(
                file_path=file_path,
                language=LANG_PYTHON,
                error_msg=str(e),
                line_number=getattr(e, 'lineno', None)
            )
            # Fall back to generic chunking
            return self._chunk_by_size(
                content, file_path, FILE_TYPE_CODE, LANG_PYTHON,
                chunk_size=1500, overlap=200
            )

        except Exception as e:
            self.logger.error(f"Error processing Python file {file_path}: {e}", exc_info=True)
            # Fall back to generic chunking
            return self._chunk_by_size(
                content, file_path, FILE_TYPE_CODE, LANG_PYTHON,
                chunk_size=1500, overlap=200
            )

    def _process_class_node(self, class_node, file_path, content):
        local_chunks = []
        def get_node_text(node):
            return content[node.start_byte:node.end_byte]

        def get_line_range(node):
            return node.start_point[0] + 1, node.end_point[0] + 1

        class_text = get_node_text(class_node)
        start, end = get_line_range(class_node)
        class_name_node = class_node.child_by_field_name("name")
        class_name = get_node_text(class_name_node) if class_name_node else "UnknownClass"

        local_chunks.append(ContentChunk(
            content=class_text,
            file_path=file_path,
            chunk_type=FILE_TYPE_CODE,
            start_line=start,
            end_line=end,
            language="java",
            name=class_name,
            metadata={"type": "class"}
        ))

        for child in class_node.children:
            if child.type == "method_declaration":
                method_text = get_node_text(child)
                method_name_node = child.child_by_field_name("name")
                method_name = get_node_text(method_name_node) if method_name_node else "UnknownMethod"
                m_start, m_end = get_line_range(child)

                local_chunks.append(ContentChunk(
                    content=method_text,
                    file_path=file_path,
                    chunk_type=FILE_TYPE_CODE,
                    start_line=m_start,
                    end_line=m_end,
                    language="java",
                    parent=class_name,
                    name=method_name,
                    metadata={"type": "method"}
                ))

        return local_chunks

    def _process_go_code(self, file_path, content):
        try:
            parser = Parser()
            parser.set_language(GO_LANGUAGE)
            tree = parser.parse(bytes(content, "utf8"))
            root = tree.root_node

            # Check if tree-sitter encountered errors during parsing
            if root.has_error:
                # Track the error but continue processing
                self.error_tracker.add_error(
                    file_path=file_path,
                    language=LANG_GO,
                    error_msg="Go syntax error detected by tree-sitter"
                )
                self.logger.warning(f"Syntax error in Go file {file_path}")

            chunks = []

            def extract_chunks(node):
                if node.type in ["function_declaration", "method_declaration", "type_declaration", "struct_type"]:
                    code = content[node.start_byte:node.end_byte]
                    name_node = node.child_by_field_name("name")
                    name = name_node.text.decode() if name_node else "unnamed"
                    chunks.append({
                        "file_path": file_path,
                        "language": "go",
                        "name": name,
                        "type": node.type,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "content": code.strip()
                    })
                for child in node.children:
                    extract_chunks(child)

            extract_chunks(root)
            return chunks

        except Exception as e:
            self.logger.error(f"Error processing Go file {file_path}: {e}", exc_info=True)
            self.error_tracker.add_error(
                file_path=file_path,
                language=LANG_GO,
                error_msg=f"Failed to parse: {str(e)}"
            )
            return []  # Return empty list on failure

    def _process_java_code(self, file_path: str, content: str) -> List[ContentChunk]:
        """Process Java file using Tree-sitter to extract classes and methods."""
        chunks = []
        try:
            tree = JAVA_PARSER.parse(bytes(content, "utf8"))
            root = tree.root_node

            # Check if tree-sitter encountered errors during parsing
            if root.has_error:
                # Find all error nodes
                error_nodes = []

                def find_error_nodes(node):
                    if node.has_error or node.is_missing or node.type == "ERROR":
                        error_nodes.append(node)
                    for child in node.children:
                        find_error_nodes(child)

                find_error_nodes(root)

                for error_node in error_nodes:
                    # Get line and column info
                    line_num = error_node.start_point[0] + 1
                    col_num = error_node.start_point[1] + 1

                    # Get context (3 lines before and after)
                    lines = content.splitlines()
                    start_line = max(0, line_num - 4)
                    end_line = min(len(lines), line_num + 3)
                    context_lines = lines[start_line:end_line]
                    context = "\n".join(context_lines)

                    # Determine error type
                    error_type = "Unknown"
                    if error_node.type == "ERROR":
                        error_type = "Syntax Error"
                    elif error_node.is_missing:
                        error_type = "Missing Element"

                    # Determine containing element
                    containing_element = "Unknown"
                    parent = error_node.parent
                    while parent and parent != root:
                        if parent.type in ["class_declaration", "method_declaration", "field_declaration"]:
                            containing_element = parent.type
                            break
                        parent = parent.parent

                    # Add detailed error
                    self.error_tracker.add_error(
                        file_path=file_path,
                        language=LANG_JAVA,
                        error_msg=f"{error_type} in {containing_element}",
                        line_number=line_num,
                        function_name=containing_element,
                        metadata={
                            "column": col_num,
                            "context": context,
                            "error_node_type": error_node.type,
                            "containing_element": containing_element
                        }
                    )

                    self.logger.warning(f"Syntax error in Java file {file_path} at line {line_num}, column {col_num}")

                # If no specific error nodes found, add a generic error
                if not error_nodes:
                    self.error_tracker.add_error(
                        file_path=file_path,
                        language=LANG_JAVA,
                        error_msg="Java syntax error detected by tree-sitter"
                    )
                    self.logger.warning(f"Syntax error in Java file {file_path}")

                def get_node_text(node):
                    return content[node.start_byte:node.end_byte]

                def get_line_range(node):
                    return node.start_point[0] + 1, node.end_point[0] + 1

                def find_classes_and_methods(node):
                    for child in node.children:
                        if child.type == "class_declaration":
                            class_chunks = self._process_class_node(child, file_path, content)
                            chunks.extend(class_chunks)
                        else:
                            find_classes_and_methods(child)

                find_classes_and_methods(root)
                return chunks
        except Exception as e:
            self.logger.error(f"Error processing Java file {file_path}: {e}", exc_info=True)
            self.error_tracker.add_error(
                file_path=file_path,
                language=LANG_JAVA,
                error_msg=f"Failed to parse: {str(e)}"
            )
            # Fall back to generic chunking
            return self._chunk_by_size(
                content, file_path, FILE_TYPE_CODE, LANG_JAVA,
                chunk_size=1500, overlap=200
            )