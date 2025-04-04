#!/usr/bin/env python3
"""
RAG Engine Module

This module implements the Retrieval-Augmented Generation (RAG) engine
for answering questions about repositories. It provides functionality to:
1. Parse role-specific questions
2. Retrieve relevant content using the embeddings
3. Generate detailed answers using an LLM
4. Format answers for presentation

It implements comprehensive logging for tracking the question-answering process.
"""

import os
import json
import time
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
import importlib.util
from pathlib import Path

from embeddings_manager import EmbeddingsManager

logger = logging.getLogger("rag_engine")
logger.propagate = False
logger.setLevel(logging.DEBUG)

# Create console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

with open("retrieved_chunks_all.txt", "w", encoding="utf-8") as f:
    f.write("📄 Retrieved Chunks Log\n")
    f.write("=" * 40 + "\n\n")
def test_chunk_print(question: str, retriever, top_k: int = 5, output_file: str = "retrieved_chunks.txt"):
    """
    Utility function to write top-k retrieved chunks to a file for inspection.
    Overwrites the output file each time.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"🔍 Testing chunk retrieval for question: \"{question}\"\n\n")

        results = retriever.retrieve_relevant_chunks(question, top_k=top_k)

        for i, chunk in enumerate(results):
            file = chunk.get('file', 'unknown')
            content = chunk.get('content', '')[:800]  # limit to 800 chars
            f.write(f"\n📦 Chunk {i+1} - File: {file}\n")
            f.write("-" * 60 + "\n")
            f.write(content + "\n")
            f.write("-" * 60 + "\n\n")

    print(f"✅ Retrieved chunks written to: {output_file}")

class RAGEngine:
    """
    Retrieval-Augmented Generation engine for answering repository questions.
    """

    # Define question templates for different roles
    QUESTION_TEMPLATES = {
        "programmer": [
            "What programming languages are used in this project?",
            "What is the project's architecture/structure?",
            "What are the main components/modules of the project?",
            "What testing framework(s) are used?",
            "What dependencies does this project have?",
            "What is the code quality like (comments, documentation, etc.)?",
            "Are there any known bugs or issues?",
            "What is the build/deployment process?",
            "How is version control used in the project?",
            "What coding standards or conventions are followed?"
        ],
        "ceo": [
            "What is the overall purpose of this project?",
            "What business problem does this solve?",
            "What is the target market or user base?",
            "How mature is the project (stable, beta, etc.)?",
            "What is the competitive landscape for this project?",
            "What resources are required to maintain/develop this project?",
            "What are the potential revenue streams for this project?",
            "What are the biggest risks associated with this project?",
            "What metrics should be tracked to measure success?",
            "What is the roadmap for future development?"
        ],
        "sales_manager": [
            "What problem does this product solve for customers?",
            "What are the key features and benefits?",
            "Who is the target customer for this product?",
            "How does this product compare to competitors?",
            "What is the current state/version of the product?",
            "What are the technical requirements for using this product?",
            "Are there any case studies or success stories?",
            "What is the pricing model or strategy?",
            "What are common objections customers might have?",
            "What support options are available for customers?"
        ]
    }

    def __init__(self, embeddings_dir, repo_info, use_openai, use_local_llm,
                 local_llm_path, local_llm_type, log_level, qa_model="gpt-3.5-turbo"):
        """
        Initialize the RAG engine.

        Args:
            embeddings_dir: Directory containing embeddings and vector index
            repo_info: Dictionary with repository metadata
            use_openai: Whether to use OpenAI API for generation
            use_local_llm: Whether to use local LLM for generation
            local_llm_path: Path to local LLM model file
            local_llm_type: Type of local LLM model ('llama2' or 'codellama')
            log_level: Logging level for this engine instance
        """
        self.embeddings_dir = embeddings_dir
        self.repo_info = repo_info
        self.use_openai = use_openai
        self.use_local_llm = use_local_llm
        self.local_llm_path = local_llm_path
        self.local_llm_type = local_llm_type
        self.qa_model = qa_model

        # Load embeddings manager
        self.embedding_manager = EmbeddingsManager(
            output_dir=embeddings_dir,
            log_level=log_level
        )

        # Setup logger
        self.logger = logging.getLogger(f"rag_engine.{os.path.basename(embeddings_dir)}")
        self.logger.setLevel(log_level)

        # Remove all existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler for this instance
        log_dir = os.path.join(embeddings_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"rag_{int(time.time())}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.info(f"Initialized RAG engine for {repo_info.get('name', 'unknown repository')}")

        # Check if OpenAI is installed
        self._has_openai = importlib.util.find_spec("openai") is not None
        if use_openai and not self._has_openai:
            self.logger.warning(
                "openai package not found but use_openai=True. "
                "Install with: pip install openai"
            )

        # Check if LocalLLM support is available
        self._has_local_llm = importlib.util.find_spec("local_llm") is not None
        if use_local_llm and not self._has_local_llm:
            self.logger.warning(
                "local_llm module not found but use_local_llm=True. "
                "Make sure local_llm.py is in your Python path."
            )

        # Load LLM
        self._init_llm()

    def _init_llm(self):
        """Initialize the LLM for generation."""
        self.openai_client = None
        self.local_llm = None

        self.logger.info(f"Initializing LLM: use_openai={self.use_openai}, use_local_llm={self.use_local_llm}")

        if self.use_openai and self._has_openai:
            try:
                import openai

                # Check for API key in environment
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    self.logger.warning(
                        "OPENAI_API_KEY environment variable not set. "
                        "Set it with: export OPENAI_API_KEY=your_key"
                    )

                self.logger.info("OpenAI client initialized")
                self.openai_client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                self.logger.error(f"Error initializing OpenAI client: {e}", exc_info=True)
                self.openai_client = None

        # Add debug logs for local LLM import
        if self.use_local_llm:
            try:
                self.logger.info("Attempting to import local_llm module")
                from local_llm import LocalLLM
                self.logger.info("Local LLM module imported successfully")

                # Add checks for local LLM path
                if self.local_llm_path:
                    self.logger.info(f"Checking local LLM path: {self.local_llm_path}")
                    if os.path.exists(self.local_llm_path):
                        self.logger.info(f"Local LLM file exists: {self.local_llm_path}")

                        # Initialize the local LLM
                        self.logger.info(f"Initializing local LLM: {self.local_llm_path} (type: {self.local_llm_type})")
                        self.local_llm = LocalLLM(
                            model_path=self.local_llm_path,
                            model_type=self.local_llm_type
                        )
                    else:
                        self.logger.error(f"Local LLM file not found: {self.local_llm_path}")
                else:
                    self.logger.warning("No local LLM path provided")

                    # Try to find a default model
                    possible_paths = [
                        "../embeded_qa/llama_model/codellama-7b.Q4_K_M.gguf",
                        "../embeded_qa/models/llama-2-7b-chat.gguf"
                    ]

                    for path in possible_paths:
                        if os.path.exists(path):
                            self.logger.info(f"Found default LLM path: {path}")
                            self.local_llm_path = path
                            if "codellama" in path:
                                self.local_llm_type = "codellama"
                            else:
                                self.local_llm_type = "llama2"

                            # Initialize with the default model
                            self.logger.info(f"Initializing local LLM with default path: {self.local_llm_path}")
                            self.local_llm = LocalLLM(
                                model_path=self.local_llm_path,
                                model_type=self.local_llm_type
                            )
                            break

                    if not self.local_llm_path:
                        self.logger.error("Could not find a default LLM path")
            except ImportError as e:
                self.logger.error(f"Failed to import local_llm module: {e}", exc_info=True)
            except Exception as e:
                self.logger.error(f"Error initializing local LLM: {e}", exc_info=True)
                self.local_llm = None

        if not (self.openai_client or self.local_llm):
            self.logger.info("Using fallback generation (no LLM)")

    def load_data(self) -> bool:
        """Load embeddings and chunks data."""
        # Load embeddings and vector DB
        embeddings_loaded = self.embedding_manager.load_embeddings()
        vector_db_loaded = self.embedding_manager.load_vector_db()

        # Get repository name and owner
        repo_name = self.repo_info.get('name', '')
        repo_owner = self.repo_info.get('owner', '')

        # Possible filename patterns
        chunk_filenames = [
            "chunks.json",                      # Generic name
            f"{repo_name}_chunks.json",         # Just repo name
            f"{repo_owner}_{repo_name}_chunks.json"  # Owner and repo name
        ]

        # Possible directory locations
        possible_dirs = [
            self.embeddings_dir,
            os.path.dirname(self.embeddings_dir),
            os.path.join(os.path.dirname(self.embeddings_dir), "data")
        ]

        # Generate all possible locations
        possible_locations = []
        for directory in possible_dirs:
            for filename in chunk_filenames:
                possible_locations.append(os.path.join(directory, filename))

        # Add any additional paths you've seen in the wild
        possible_locations.extend([
            os.path.join(os.path.dirname(self.embeddings_dir), "data", "Textualize_rich_chunks.json"),
            os.path.join(os.path.dirname(self.embeddings_dir), "data", "rich_chunks.json"),
            os.path.join(os.path.dirname(self.embeddings_dir), "data", "_chunks.json")
        ])

        # For debugging
        self.logger.info(f"Looking for chunks files with patterns: {chunk_filenames}")

        chunks_loaded = False
        for location in possible_locations:
            if os.path.exists(location):
                self.logger.info(f"Found chunks file at: {location}")
                chunks_loaded = self.embedding_manager.load_chunks(location)
                break

        if not chunks_loaded:
            locations_str = "\n - ".join(possible_locations)
            self.logger.error(f"Chunks file not found in any of these locations:\n - {locations_str}")
            return False

        if not (embeddings_loaded and vector_db_loaded):
            self.logger.error("Failed to load embeddings or vector DB")
            return False

        self.logger.info("RAG engine data loaded successfully")
        return True

    def _expand_query(self, question: str) -> str:
        """
        Expand the query to improve retrieval by adding context and keywords.

        Args:
            question: Original question text

        Returns:
            Expanded query text
        """
        # Add project context
        expanded = question
        repo_name = self.repo_info.get('name', '')

        if "this project" in question.lower() or "the project" in question.lower():
            expanded = expanded.replace("this project", f"the {repo_name} project")
            expanded = expanded.replace("the project", f"the {repo_name} project")

        # Add keywords for specific question types
        if "architecture" in question.lower() or "structure" in question.lower():
            expanded += " structure organization components modules design patterns"
        elif "dependencies" in question.lower() or "requirements" in question.lower():
            expanded += " requirements.txt pyproject.toml package.json npm dependencies imports modules"
        elif "testing" in question.lower() or "tests" in question.lower():
            expanded += " tests pytest unittest test cases assertions fixtures"
        elif "documentation" in question.lower() or "docs" in question.lower():
            expanded += " documentation readme.md docs examples tutorials"
        elif "api" in question.lower() or "interface" in question.lower():
            expanded += " api endpoint interface function method class public"

        self.logger.debug(f"Expanded query: '{question}' -> '{expanded}'")
        return expanded
    def get_questions_for_role(self, role: str) -> List[str]:
        """
        Get predefined questions for a specific role.

        Args:
            role: Role (programmer, ceo, sales_manager)

        Returns:
            List of questions for the role
        """
        if role.lower() not in self.QUESTION_TEMPLATES:
            self.logger.warning(f"Unknown role: {role}, using programmer as default")
            return self.QUESTION_TEMPLATES["programmer"]

        return self.QUESTION_TEMPLATES[role.lower()]

    def retrieve_relevant_chunks(
        self, question: str, top_k: int = 5, threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks relevant to the question.

        Args:
            question: Question text
            top_k: Number of chunks to retrieve
            threshold: Minimum similarity threshold

        Returns:
            List of relevant chunks with metadata
        """
        # Expand the query to improve retrieval
        expanded_query = self._expand_query(question)

        # Search for similar chunks - get more than we need for reranking
        initial_k = min(top_k * 3, 15)  # Get 3x results but cap at 15
        results = self.embedding_manager.search_similar_chunks(expanded_query, top_k=initial_k)

        # Filter by similarity threshold
        filtered_results = [r for r in results if r["similarity"] >= threshold]

        # Rerank results
        if len(filtered_results) > 0:
            filtered_results = self.embedding_manager.rerank_search_results(filtered_results, question)

        # Take only the top_k results after reranking
        final_results = filtered_results[:top_k]

        # Log the retrieval results
        self.logger.info(
            f"Retrieved {len(final_results)} chunks for question (from {len(results)} initial matches): "
            f"\"{question[:50]}{'...' if len(question) > 50 else ''}\""
        )

        # Log the top chunks for debugging
        for i, result in enumerate(final_results[:3], 1):
            chunk = result["chunk"]
            self.logger.debug(
                f"Top result {i}: File={chunk['file_path']}, "
                f"Score={result.get('adjusted_score', result['similarity']):.4f}, "
                f"Type={chunk.get('chunk_type', 'unknown')}"
            )

        return final_results

    def _post_process_chunks(self, question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-process chunks to ensure we have relevant context for specific questions.

        Args:
            question: The original question
            chunks: The retrieved chunks

        Returns:
            Enhanced or filtered chunks
        """
        question_lower = question.lower()

        # Check for specific question types and ensure we have appropriate context
        if "programming languages" in question_lower:
            # Add repository stats if not already included
            has_language_stats = any("languages" in c["chunk"].get("content", "").lower() for c in chunks)
            if not has_language_stats and self.repo_info.get('languages'):
                repo_info = self.repo_info.get('languages', {})
                if repo_info:
                    # Create language distribution info
                    languages = list(repo_info.items())
                    languages.sort(key=lambda x: x[1], reverse=True)
                    language_text = ", ".join([f"{lang} ({pct}%)" for lang, pct in languages[:5]])

                    languages_content = f"This repository contains multiple programming languages. " \
                                        f"The primary languages used are: {language_text}"

                    # Create a synthetic chunk with this information
                    chunks.insert(0, {
                        "chunk": {
                            "content": languages_content,
                            "file_path": "repository_info/languages",
                            "chunk_type": "metadata",
                            "language": "text"
                        },
                        "similarity": 0.95,  # High relevance for this question
                        "adjusted_score": 0.95
                    })

        # For architecture questions, ensure we have high-level module information
        if "architecture" in question_lower or "structure" in question_lower:
            has_architecture_info = any("structure" in c["chunk"].get("content", "").lower() or
                                        "architecture" in c["chunk"].get("content", "").lower()
                                       for c in chunks)

            if not has_architecture_info:
                # Simply add a note about directory structure instead of accessing file system
                structure_content = f"The repository {self.repo_info.get('name', 'Unknown')} likely follows " \
                                    f"a structured layout with source code, documentation, and configuration files."

                chunks.insert(0, {
                    "chunk": {
                        "content": structure_content,
                        "file_path": "repository_info/structure",
                        "chunk_type": "metadata",
                        "language": "text"
                    },
                    "similarity": 0.9,
                    "adjusted_score": 0.9
                })

        return chunks
    def _generate_answer_with_llm(
        self, question: str, chunks: List[Dict[str, Any]], max_tokens: int = 1000
    ) -> str:
        """
        Generate answer using an LLM.

        Args:
            question: Question text
            chunks: Relevant chunks
            max_tokens: Maximum tokens for generation

        Returns:
            Generated answer
        """
        self.logger.info(f"Generating answer with LLM. Available: OpenAI={self.openai_client is not None}, Local LLM={self.local_llm is not None}")

        # Try OpenAI first if configured
        if self.openai_client:
            self.logger.info("Using OpenAI for generation")
            return self._generate_with_openai(question, chunks, max_tokens)

        # Try local LLM if available
        if self.local_llm:
            self.logger.info("Using local LLM for generation")
            return self._generate_with_local_llm(question, chunks, max_tokens)

        # Fall back to basic generation
        self.logger.warning("No LLM available, using fallback generation")
        return self._generate_answer_fallback(question, chunks)

    def _analyze_context_gaps(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Analyze chunks for potential information gaps related to the question.

        Args:
            question: Original question
            chunks: Retrieved chunks

        Returns:
            String with additional context information or empty string if no gaps
        """
        question_lower = question.lower()
        content_texts = [chunk["chunk"].get("content", "").lower() for chunk in chunks]
        file_paths = [chunk["chunk"].get("file_path", "") for chunk in chunks]

        additional_context = []

        # Check for programming languages gap
        if "programming languages" in question_lower:
            # Check if any language identifiers are present in the chunks
            common_languages = ["python", "javascript", "typescript", "java", "go", "rust", "c++", "php", "ruby"]
            languages_mentioned = set()

            for text in content_texts:
                for lang in common_languages:
                    if lang in text:
                        languages_mentioned.add(lang)

            if not languages_mentioned and self.repo_info.get('languages'):
                # Add language information from repo stats
                lang_info = self.repo_info.get('languages', {})
                if lang_info:
                    languages = sorted(lang_info.items(), key=lambda x: x[1], reverse=True)
                    primary_lang = languages[0][0] if languages else "Unknown"
                    additional_context.append(
                        f"Based on repository statistics, the project primarily uses {primary_lang}."
                    )

        # Check for architecture/structure gap
        if "architecture" in question_lower or "structure" in question_lower:
            has_architecture_info = any(
                "architecture" in text or
                "structure" in text or
                "components" in text or
                "modules" in text
                for text in content_texts
            )

            if not has_architecture_info:
                # Instead of inferring from file paths, provide a generic hint
                repo_name = self.repo_info.get('name', 'Unknown')
                additional_context.append(
                    f"The repository {repo_name} is organized with a structure typical for its primary programming language."
                )

        if additional_context:
            return "Additional context:\n" + "\n".join(additional_context)
        return ""
    def _generate_with_openai(
        self, question: str, chunks: List[Dict[str, Any]], max_tokens: int = 1000
    ) -> str:
        """Generate answer using OpenAI API with improved prompting."""
        try:
            # Import tiktoken for token counting
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

            # Prepare repository information
            repo_name = self.repo_info.get('name', 'Unknown')
            repo_owner = self.repo_info.get('owner', 'Unknown')
            repo_type = self.repo_info.get('type', 'Unknown')
            repo_languages = self.repo_info.get('languages', {})

            # Format languages if available
            languages_str = "unknown"
            if repo_languages:
                languages_list = sorted(repo_languages.items(), key=lambda x: x[1], reverse=True)
                languages_str = ", ".join([f"{lang}" for lang, pct in languages_list[:3]])

            # Create more focused system prompt based on question type
            system_prompt = f"""
            You are a technical analyst providing detailed information about the repository {repo_name} by {repo_owner}.
            
            For this specific question: "{question}"
            
            Your task is to provide a comprehensive, accurate answer based ONLY on the context provided.
            """

            # Add question-specific guidelines
            question_lower = question.lower()
            if "programming languages" in question_lower:
                system_prompt += """
                Guidelines for this language-related question:
                - Focus on identifying all programming languages used in the repository
                - Report the percentage breakdown of languages if available
                - Mention the primary language and secondary languages
                - Refer to specific files or directories that demonstrate language usage
                - If language statistics are provided, include them in your answer
                """
            elif "architecture" in question_lower or "structure" in question_lower:
                system_prompt += """
                Guidelines for this architecture-related question:
                - Describe the high-level architecture of the project
                - Identify the main directories/modules and their responsibilities
                - Explain how the components interact with each other
                - Mention any design patterns or architectural styles used
                - Focus on the organizational structure rather than implementation details
                """
            elif "components" in question_lower or "modules" in question_lower:
                system_prompt += """
                Guidelines for this component-related question:
                - List all major components/modules in the project
                - Describe the purpose of each component
                - Explain how the components are organized
                - Highlight key files or functionalities within each component
                - Include information about how components interact if available
                """
            elif "testing" in question_lower:
                system_prompt += """
                Guidelines for this testing-related question:
                - Identify the testing frameworks and tools used
                - Describe the testing approach (unit tests, integration tests, etc.)
                - Mention any test fixtures, mocks, or utilities
                - Note the test directory structure and organization
                - Highlight any test configuration files or patterns
                """
            elif "dependencies" in question_lower:
                system_prompt += """
                Guidelines for this dependency-related question:
                - List all external dependencies/libraries used by the project
                - Identify the primary/core dependencies
                - Note any dependency management tools used (npm, pip, etc.)
                - Mention any version constraints or requirements
                - Group related dependencies by their purpose if possible
                """
            elif "code quality" in question_lower or "comments" in question_lower or "documentation" in question_lower:
                system_prompt += """
                Guidelines for this code quality question:
                - Assess the level of comments and documentation in the code
                - Mention any code formatting tools or linters being used
                - Note the presence of docstrings, type hints, or API docs
                - Evaluate consistency in coding style and patterns
                - Look for evidence of documentation practices or standards
                """
            elif "bugs" in question_lower or "issues" in question_lower:
                system_prompt += """
                Guidelines for this bug/issue question:
                - Identify any explicitly mentioned bugs or issues
                - Look for TODO comments, FIXME notes, or issue references
                - Note any issue templates or bug reporting procedures
                - Mention any open issues or known limitations documented
                - Avoid speculation about potential bugs not mentioned in the context
                """

            # Default guidelines for all question types

            system_prompt += """
                General guidelines:
                - Base your answer ONLY on the provided context
                - If the context doesn't contain enough information, say so clearly
                - Cite specific file names when relevant
                - Organize your answer in a clear, structured format
                - Be specific and comprehensive
                - Focus on factual information from the repository
            """
            role = self.repo_info.get("role", "").lower()
            if role in ["ceo", "sales", "sales_manager", "marketing"]:
                system_prompt += """
                Additional role-specific instruction:
                Since this report is intended for a CEO or business-facing role, do not include any code snippets or overly technical implementation details.
                Focus instead on architecture, design strategy, business value, maintainability, scalability, and product impact.
                Present findings in conceptual, strategic terms.
                """
            else:
                system_prompt += """
                If relevant, include short code snippets in markdown-style code blocks (```), and ensure the syntax is consistent with the detected language.
                Choose snippets that help clarify your explanation, such as showing key decorators, class definitions, or test cases.
                """
            # Count tokens in the system prompt
            system_tokens = len(encoding.encode(system_prompt))
            question_tokens = len(encoding.encode(question))

            # Set maximum context tokens (leaving room for response)
            max_context_tokens = 8000 - system_tokens - question_tokens - max_tokens

            # Group chunks by type for better organization
            code_chunks = []
            doc_chunks = []
            config_chunks = []
            other_chunks = []

            for chunk_data in chunks:
                chunk = chunk_data["chunk"]
                chunk_type = chunk.get("chunk_type", "unknown")

                if chunk_type == "code":
                    code_chunks.append(chunk_data)
                elif chunk_type == "documentation":
                    doc_chunks.append(chunk_data)
                elif chunk_type == "configuration":
                    config_chunks.append(chunk_data)
                else:
                    other_chunks.append(chunk_data)

            # Prepare context from chunks, organizing by type
            context_parts = []
            current_tokens = 0

            # Add documentation context first (helps with conceptual understanding)
            if doc_chunks:
                context_parts.append("\n## DOCUMENTATION CONTEXT")
                for i, chunk_data in enumerate(doc_chunks[:3], 1):  # Limit to top 3 doc chunks
                    chunk = chunk_data["chunk"]
                    chunk_text = f"\nDOC {i} - File: {chunk['file_path']}"
                    if chunk.get('name'):
                        chunk_text += f"\nSection: {chunk['name']}"

                    # Truncate content if needed
                    content = chunk['content']
                    if len(content) > 1000:
                        content = content[:1000] + "... [truncated]"

                    chunk_text += f"\n{content}\n"

                    # Count tokens and add if within limit
                    chunk_tokens = len(encoding.encode(chunk_text))
                    if current_tokens + chunk_tokens <= max_context_tokens:
                        context_parts.append(chunk_text)
                        current_tokens += chunk_tokens
                    else:
                        break

            # Add code context next (implementation details)
            if code_chunks and current_tokens < max_context_tokens:
                context_parts.append("\n## CODE IMPLEMENTATION CONTEXT")
                for i, chunk_data in enumerate(code_chunks[:5], 1):  # Limit to top 5 code chunks
                    chunk = chunk_data["chunk"]
                    chunk_text = f"\nCODE {i} - File: {chunk['file_path']}"

                    if chunk.get('name'):
                        if chunk.get('parent'):
                            chunk_text += f"\nDef: {chunk['parent']}.{chunk['name']}"
                        else:
                            chunk_text += f"\nDef: {chunk['name']}"

                    # Truncate content if needed
                    content = chunk['content']
                    if len(content) > 1500:  # Allow code chunks to be a bit larger
                        content = content[:1500] + "... [truncated]"

                    chunk_text += f"\n```\n{content}\n```\n"

                    # Count tokens and add if within limit
                    chunk_tokens = len(encoding.encode(chunk_text))
                    if current_tokens + chunk_tokens <= max_context_tokens:
                        context_parts.append(chunk_text)
                        current_tokens += chunk_tokens
                    else:
                        break

            # Add configuration context last
            if config_chunks and current_tokens < max_context_tokens:
                context_parts.append("\n## CONFIGURATION CONTEXT")
                for i, chunk_data in enumerate(config_chunks[:2], 1):  # Limit to top 2 config chunks
                    chunk = chunk_data["chunk"]
                    chunk_text = f"\nCONFIG {i} - File: {chunk['file_path']}"

                    # Truncate content if needed
                    content = chunk['content']
                    if len(content) > 800:
                        content = content[:800] + "... [truncated]"

                    chunk_text += f"\n{content}\n"

                    # Count tokens and add if within limit
                    chunk_tokens = len(encoding.encode(chunk_text))
                    if current_tokens + chunk_tokens <= max_context_tokens:
                        context_parts.append(chunk_text)
                        current_tokens += chunk_tokens
                    else:
                        break

            # Add any other chunks if there's still room
            if other_chunks and current_tokens < max_context_tokens:
                context_parts.append("\n## OTHER CONTEXT")
                for i, chunk_data in enumerate(other_chunks[:2], 1):
                    chunk = chunk_data["chunk"]
                    chunk_text = f"\nOTHER {i} - File: {chunk['file_path']}"

                    # Truncate content
                    content = chunk['content']
                    if len(content) > 500:
                        content = content[:500] + "... [truncated]"

                    chunk_text += f"\n{content}\n"

                    # Count tokens and add if within limit
                    chunk_tokens = len(encoding.encode(chunk_text))
                    if current_tokens + chunk_tokens <= max_context_tokens:
                        context_parts.append(chunk_text)
                        current_tokens += chunk_tokens
                    else:
                        break

            context_text = "\n".join(context_parts)

            # Final prompt
            user_prompt = f"""
            Question: {question}
            
            Here is the relevant information from the repository:
            
            {context_text}
            
            Answer the question based on this information. Be specific and reference relevant files or code when appropriate.
            """

            # Calculate final token count
            total_input_tokens = system_tokens + len(encoding.encode(user_prompt))
            self.logger.info(f"Total input tokens: {total_input_tokens}")

            # If still too large, further reduce
            if total_input_tokens > 15000:  # OpenAI's limit is higher, but be safe
                self.logger.warning(f"Input still too large ({total_input_tokens} tokens), reducing further")
                return self._generate_answer_fallback(question, chunks)

            # Call OpenAI API
            start_time = time.time()
            with open("retrieved_chunks_all.txt", "a", encoding="utf-8") as f:
                f.write(f"\n\n====== Retrieved Chunks for Question: \"{question}\" ======\n\n")
                for i, result in enumerate(chunks):
                    inner = result.get("chunk", {})
                    file = inner.get("file_path", "unknown")
                    content = inner.get("content", "<empty>")
                    f.write(f"\n📦 Chunk {i+1} - File: {file}\n")
                    f.write("-" * 60 + "\n")
                    f.write(content[:800] + "\n")
                    f.write("-" * 60 + "\n")

            with open("diagnose_chunk_format.json", "w", encoding="utf-8") as f:
                json.dump(chunks[0], f, indent=2)
            response = self.openai_client.chat.completions.create(
                #model="gpt-3.5-turbo",  # Using a smaller model to reduce token costs
                 model=self.qa_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.2
            )

            elapsed = time.time() - start_time

            # Extract answer from response
            answer = response.choices[0].message.content
            answer_text = response.choices[0].message.content.strip()
            # ✅ Log both the question and generated answer
            with open("debug_phase1_answers.txt", "a", encoding="utf-8") as f:
                f.write(f"Q: {question}\n")
                f.write(f"A: {answer_text}\n")
                f.write("=" * 80 + "\n")
            # Log token usage
            usage = response.usage
            self.logger.info(
                f"Token usage - Prompt: {usage.prompt_tokens}, "
                f"Completion: {usage.completion_tokens}, "
                f"Total: {usage.total_tokens}"
            )

            self.logger.info(
                f"Generated answer with OpenAI in {elapsed:.2f}s, "
                f"{len(answer)} chars"
            )

            return answer

        except Exception as e:
            self.logger.error(f"Error generating answer with OpenAI: {e}", exc_info=True)
            return self._generate_answer_fallback(question, chunks)
    def _generate_with_local_llm(
        self, question: str, chunks: List[Dict[str, Any]], max_tokens: int = 1000
    ) -> str:
        """
        Generate answer using local LLM.

        Args:
            question: Question text
            chunks: Relevant chunks
            max_tokens: Maximum tokens for generation

        Returns:
            Generated answer
        """
        try:
            # Prepare context from chunks
            context_parts = []
            for i, chunk_data in enumerate(chunks, 1):
                chunk = chunk_data["chunk"]
                context_parts.append(f"[Document {i}]")
                context_parts.append(f"File: {chunk['file_path']}")
                if chunk.get('name'):
                    context_parts.append(f"Name: {chunk['name']}")
                context_parts.append(f"Content:\n{chunk['content']}\n")

            context_text = "\n".join(context_parts)

            # Generate answer with local LLM
            start_time = time.time()

            self.logger.info(f"Generating answer with local LLM for question: {question[:50]}...")
            answer = self.local_llm.generate_answer(
                question=question,
                context=context_text,
                max_tokens=max_tokens,
                temperature=0.2
            )

            elapsed = time.time() - start_time

            self.logger.info(
                f"Generated answer with local LLM in {elapsed:.2f}s, "
                f"{len(answer)} chars"
            )

            return answer

        except Exception as e:
            self.logger.error(f"Error generating answer with local LLM: {e}", exc_info=True)
            return self._generate_answer_fallback(question, chunks)

    def _generate_answer_fallback(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a simple answer without LLM when OpenAI is not available.

        Args:
            question: Question text
            chunks: Relevant chunks

        Returns:
            Generated answer
        """
        # Create a basic response from the chunks
        lines = [f"Answer to: {question}\n"]
        lines.append("Based on the repository information:\n")

        for i, chunk_data in enumerate(chunks[:3], 1):  # Limit to top 3 chunks
            chunk = chunk_data["chunk"]
            lines.append(f"Source {i}: {chunk['file_path']}")

            if chunk.get('name'):
                lines.append(f"Name: {chunk['name']}")

            # Truncate content
            content = chunk['content']
            if len(content) > 300:
                content = content[:300] + "..."

            lines.append(f"Content excerpt:\n{content}\n")

        lines.append("\nNote: This is a simplified answer without using an LLM. For more comprehensive answers, please set up OpenAI API access.")

        return "\n".join(lines)

    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a question about the repository with improved context handling.

        Args:
            question: Question text
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with question, answer, and supporting chunks
        """
        start_time = time.time()
        self.logger.info(f"Processing question: {question}")

        # Retrieve relevant chunks
        chunks = self.retrieve_relevant_chunks(question, top_k=top_k)

        # Post-process chunks to ensure we have relevant context
        chunks = self._post_process_chunks(question, chunks)

        # Analyze for information gaps
        gap_context = self._analyze_context_gaps(question, chunks)

        if not chunks:
            answer = "I couldn't find relevant information in the repository to answer this question."
            self.logger.warning(f"No relevant chunks found for question: {question}")
        else:
            # Add gap information to chunks if needed
            if gap_context:
                chunks.insert(0, {
                    "chunk": {
                        "content": gap_context,
                        "file_path": "repository_info/analysis",
                        "chunk_type": "metadata",
                        "language": "text"
                    },
                    "similarity": 0.9,
                    "adjusted_score": 0.9
                })

            # Generate answer
            answer = self._generate_answer_with_llm(question, chunks)

        elapsed = time.time() - start_time

        # Format result
        result = {
            "question": question,
            "answer": answer,
            "supporting_chunks": [c["chunk"] for c in chunks],
            "processing_time": elapsed
        }

        self.logger.info(f"Answered question in {elapsed:.2f}s")
        return result

    def process_role_questions(self, role: str) -> List[Dict[str, Any]]:
        """
        Process all questions for a specific role.

        Args:
            role: Role (programmer, ceo, sales_manager)

        Returns:
            List of question-answer dictionaries
        """
        questions = self.get_questions_for_role(role)

        self.logger.info(f"Processing {len(questions)} questions for role: {role}")
        start_time = time.time()

        results = []
        for i, question in enumerate(questions, 1):
            self.logger.info(f"Question {i}/{len(questions)}: {question}")

            result = self.answer_question(question)
            results.append(result)

        elapsed = time.time() - start_time
        self.logger.info(f"Processed all {len(questions)} questions in {elapsed:.2f}s")

        return results

    def generate_report_data(self, role: str) -> Dict[str, Any]:
        """
        Generate data for a report.

        Args:
            role: Role (programmer, ceo, sales_manager)

        Returns:
            Dictionary with report data
        """
        # Process role questions
        qa_results = self.process_role_questions(role)

        # Format report data
        report_data = {
            "repository": self.repo_info,
            "role": role,
            "qa_pairs": qa_results,
            "generation_time": sum(r["processing_time"] for r in qa_results),
            "timestamp": time.time()
        }

        # Save report data to file
        output_dir = os.path.join(self.embeddings_dir, "reports")
        os.makedirs(output_dir, exist_ok=True)

        repo_name = self.repo_info.get('name', 'unknown')
        output_file = os.path.join(
            output_dir,
            f"{repo_name}_{role}_report_{int(time.time())}.json"
        )

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Report data saved to {output_file}")

        return report_data


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Answer questions about repositories using RAG")
    parser.add_argument("--embeddings-dir", required=True, help="Directory containing embeddings")
    parser.add_argument("--repo-info", required=True, help="JSON file with repository info")
    parser.add_argument("--role", default="programmer", choices=["programmer", "ceo", "sales_manager"],
                      help="Role perspective for analysis")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI for generation")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--question", help="Single question to answer (optional)")

    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)

    # Load repo info
    with open(args.repo_info, 'r', encoding='utf-8') as f:
        repo_info = json.load(f)

    # Initialize RAG engine
    rag_engine = RAGEngine(
        embeddings_dir=dirs["embeddings"],
        repo_info=repo_info,
        use_openai=args.use_openai,
        use_local_llm=args.use_local_llm,
        local_llm_path=args.local_llm_path,
        local_llm_type=args.local_llm_type,
        log_level=log_level
    )


    # Load data
    if not engine.load_data():
        print("Failed to load data. Exiting.")
        exit(1)

    if args.question:
        # Answer single question
        result = engine.answer_question(args.question)

        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nBased on {len(result['supporting_chunks'])} chunks in {result['processing_time']:.2f}s")
    else:
        # Process all questions for the role
        report_data = engine.generate_report_data(args.role)

        print(f"\nGenerated report data for {args.role} role")
        print(f"Repository: {report_data['repository']['name']}")
        print(f"Processed {len(report_data['qa_pairs'])} questions")
        print(f"Total processing time: {report_data['generation_time']:.2f}s")