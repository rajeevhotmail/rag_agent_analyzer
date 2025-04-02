
    #!/usr/bin/env python3
"""
Repository Analysis Tool

This script analyzes GitHub or GitLab repositories based on a specified role
and provides answers to predefined questions in a PDF document.

Usage:
    python main_repo_analyst.py --url <repository_url> --role <role> [--persistent]
    python main_repo_analyst.py --local-path <repository_path> --role <role>

Example:
    python main_repo_analyst.py --url https://github.com/fastapi-users/fastapi-users --role programmer --persistent
"""
import argparse
import os
import sys
import json
import logging
import time
from pathlib import Path

from repo_fetcher import RepoFetcher
from content_processor import ContentProcessor
from embeddings_manager import EmbeddingsManager
from rag_engine import RAGEngine
from narrative_stitcher import extract_narrative_and_key_findings
from narrative_agent import NarrativeAgent
from weasy_pdf_writer import WeasyPDFWriter
from urllib.parse import urlparse

# Define questions for each role
QUESTIONS = {
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze GitHub/GitLab repositories")
    parser.add_argument("--url")
    parser.add_argument("--local-path")
    parser.add_argument("--role", required=True, choices=QUESTIONS.keys())
    parser.add_argument("--github-token")
    parser.add_argument("--gitlab-token")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--existing-json")
    parser.add_argument("--use-local-llm", action="store_true")
    parser.add_argument("--local-llm-path")
    parser.add_argument("--local-llm-type", default="llama2", choices=["llama2", "codellama"])
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-process", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--skip-rag",  action="store_true",    help="Skip RAG generation and reuse existing report_data.json")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--output", default="report.pdf")

    args = parser.parse_args()

    if not args.url and not args.local_path and not (args.skip_fetch and args.skip_process):
        parser.error("Either --url or --local-path is required")
    return args

def get_repo_key(url=None, local_path=None):
    if url and url.startswith("http"):
        parsed = urlparse(url)
        path = parsed.path.strip("/")  # e.g., andrivet/python-asn1
        return path.replace("/", "_").replace(".git", "")
    elif local_path:
        # Check if .git/config exists and parse remote origin
        try:
            repo = Repo(local_path)
            remote_url = next(repo.remote().urls)
            return get_repo_key(url=remote_url)
        except Exception:
            # fallback to folder name
            return os.path.basename(os.path.normpath(local_path))
    else:
        raise ValueError("Must provide --url or --local-path")

def setup_logging(log_level):
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", f"repo_analysis_{int(time.time())}.log")
    logging.basicConfig(level=getattr(logging, log_level),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset.timer").setLevel(logging.WARNING)
    return logging.getLogger("main")

def setup_directories(base_dir, repo_name):
    repo_dir = os.path.join(base_dir, repo_name)
    data_dir = os.path.join(repo_dir, "data")
    embeddings_dir = os.path.join(repo_dir, "embeddings")
    reports_dir = os.path.join(repo_dir, "reports")
    for d in [repo_dir, data_dir, embeddings_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)
    return {"repo": repo_dir, "data": data_dir, "embeddings": embeddings_dir, "reports": reports_dir}

def fetch_repository(args, logger):
    logger.info("Fetching repository...")
    with RepoFetcher(github_token=args.github_token, gitlab_token=args.gitlab_token) as fetcher:
        repo_path = fetcher.fetch_repo(url=args.url or "local", local_path=args.local_path, persistent=args.persistent)
        repo_info = fetcher.get_basic_repo_info()
        return repo_path, repo_info

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_repository_content(repo_path, data_dir, log_level):
    processor = ContentProcessor(repo_path, log_level=log_level)
    chunks = processor.process_repository()
    processor.chunks = chunks  # Required!
    return processor.save_chunks(data_dir)

def generate_embeddings(embeddings_dir, chunks_file, log_level):
    manager = EmbeddingsManager(output_dir=embeddings_dir, log_level=log_level)
    manager.load_chunks(chunks_file)
    return manager.generate_embeddings()

def generate_report_data(args, rag_engine, logger):
    if args.existing_json:
        logger.info("Loading existing report JSON...")
        return load_json(args.existing_json)
    if not rag_engine.load_data():
        logger.error("Failed to load data for RAG engine")
        sys.exit(1)
    return rag_engine.generate_report_data(args.role)

def generate_pdf(report_data):
    agent = NarrativeAgent(role=report_data["role"], repo_name=report_data["repository"]["name"],
                           qa_pairs=report_data["qa_pairs"], model="gpt-3.5-turbo")
    narrative, findings = extract_narrative_and_key_findings(agent.build_narrative())
    writer = WeasyPDFWriter()
    return writer.write_pdf(narrative, report_data["repository"]["name"], report_data["role"], findings)

def main():
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    logger.info("Starting repository analysis")
    try:
        repo_path, repo_info = None, None

        if not args.skip_fetch:
            # Fetch the repository
            repo_path, repo_info = fetch_repository(args, logger)

            # Use the name from repo_info to derive a consistent repo_key
            repo_key = get_repo_key(url=args.url, local_path=args.local_path)

            # Setup directory structure
            dirs = setup_directories(args.output_dir, repo_key)

            # Save repo_info.json into data dir
            repo_info_path = os.path.join(dirs["data"], "repo_info.json")
            with open(repo_info_path, "w") as f:
                json.dump(repo_info, f, indent=2)

        else:
            # Skip fetch: infer repo_key from local path or URL
            repo_key = get_repo_key(url=args.url, local_path=args.local_path)
            dirs = setup_directories(args.output_dir, repo_key)

            # Load repo_info.json from data dir
            repo_info_path = os.path.join(dirs["data"], "repo_info.json")
            if not os.path.exists(repo_info_path):
                logger.error("Missing repo_info.json and --skip-fetch was used")
                sys.exit(1)

            with open(repo_info_path, "r") as f:
                repo_info = json.load(f)
            repo_path = args.local_path

        # Process the repo content into chunks
        if not args.skip_process:
            chunks_file = process_repository_content(repo_path, dirs["data"], args.log_level)
        else:
            chunks_file = os.path.join(dirs["data"], "_chunks.json")
            if not os.path.exists(chunks_file):
                logger.error("Missing chunks file and --skip-process was used")
                sys.exit(1)

        # Generate embeddings
        if not args.skip_embed:
            generate_embeddings(dirs["embeddings"], chunks_file, args.log_level)

        # Initialize RAG engine
        rag_engine = RAGEngine(
            embeddings_dir=dirs["embeddings"],
            repo_info=repo_info,
            use_openai=args.use_openai,
            use_local_llm=args.use_local_llm,
            local_llm_path=args.local_llm_path,
            local_llm_type=args.local_llm_type,
            log_level=args.log_level
        )

        # Generate report data and save it
        report_data_file = os.path.join(dirs["repo"], "report_data.json")
        if args.skip_rag:
            if not os.path.exists(report_data_file):
                logger.error("Missing report_data.json and --skip-rag was used")
                sys.exit(1)
            report_data = load_json(report_data_file)
            logger.info("✅ Loaded existing report_data.json")
        else:
            report_data = generate_report_data(args, rag_engine, logger)
            save_json(report_data_file, report_data)
            logger.info("💾 Saved new report_data.json")

        # Generate PDF report
        pdf_path = generate_pdf(report_data)
        print(f"PDF saved to: {pdf_path}")

    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
