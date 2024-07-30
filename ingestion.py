import os
import shutil

from dotenv import load_dotenv
from git import Repo
from langchain_community.document_loaders import GitLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_text_splitters import CharacterTextSplitter, MarkdownHeaderTextSplitter


def ingest_git_wiki_repo(repo_url):
    print(f"Ingesting git repo: {repo_url}")
    temp_dir = "/tmp/clone"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Directory {temp_dir} has been deleted.")

    os.makedirs(temp_dir, exist_ok=True)
    print("Cloning wiki repository...")
    repo = Repo.clone_from(repo_url, "/tmp/clone")
    branch = repo.head.reference
    print(f"Cloned repo: {repo_url} to {temp_dir}")

    print("Loading documents...")
    loader = GitLoader(repo_path="/tmp/clone", branch=branch.name)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6")
    ]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    docs = loader.load()
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_text(doc.page_content))
    embedding = OpenAIEmbeddings()
    print(f"Documents loaded and split to ${len(docs)} chunks.")

    print("Ingesting documents...")
    Qdrant.from_documents(
        chunks,
        embedding=embedding,
        url="http://localhost:6333",
        api_key="qdrant",
        collection_name="wiki"
    )
    print("Documents ingested successfully.")


load_dotenv()

if __name__ == "__main__":
    print("Hello")
    ingest_git_wiki_repo("")
