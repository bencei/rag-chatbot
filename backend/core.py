import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import Qdrant


def qa(query: str, chat_history: List[Dict[str, Any]] = []):
    print("Prompting...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    print("Retrieving...")
    vectorstore = Qdrant.from_existing_collection(
        embedding=embeddings,
        collection_name="wiki",
        url="http://localhost:6333",
        api_key="qdrant"
    )

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=vectorstore.as_retriever(), prompt=rephrase_prompt
    )
    retrieval_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query, "chat_history": chat_history})
    response = {
        "query": query,
        "result": result["answer"],
        "source_documents": result["context"]
    }
    return response


load_dotenv()

if __name__ == "__main__":
    res = qa("What is the Merlin PaaS?")
    print(res["answer"])
