import os
import pypandoc
import uuid
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

SOURCE_CONFIG = {
    # Pandas  
    "pandas": {
        "path": "../Docs/Pandas_doc_233",
        "file_types": [".rst",".py", ".md"]
    },
    # Scikit_learn
    "scikit_learn": {
        "path": "../Docs/Scikit_Learn_doc_172", 
        "file_types": [".rst",".py", ".md"]
    },
    # Pytorch
    "pytorch": {
        "path": "../Docs/Pytorch_doc_290",
        "file_types": [".rst",".py", ".md"]
    },
    # Python
    "python": {
        "path": "../Docs/Python_doc_314",
        "file_types": [".rst",".py", ".md"]
    }
}


PERSIST_DIRECTORY = "./chroma_db_multi_source"

def load_and_process_docs(config):

    all_documents = []
    print("Starting document processing...")

    for metadata_key, settings in config.items():
        source_path = settings["path"]
        allowed_types = settings["file_types"]
        
        if not os.path.exists(source_path) or "[!]" in source_path:
            print(f"[Warning] Path is invalid or not set, skipping: {source_path}")
            continue

        print(f"Processing: '{metadata_key}' from path: {source_path}")

        for root, dirs, files in os.walk(source_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1]

                # 허용된 파일 확장자인지 확인
                if file_ext not in allowed_types:
                    continue

                try:
                    page_content = ""
                    
                    if file_ext == ".rst":
                        page_content = pypandoc.convert_file(
                            file_path, 'plain', format='rst', extra_args=['--wrap=none']
                        )
                    elif file_ext == ".md" or file_ext == ".py":
                        with open(file_path, 'r', encoding='utf-8') as f:
                            page_content = f.read()
                    
                    if page_content.strip():
                        
                        doc = Document(
                            page_content=page_content,
                            metadata={
                                "source_type": metadata_key, # 예: "pandas"
                                "source_file": file_path
                            }
                        )
                        all_documents.append(doc)

                except Exception as e:
                    print(f"  [Error] Failed to process file {file_path}: {e}")

    print(f"Total documents loaded before splitting: {len(all_documents)}")
    return all_documents

def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    all_docs = load_and_process_docs(SOURCE_CONFIG)

    if not all_docs:
        print("No documents were loaded. Check paths in SOURCE_CONFIG.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Total chunks created after splitting: {len(split_docs)}")

    embedding_func = OpenAIEmbeddings(model='text-embedding-3-small')
    
    print(f"Creating persistent vector store at {PERSIST_DIRECTORY} using cosine similarity...")
    
    vectorstore = Chroma.from_documents(
        split_docs, 
        embedding_func,
        collection_metadata={"hnsw:space": "cosine"}, # 코사인 유사도
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"Ingestion complete. Vector store created at '{PERSIST_DIRECTORY}'")
    print(f"Total chunks stored: {vectorstore._collection.count()}")

if __name__ == "__main__":
    main()