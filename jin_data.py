import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document


# 로컬 문서 경로 설정
PANDAS_DOC_PATH = Path("pandas/doc/source")
SKLEARN_DOC_PATH = Path("scikit-learn/doc")

print(f"PANDAS_DOC_PATH = {PANDAS_DOC_PATH}")
print(f"SKLEARN_DOC_PATH = {SKLEARN_DOC_PATH}")


# 문서 로더 함수
def load_documents_from_dir(path: Path, glob="**/*", allowed_exts=(".rst", ".md", ".txt", ".html")):
    """디렉터리 내 허용 확장자 파일만 개별 로드합니다. 실패 시 텍스트 직접 읽기로 폴백."""
    if not path.exists():
        print(f"경고: 경로가 존재하지 않습니다: {path}")
        return []

    files = [p for p in path.rglob(glob) if p.is_file() and p.suffix.lower() in allowed_exts]
    if not files:
        print(f"경고: 로드할 파일이 없습니다(확장자 필터 적용). 경로: {path}")
        return []

    docs = []
    lib_tag = "pandas" if "pandas" in str(path) else "sklearn"
    for f in files:
        try:
            loader = UnstructuredFileLoader(str(f), autodetect_encoding=True)
            loaded = loader.load()
            # loader.load()가 리스트 반환을 보장하지 않을 수 있으니 안전 처리
            if isinstance(loaded, list):
                docs.extend(loaded)
            elif loaded is not None:
                docs.append(loaded)
        except Exception as e:
            print(f"UnstructuredFileLoader 실패: {f} -> {e}. 텍스트로 직접 읽기 시도...")
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                try:
                    text = f.read_text(encoding="latin-1")
                except Exception as e2:
                    print(f"직접 읽기 실패: {f} -> {e2}. 건너뜁니다.")
                    continue
            docs.append(Document(page_content=text, metadata={"source": str(f)}))

    for d in docs:
        d.metadata["library"] = lib_tag
    print(f"로드 완료: {len(docs)} documents from {path}")
    return docs


# 청크 분할 함수
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_documents(documents)

# 임베딩 모델 및 persist 경로
EMB_MODEL = "text-embedding-3-small"
PERSIST_BASE = Path("./chromadb")
PANDAS_CHROMA_DIR = PERSIST_BASE / "pandas_rst"
SKLEARN_CHROMA_DIR = PERSIST_BASE / "sklearn_rst"
COMBINED_CHROMA_DIR = PERSIST_BASE / "both_rst"

embeddings = OpenAIEmbeddings(model=EMB_MODEL)

# 로드 및 빌드
pandas_docs = load_documents_from_dir(PANDAS_DOC_PATH, glob="**/*.rst")
sklearn_docs = load_documents_from_dir(SKLEARN_DOC_PATH, glob="**/*")

if not pandas_docs and not sklearn_docs:
    raise RuntimeError("둘 다 문서가 없습니다. 경로 및 파일을 확인하세요.")

# 분할
pandas_chunks = split_documents(pandas_docs) if pandas_docs else []
sklearn_chunks = split_documents(sklearn_docs) if sklearn_docs else []

# 생성/저장 함수 (존재하면 덮어쓰기 경고 없이 재생성)
def create_and_persist_chroma(chunks, persist_dir: Path):
    if not chunks:
        print(f"생성할 청크가 없습니다: {persist_dir}")
        return None
    persist_dir.mkdir(parents=True, exist_ok=True)
    vs = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(persist_dir)
    )
    print(f"Chroma DB 생성 완료: {persist_dir} (문서 수: {len(chunks)})")
    return vs


# 1) pandas 전용 DB
pandas_vs = create_and_persist_chroma(pandas_chunks, PANDAS_CHROMA_DIR)

# 2) sklearn 전용 DB
sklearn_vs = create_and_persist_chroma(sklearn_chunks, SKLEARN_CHROMA_DIR)

# 3) 둘을 합친 combined DB (두 컬렉션 모두 사용)
combined_chunks = pandas_chunks + sklearn_chunks
combined_vs = create_and_persist_chroma(combined_chunks, COMBINED_CHROMA_DIR)

print("생성된 DB 경로:")
print("- pandas :", PANDAS_CHROMA_DIR if pandas_vs else "없음")
print("- sklearn:", SKLEARN_CHROMA_DIR if sklearn_vs else "없음")
print("- both   :", COMBINED_CHROMA_DIR if combined_vs else "없음")


# 테스트 검색 (간단 확인)
def test_search(vs, query="Pandas에서 누락된 값"):
    if vs is None:
        print("Vectorstore 없음 - 테스트 불가")
        return
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    print(f"쿼리: {query} -> 결과 {len(docs)}개")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.metadata.get('source','알 수 없음')} [{d.metadata.get('library','')}]")
        print(d.page_content[:200].replace("\n", " ") + "...")
    print("----")

print("\n테스트 검색 (pandas):")
test_search(pandas_vs, "누락된 값 확인 방법")
print("\n테스트 검색 (sklearn):")
test_search(sklearn_vs, "scikit-learn estimator 사용법")
print("\n테스트 검색 (combined):")
test_search(combined_vs, "DataFrame과 estimator 연동")