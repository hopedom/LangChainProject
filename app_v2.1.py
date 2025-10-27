import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_chroma import Chroma
from openai import AuthenticationError
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.documents import Document
from operator import itemgetter
import json


# -----------------------------------------------------------
# 0. 초기 설정 및 환경 변수 로드
# -----------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API 키가 설정되지 않았습니다. .env 파일 또는 환경 변수를 확인하세요.")
    st.stop()

# -----------------------------------------------------------
# 1. 모델/DB 캐싱 함수
# -----------------------------------------------------------
@st.cache_resource
def initialize_models():
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0, openai_api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    embeddings.embed_query("test")  # API 키 인증 확인
    return llm, embeddings

@st.cache_resource
def load_vectorstore_and_retrievers(embeddings, k_value=10):
    PERSIST_DIRECTORY = r'C:\Users\Hopedom\Documents\DS5-LangChain\Langchain-RAG\chromadb\pandas_rst'
    try:
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        if vectorstore._collection.count() < 10:
            st.warning("🚨 벡터 저장소 문서가 부족합니다. DB 구축을 먼저 진행해 주세요.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Chroma DB 로드 중 오류 발생: {e}")
        st.stop()

    pandas_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k_value, "filter": {"library": "pandas"}}
    )
    sklearn_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k_value, "filter": {"library": "scikit-learn"}}
    )

    def combine_docs(result: dict) -> List[Document]:
        return result['pandas_docs'] + result['sklearn_docs']

    combined_retriever = RunnableParallel(
        pandas_docs=pandas_retriever,
        sklearn_docs=sklearn_retriever
    ) | RunnableLambda(combine_docs)

    return {
        'Pandas': pandas_retriever,
        'Scikit-learn': sklearn_retriever,
        '둘 다': combined_retriever
    }

# -----------------------------------------------------------
# 2. RAG Chain 구성 & 헬퍼
# -----------------------------------------------------------
def format_docs(docs: List[Document]) -> str:
    formatted_docs = []
    for doc in docs:
        domain = doc.metadata.get('library', 'General')
        source = doc.metadata.get('source', 'N/A')
        formatted_docs.append(f"출처 도메인: {domain}\n파일: {source}\n내용: {doc.page_content}")
    return "\n\n".join(formatted_docs)

def get_llm_chain(llm):
    SYSTEM_INSTRUCTION = (
        "You are a professional Python code generation and technical documentation assistant. "
        "Your primary goal is to generate accurate, well-structured Korean answers based **ONLY** on the provided context, "
        "while intelligently handling comparison or partial-context situations."
        
        "---"
        
        "**[RULES]**"
        "1. **Context Relevance:** Always prioritize information derived from the given context. "
        "If the context is fully irrelevant to the question, respond ONLY with: "
        "'죄송합니다. 현재 검색된 문서의 내용이 질문과 관련이 없어 답변을 생성할 수 없습니다. (No Relevant Context)'. "
        
        "2. **Partial Context Mode:** If the context partially covers the question, "
        "generate an answer using the available context and clearly state which parts are missing or generalized."
        
        "3. **Partial Comparison Mode:** If the question includes comparison intent "
        "(e.g., contains 'vs', 'difference', '비교'), "
        "and the context covers only one side (e.g., only Pandas), "
        "you may combine general domain knowledge to complete the comparison. "
        "Always clarify which portions are from the context and which are from general knowledge."
        
        "4. **Categorical Encoding Comparison:** If the question or context mentions '원핫 인코딩', 'get_dummies', or 'OneHotEncoder', "
        "provide a detailed comparison among these methods."
        
        "5. **General Answer:** For all other relevant questions, generate a concise, structured, and accurate answer."

        "---"
        "**[ADAPTIVE FORMAT RULES]**"
        "• Comparison Questions: summary + Markdown table + key takeaways"
        "• Function/Method Questions: short summary + numbered list + code examples"
        "• Conceptual Questions: concise explanation (2-3 sentences) + optional example"
        "• Code-focused Questions: minimal text + runnable code in Markdown"
        "• General Questions: structured explanation using bold headings + numbered lists"
        "---"
        "**[STRUCTURE & FORMAT]**"
        "A. Explanation: 2-3 sentence Korean summary"
        "B. Comparison Table (if applicable)"
        "C. Structure: bold headings + numbered lists"
        "D. Code Examples: runnable Python blocks"
        "E. Formatting: Markdown, horizontal rules (`---`) for sections"
        "F. Context Attribution: mark general knowledge vs context-derived content"
        "G. Metadata: do not output source domains, file paths, or previous conversation remnants"
        "---"
        "**[EXAMPLE BEHAVIOR]**"
        "• Question: '원핫 인코딩' | Context: Pandas + Scikit-learn → "
        "Provide summary, Pandas `get_dummies` example, Scikit-learn `OneHotEncoder` example, comparison table, "
        "label context vs general knowledge parts."
        "\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INSTRUCTION),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()

# -----------------------------------------------------------
# 2-1. Self-Evaluation & Auto-Rewrite
# -----------------------------------------------------------
def self_evaluate_response(llm, context: str, question: str, answer: str) -> dict:
    eval_prompt = """
    아래 답변을 평가하세요. 정확성, 완전성, 맥락 적합성을 고려해 0~100 점수와 간단한 코멘트를 제공해주세요.

    질문: {question}
    문맥: {context}
    답변: {answer}

    JSON 형태 출력:
    {{ "score": 숫자, "comment": "평가 코멘트" }}
    """

    eval_chain = ChatPromptTemplate.from_messages([
        ("system", "You are an expert evaluator for Python documentation and code explanations."),
        ("human", eval_prompt)
    ]) | llm | StrOutputParser()

    result = eval_chain.invoke({"context": context, "question": question, "answer": answer})
    try:
        return json.loads(result)
    except Exception:
        return {"score": None, "comment": result}

def auto_rewrite_response(llm, context: str, question: str, answer: str) -> str:
    rewrite_prompt = """
    이전 답변을 개선하여 정확하고 완전하게 작성해주세요.
    이전 답변: {answer}
    질문: {question}
    문맥: {context}
    새롭게 작성된 답변만 Markdown 형식으로 출력해주세요.
    """
    rewrite_chain = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Python documentation assistant."),
        ("human", rewrite_prompt)
    ]) | llm | StrOutputParser()

    return rewrite_chain.invoke({
        "context": context,
        "question": question,
        "answer": answer
    })
    #return rewrite_chain.invoke({"context": context, "question": question})

# -----------------------------------------------------------
# 3. Streamlit 앱
# -----------------------------------------------------------
st.set_page_config(page_title="DS5 멀티 도메인 RAG 챗봇", layout="wide")
st.title("📚 Pandas & Scikit-learn RAG 챗봇")
st.caption("🔍 질문에 따라 Pandas, Scikit-learn, 또는 두 라이브러리의 통합 문서에서 검색합니다.")

# 사이드바: 도메인 선택 + 검색 문서 수
with st.sidebar:
    st.header("⚙️ 검색 옵션")
    use_pandas = st.checkbox("Pandas", value=True)
    use_sklearn = st.checkbox("Scikit-learn", value=True)
    k_value = st.slider("검색 문서 수 (k)", min_value=3, max_value=15, value=10)
    auto_rewrite_threshold = st.slider("자동 재작성 임계점 점수", min_value=0, max_value=100, value=80)


# 모델 및 벡터스토어 초기화
llm, embeddings = initialize_models()
retrievers = load_vectorstore_and_retrievers(embeddings, k_value=k_value)

# 세션 상태 초기화
if 'messages' not in st.session_state: st.session_state.messages = []

# 선택 상태에 따른 리트리버 매핑
retriever_map = {
    (True, False): 'Pandas',
    (False, True): 'Scikit-learn',
    (True, True): '둘 다',
    (False, False): '둘 다'
}
selected_domain = retriever_map[(use_pandas, use_sklearn)]
current_retriever = retrievers[selected_domain]
selected_domain_display = {
    'Pandas': 'Pandas',
    'Scikit-learn': 'Scikit-learn',
    '둘 다': 'Pandas + Scikit-learn (통합 검색)'
}[selected_domain]

# 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# 사용자 입력 처리
if prompt_input := st.chat_input("질문을 입력하세요..."):
    # 1) 사용자 메시지 세션 저장 및 바로 출력
    user_msg = {"role": "user", "content": prompt_input}
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt_input)

    # 2) 어시스턴트 답변 생성
    with st.chat_message("assistant"):
        domain_header = f"🔍 검색 도메인: **{selected_domain_display}**\n\n"
        st.markdown(domain_header, unsafe_allow_html=True)
        response_placeholder = st.empty()
        response_placeholder.info("⏳ 답변 생성 중입니다. 잠시만 기다려 주세요...")

        # 1단계: 검색
        try:
            retriever_chain = itemgetter("question") | current_retriever
            retrieved_docs = retriever_chain.invoke({"question": prompt_input})
            context_text = format_docs(retrieved_docs)
        except Exception as e:
            error_msg = f"❌ 검색 중 오류 발생: {e}"
            response_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": domain_header + error_msg})
            st.stop()
        
        # 2단계: 답변 생성
        llm_chain = get_llm_chain(llm)
        try:
            response_stream = llm_chain.stream({"context": context_text, "question": prompt_input})
            full_response = response_placeholder.write_stream(response_stream)

            # 3단계: Self-Evaluation
            eval_result = self_evaluate_response(llm, context_text, prompt_input, full_response)
            score = eval_result.get("score", "N/A")
            comment = eval_result.get("comment", "")
            st.info(f"✅ Self-Evaluation 점수: {score}\n💬 코멘트: {comment}")

            # 4단계: 자동 재작성
            if isinstance(score, (int, float)) and score < auto_rewrite_threshold:
                rewritten_response = auto_rewrite_response(llm, context_text, prompt_input, full_response)
                st.warning(f"🔄 점수 {score} 미만으로 자동 재작성 수행")
                final_response = rewritten_response  # 재작성된 답변만 최종 표시
            else:
                final_response = full_response  # 점수 이상이면 원래 답변 유지

            # 화면 표시 및 세션 저장
            response_placeholder.markdown(final_response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": domain_header + final_response})

        except AuthenticationError:
            msg = "❌ 답변 생성 중 OpenAI API 키 인증 실패."
            response_placeholder.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": domain_header + msg})
        except Exception as e:
            msg = f"❌ 답변 생성 오류: {type(e).__name__} - {e}"
            response_placeholder.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": domain_header + msg})

            
# ----------------------------------------------------