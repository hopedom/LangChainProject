import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


os.environ["OPENAI_API_KEY"] = ""


@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-5-mini")


@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct')
    persist_directory = 'chroma_db_pd_sklearn'
    return Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)


@st.cache_resource
def get_rag_chain():
    vectorstore = get_vector_store()
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    
    RAG_PROMPT_TEMPLATE = """
    당신은 Pandas 2.3, Scikit-Learn 1.7.2의 권위자입니다. 오로지 Pandas, Scikit-Learn에 대해서만 답변하세요.
    Pandas, Scikit-Learn과 관련이 없는 질문에 대해서는 "해당 내용은 공식 문서에 없습니다."라고 답변하세요.
    반드시 아래에 제공되는 [검색된 Pandas, Scikit-Learn 문서 내용]을 근거로 하여 사용자의 질문에 답변해야 합니다.

    [검색된 Pandas, Scikit-Learn 문서 내용]:
    {context}

    [사용자 질문]:
    {question}

    [답변]:
    """
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    def format_docs(docs):
        formatted_strings = []
        for doc in docs:
            source = doc.metadata.get('source', 'N/A').split('/')[-1]
            content_preview = doc.page_content[:300] + "..."
            formatted_strings.append(f"--- [출처: {source}] ---\n{content_preview}")
            
        if not formatted_strings:
            return "검색된 근거 문서가 없습니다."
        
        return "\n\n".join(formatted_strings)
    
    answer_chain = (
        rag_prompt
        | llm
        | StrOutputParser()
    )
    
    context_chain = retriever | format_docs
    
    rag_chain_with_source = RunnableParallel(
        context=context_chain,
        question=RunnablePassthrough(),
        
        answer=RunnableParallel(
            context=context_chain,
            question=RunnablePassthrough(),
        ) | answer_chain
    )

    return rag_chain_with_source



st.set_page_config(page_title="Pandas & Scikit-Learn RAG 챗봇")
st.title("Pandas & Scikit-Learn 공식 문서 어시스턴트")
st.caption("공식 문서를 기반으로 질문에 답변합니다.")

try:
    rag_chain = get_rag_chain()
except Exception as e:
    st.error(f"RAG Chain 로드 중 오류 발생: {e}")
    st.stop()

# 1. 채팅 기록 초기화 (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! Pandas와 Scikit-Learn에 대해 무엇이든 물어보세요."}
    ]


# 2. 이전 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "context" in message:
            with st.expander("답변 근거 보기 (출처 문서)"):
                st.markdown(message["context"])


# 3. 사용자 입력 받기 (chat_input)
if prompt := st.chat_input("Pandas와 Scikit-Learn에 대해 질문하세요..."):
    # 3-1. 사용자 메시지를 채팅 기록에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3-2. RAG Chain을 실행하여 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("공식 문서를 검색하고 답변을 생성 중입니다...") as spinner:
            try:
                response_dict = rag_chain.invoke(prompt)
                
                final_answer = response_dict.get("answer", "오류: 답변을 생성하지 못했습니다.")
                retrieved_context = response_dict.get("context", "오류: 근거 문서를 찾지 못했습니다.")            
                
                st.markdown(final_answer)
                
                with st.expander("답변 근거 보기 (출처 문서)"):
                    st.markdown(retrieved_context)
                
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer, "context": retrieved_context}
                    )
            
            except Exception as e:
                st.error(f"답변 생성 중 오류 발생: {e}")