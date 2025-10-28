# 🚀 Pandas/Scikit-Learn 공식 문서 기반 챗봇 어시스턴트

**프로젝트 팀:** 너도해 (진용현, 이환철, 이종석, 김건영)

---

## 💡 프로젝트 개요 (Introduction)

이 프로젝트는 데이터 사이언스의 대표적인 라이브러리인 **Pandas**와 **Scikit-Learn**의 공식 문서를 기반으로, **정확하고 신뢰할 수 있는 답변**을 제공하는 **RAG(검색 증강 생성) 챗봇 어시스턴트**를 개발하는 것을 목표로 설정하였습니다.

### 해결하고자 하는 문제점

1.  **공식 문서의 방대함:** 공식 문서의 양이 너무 방대하여 사용자가 필요한 정보를 **탐색하기 어렵습니다**.
2.  **LLM 답변의 부정확성:** LLM에게 물어봐도 답변이 시원찮거나 부정확한 답변을 하는 경우가 있습니다.

### 프로젝트의 가치

* **정확한 안내:** Pandas, Scikit-Learn 코드의 **의미와 사용법에 대한 정확한 안내**를 제공합니다.
* **정보 확산 근절 기대:** Pandas, Scikit-Learn 외에도 확대하여 부정확한 정보 확산 근절을 기대합니다.

---

## ⚙️ 기술 설계 및 사용 스택 (Technical Design & Stack)

본 프로젝트는 RAG(Retrieval-Augmented Generation) 아키텍처를 기반으로 설계되었습니다.

### 1. 데이터 수집 및 모델

| 구분 | 내용 |
| :--- | :--- |
| **사용 문서** | Pandas, Scikit-Learn 공식 문서|
| **수집 방식** | 라이브러리 Git Clone 후, `rst`, `md`, `py` 파일 등을 추출하여 사용 |
| **임베딩 모델** | `text-embedding-3-small` (OpenAI), `intfloat/multilingual-e5-large-instruct` (HuggingFace) 등을 고려|
| **LLM 모델** | GPT-4o-mini, GPT-4o, GPT-5-nano, GPT-5-mini 등의 모델을 고려|

### 2. 추가 내용
자세한 내용은 GitHub 저장소 내 발표자료를 참조해주세요.
