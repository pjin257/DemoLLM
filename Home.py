import streamlit as st

st.set_page_config(
    page_title="LLM Demo",
    page_icon="🤖",
)

st.title("LLM Demo Home")

st.markdown("""
환영합니다!

LLM과 RAG을 사용해서 문서를 검색하는 챗봇 데모입니다. 
            
이 데모는 짧은 코드로 「국방정보화업무 훈령」에 대한 검색 기능을 제공합니다. 
            
구동을 위한 LLM은 아래 2가지 중 선택할 수 있습니다.      
            
""")

st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 10px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
        padding-left: 5px;
        padding-right: 5px;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #FFFFFF;
	}

</style>""", unsafe_allow_html=True)

tab_one, tab_two= st.tabs(["GPT-4o", "Private sLLM"])

with tab_one:
    st.markdown("""
    1. OPEN AI의 추론용 API 사용

    2. 장점 : 인프라 없이 API로 손쉽게 LLM 사용 가능. 오픈소스 아닌 LLM 모델 사용 가능
                                
    3. 단점 : 내부망 사용 불가. 외부 API를 사용하므로 데이터 유통에 따른 보안문제, API 비용 등 제한사항 사전 고려 필요
    """)

with tab_two:
    st.markdown("""
    1. 서버에 저장된 LLM 사용

    2. 장점 : 내부망에서 사용 가능. 데이터 외부 유통 없음. 추론 수행에 대한 별도 사용료 없음.
                                
    3. 단점 : 인프라 구축 필요. 오픈소스 LLM만 사용 가능.
    """)

st.markdown("""
            -------------

좌측 상단의 사이드탭을 열어 데모 기능을 확인하세요!
            
""")