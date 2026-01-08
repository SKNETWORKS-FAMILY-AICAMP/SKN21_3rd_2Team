> SK네트웍스 Family AI 캠프 21기 3차 프로젝트  
> 개발기간: 26.01.07 ~ 26.01.08 <br>
> 주제 : RAG 기반 챗봇 개발

<br>

# 🖐️ 팀 소개

- ## 팀명 : **이명준(님) 짝꿍 구하기**
- ### 팀원 소개 :

<div align="center">
	
| [@김승룡] | [@정덕규] | [@이의정] | [@진승언] | [@이명준] |
| :----------------------------------------: | :----------------------------------------: | :----------------------------------------: | :----------------------------------------: | :----------------------------------------: |
| <img src="data/image/BSJ.png" width="120"> | <img src="data/image/PSH.png" width="120"> | <img src="data/image/LSW.png" width="120"> | <img src="data/image/KSU.png" width="120"> | <img src="data/image/PJH.png" width="120"> |

</div>

<br>

---

# 📖 프로젝트 개요

##  주제 : 선사시대부터 가장 중요했던 연애!, <br> 어떻게 하면 더 나은 연애를 할 수 있을지 상담해주는 AI 기반의 '연애 상담' 챗봇 서비스

- ###  **프로젝트 소개**

"**명준(님) 짝꿍구하기**"는 사용자의 연애에 대한 고민을 상담해주는 AI 대화형 챗봇 서비스입니다.  
**연애를 하면서 생기는 다양한 상황과 질문에 대한 답변를 RAG 기반으로 안내**하며  
**사용자가 선호하는 연애 유튜버의 말투로 답변**합니다.

- ###  **프로젝트 배경**

"연애를 하다보면, 또는 연애를 시작하려고 하면<br>
남들에게 물어보기도 어려운 다양한 상황에 직면하게 됩니다.<br>
이럴 때 저희 AI 챗봇에 물어보면 됩니다.."<br>

## 관련 기사

### ■ 모쏠 공대남, 5천명과 미팅 후 결혼까지…챗GPT가 다 했다


<img src="[data/image/news1.png](https://www.asiae.co.kr/news/img_view.htm?img=2024020108320550685_1706743925.jpg)">

- 러시아 개발자, AI로 '데이트 최적화' 프로포즈, 결혼 계획까지 AI에 맡겨

[https://www.asiae.co.kr/article/2024020110103092016](https://www.asiae.co.kr/article/2024020110103092016)
##### 출처: 아시아경제

<br>

#### => 연애는 하고 싶지만, 어떻게 해야할지 모르고, 어떤 얘기를 해야할지 모르겠는 사람이 많음.
	
### "이러한 배경에서 저희는 **연애 전문가(연애 유튜브)의 말투로 상황에 따른 대답을 해주는 AI 챗봇의 필요성**을 느끼고 도움을 줄 수 있는 챗봇을 구상하게 되었습니다."

- ### **프로젝트 목표**

1.  **핵심 정보 제공**: 연애하면서 맞닥뜨리게 되는 다양한 상황에 대한 대처법을 정확하게 답변해줌
    
2.  **맞춤형 대화**: 감정·기억을 반영한 개인화된 상담 제공
    
3.  **RAG 기반 정확성 향상**: GPT-4o-mini + Qdrant으로 근거 기반 답변 생성
    
4.  **명준(님), 의정(님) 짝꿍 구하기**: 명준(님)과 의정(님)의 여자친구, 남자친구를 구하는 것이 최종적인 프로젝트의 목적 입니다.

<br>

---

# 💻 기술 스택 & 사용한 모델  

| 분야                | 사용 도구 |
|---------------------|-----------|
| **Language**        | [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)](https://www.python.org/) |
| **Collaboration Tool** | [![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/) [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/) [![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/) |
| **LLM Model**       | [![GPT-4o](https://img.shields.io/badge/GPT--4o-mini%20-412991?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/) 
| **Embedding Model** | [![text-embedding-3-small](https://img.shields.io/badge/text--embedding--3--small-00A67D?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/embeddings) |
| **Vector DB**       | [![Pinecone](https://img.shields.io/badge/qdrant-0075A8?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/) |
| **Orchestration / RAG** | [![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/) |
| **Development Env** | [![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)](https://code.visualstudio.com/) [![Conda](https://img.shields.io/badge/Conda-3EB049?style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com/)

<br>

---

# 🪢시스템 아키텍처

### 프로젝트 구조

<img src="https://discord.com/channels/1455821809193713686/1455836066329264191/1458260754330292443">

<br>


### 시스템 아키텍처 구조도
![시스템 아키텍처]()




<br>

---

# 🗃️ 수집한 데이터 및 전처리 요약

### 수집한 데이터 목록
연애 유튜버들의 영상 URL 링크 수집    

### 전처리 요약

**데이터 수집 방법** : 각 연애 유튜버들의 채널을 방문하여 영상을 클릭한 후, 영상들의 URL 링크들을 불러옴.<br>
불러온 링크들을 이용하여 유튜브 영상들을 사운드 파일로 다운로드 한 후, json 포맷으로 용도에 맞게 변환

<br>

---

# 💾DB 연동

Vectorstore는 수업 시간에 배운 Qdrant를 이용

<br>

---

# 💡 테스트 계획 및 결과 보고서

### 5. 감성적인 대화
#### 질문 1 :

사실 요즘 몸이 너무 무겁고, 뭘 해보고 싶다는 생각도 잘 안 들어요…

- 예상 답변 :
```
많이 지치신 것 같아요. 몸과 마음이 무거울 때는 정말 아무 것도 하기 싫을 때가 있죠. 혹시 괜찮으시다면, 실내에서 조금씩 할 수 있는 작은 활동을 찾아보는 것도 도움이 될 수 있을 것 같아요. 지금처럼 이야기를 이어가도 괜찮습니다.
```
- 실제 답변 :
```
마음이 많이 지치신 것 같아요. 그런 날들이 있죠. 혹시 누워 계시면서도 편안함을 느꼈던 순간이 있으셨나요? 함께 작은 기쁨을 찾아보면 어떨까요?
```
- 답변 소요 시간 : 1초

#### 질문 2 :
주말에 할 수 있는 일을 하나 추천해주세요.
- 예상 답변 :
```
혹시 괜찮으시다면, 주말에 가족이나 친구와 함께 시간을 보내보는 건 어떨까요? 가까운 사람들과의 시간을 통해 조금이나마 마음의 평온을 찾으실 수 있을지도 몰라요. 부담 없으시다면, 손주나 친구와의 시간을 한번 계획해 보세요. 물론, 그냥 편안히 이야기를 나누는 것도 좋습니다.
```
- 실제 답변 :
```
혹시 괜찮으시다면, 주말에는 편안한 자세로 누워서 잡지를 읽어보시는 건 어떠세요? 마음이 조금은 가벼워지실 수도 있을 거예요. 부담 없이 편하게 쉬시는 것도 좋습니다.
```
- 답변 소요 시간 : 4초



---
# 🐛 트러블슈팅

### 검색 결과 부정확

-  주소별 벡터값의 차이가 작아 주소를 잘 못 잡아오는 문제

   → 필터 적용(지역) : get_close_matches 사용 단어 유사도 검사 후, 지역 리스트 속에서 찾아서 그것으로 필터 적용

-  지역 필터와 시설구분(시설별 지역목록 차이 존재)을 함께 쓸 경우 굳이 벡터 DB를 사용할 이유가 없어진다.

   → 충분히 많은 양의 지역을 get_close_matches 함수를 통해 가져오고 그것을 한 번의 필터(지역)만을 사용해 검색하는 방향으로 진행. 시설은 벡터 유사도로 알아서 분리.

- 특정(종교 기반) 장례 시설 검색 문제(유사도 검색) : 종교 관련 장례 시설 등의 특수한 시설을 검색하는 데 있어 적절한 검색 성능이 나오지 않는 문제

   → page_content 수정 : 장례 시설의 이름에서 특수 키워드(예 : 천주교)가 있을 경우 그러한 시설임을 내용에 추가하는 방식으로 해결. 이때 추가한 특수 상황은 종교와 공/사설 구분이다.

### 구분 없는 표 형식 텍스트 추출 문제(텍스트 추출)  

- pdf 리더가 위에서 아래로 한 줄씩 읽는 방식이라 글자가 섞여서 출력

   → OCR 사용 : PDF reader 라이브러리와 크게 다를 게 없었음. 클로바 general ocr 사용
   → 직접 손으로 드래그 + ctrl C + ctrl V : PDF어플리케이션의 경우 적절히 텍스트를 묶어주어 이것으로 해결.

### RAG 응답 정확도 개선(프롬프트 기반)	

- 질의에 대응하는 청크를 정확히 참조하지 못해 응답 품질이 떨어지는 문제가 발견

   → 프롬프트 템플릿을 재설계하고 시스템 메시지·지시문을 세분화한 뒤, 여러 유형의 질문으로 반복 검증하여 모델이 의도한 데이터 기반 응답을 안정적으로 생성하도록 해결.

### 데이터 청크 문제

- 전처리한 몇몇 데이터 중 정확한 청크가 Top-K 안에 안 뜨는 문제가 계속됨.

   → 청크 분할이 잘못되어 단계 정보(step_01, step_02)가 서로 다른 파일에 흩어져 있었음. 원문을 step 단위로 다시 청킹하고 관련 문장끼리 하나의 chunk_id로 묶음.

<br>

---

# 🤖 수행결과(시연 페이지)

<img src="data/image/preview_image1.png" width="300"> <img src="data/image/preview_image2.png" width="300">

<br>

---

# ✒️ 한 줄 회고

| 이름 | 회고 |
|----------|-------------|
| 김승룡 |  |
| 정덕규 |  |
| 이의정 |  |
| 진승언 |  |
| 이명준 |  |




