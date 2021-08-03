## word2vec과 감정 분석으로 알아본 올림픽 메달에 대한 인식 변화
[쏘프라이즈 메달에 대한 인식 변화 분석 프로젝트](https://soprize.so/answer/466)로 진행했습니다.
* 크롤링 코드 참고: https://github.com/lumyjuwon/KoreaNewsCrawler
* 2004년-2021년까지의 스포츠 기사 중 올림픽 관련 기사를 분석함 
* 크롤링한 데이터, 전처리 데이터, 학습한 word2vec 모델은 파일 크기 때문에 [드라이브](https://drive.google.com/drive/folders/1XGSXyIV1IIkbOfPDggkDxCUWqwNYNtMm?usp=sharing)에 업로드함

### 1. 연도별 '메달' 관련 단어 변화
* 연도별 올림픽 기사로 word2vec 모델을 학습한 후, 메달별 유사 단어를 비교함 
    
### 2. 연도별 '메달'이 포함된 문장의 감정 변화
* 네이버 영화평 데이터 NSMC로 학습한 [KoELECTRA](https://huggingface.co/monologg/koelectra-base-finetuned-nsmc)로 감정분석을 진행함  
* 메달별 평균 감정 점수, 메달별 감정 점수 분포를 살펴봄 

