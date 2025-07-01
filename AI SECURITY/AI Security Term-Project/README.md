# AI Security Term - Project 
NB15를 이용해 유해 트래픽 분류 
-> 정상 및 공격 분류 

### 목표 데이터 셋
  
  네트워크 트래픽 데이터
- 네트워크 환경에서 정상적인 활동과 다양한 공격 행위를 포함한 혼합 트래픽 데이터 수집를 수집하기 위해 설계된 데이터셋

- 49개의 Feature(Feature.csv 파일참고)
- 네트워크 트래픽의 다양한 특성을 반영
- 트래픽의 정상 및 공격 유형을 분류하는 데 사용
- 총 9가지 유형의 공격 포함 - Fuzzers, Analysis, Backdoors, Dos, Exploits, Generic, Reconnaissance, Shellcode, Worms

- 학습 데이터 (배포) -> 약 17만 개의 데이터로 구성
- 테스트 데이터 (평가) -> 약 8만개의 데이터로 구성

-----------------------------------------------------------
 
#### 참고 사이트 
https://research.unsw.edu.au/projects/unsw-nb15-dataset <UNSW 대학 NB15 데이터셋>
