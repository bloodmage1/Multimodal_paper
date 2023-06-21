# Multimodal AI

1. 트렌드
2. 문제점
3. 사용한 방법
   - MCB

복잡한 과제 및 높은 정확도를 달성하기 위해 다양한 데이터 종류(텍스트, 음성, 이미지, 수치형 데이터)와 스마트 처리 알고리즘을 결합한 것.

하지만 <span style="color:skyblue">**다양한 형태(modality)의 데이터를 사용한다고 해서 Multimodal AI이 아니라 변수들이 차원이 달라야 한다**</span>.

## 트렌드

- 현재 가장 직관적이고 의미있는 결과를 보이는 분야는 시각과 언어지능을 결합한 언어 시각 트랜스포머 연구가 대표적
- ImageGPT: Open AI의 이미지 사전학습 모델
- CLIP: 이미지와 언어를 결합
- DALL-E: 텍스트로부터 이미지를 생성

## 문제점

인간 행동 인식이나 감정 인식 문제에서는 단순히 이미지를 잘 분류한다고 해서 성능이 확보되지 않는다. 예를 들어 인간들의 표정 사진에서 인간조차도 기쁜지 슬픈지는 입을 크게 벌리고 웃거나 눈물을 보이지 않는 이상 알 수 없다.

즉 인간이 기쁜지 슬픈지를 알 수 있는 사람을 표현하는 데이터를 확보하고 사용할 필요가 있다.

## 사용한 방법

해당 코드에서는 Multimodal Compact Bilinear Pooling (줄여서 MCB)를 사용했다. 이 방법은 두 벡터의 외적을 계산하여 두 모달리티 간의 모든 가능한 상호작용을 포착한다. 하지만 외적은 계산적으로 **매우 비효율적**이므로,MCB는 이를 해결하기 위해 두 벡터의 외적을 근사하는 방법을 사용한다.

이 방법은 두 벡터를 먼저 각각의 고차원 공간으로 매핑한 후, 이 **두 고차원 벡터의 내적을 계산**하여 원래의 외적을 근사한다.

츨처: [https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR](https://dacon.io/forum/405915?dtype=recent)

참고할만한 유튜브: [https://www.youtube.com/watch?v=PcU2oiPFDTE&t=1037s](https://www.youtube.com/watch?v=PcU2oiPFDTE&t=1037s)