from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

def valid_mean_per_100(valid):
    val_list = []
    for i in range(0, len(valid), 100):
        val_list.append(np.mean(valid[i:i+100]))

    return val_list   

def remove_stop(text, korean_stopwords):
    txt = re.sub('[^가-힣a-z]', ' ', text) #한글과 영어 소문자만 남기고 다른 글자 모두 제거
    token = tokenizer.morphs(txt) #형태소 분석
    token = [t for t in token if t not in korean_stopwords or type(t) != float] #형태소 분석 결과 중 stopwords에 해당하지 않는 것만 추출
    return token