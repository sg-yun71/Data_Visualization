import os
import json
import re
import pandas as pd

# 평가할 핵심 키워드
keywords = [
    "attention", "transformer", "self-attention", "encoder",
    "decoder", "context", "sequence", "neural"
]

# 결과 저장 리스트
results = []

# 응답이 들어있는 폴더명
response_dir = "llm_prompts"  # 너의 프롬프트 응답 JSON이 들어있는 폴더

# 폴더 내 JSON 파일 순회
for filename in os.listdir(response_dir):
    if filename.endswith(".json"):
        with open(os.path.join(response_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        response = data["response"]
        prompt_name = filename.replace(".json", "")

        # 평가지표 계산
        length_score = len(response)  # 글자 수
        word_count = len(response.split())  # 단어 수
        sentence_count = len(re.findall(r'[.!?]', response))  # 문장 수 추정
        keyword_score = sum(1 for k in keywords if k.lower() in response.lower())  # 키워드 등장 수

        results.append({
            "prompt_name": prompt_name,
            "length_score": length_score,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "keyword_score": keyword_score
        })

# 결과를 DataFrame으로 정리
df = pd.DataFrame(results)

# CSV 파일로 저장
df.to_csv("performance_metrics.csv", index=False, encoding="utf-8-sig")
print("✅ 평가지표 계산 완료 → performance_metrics.csv 생성됨")