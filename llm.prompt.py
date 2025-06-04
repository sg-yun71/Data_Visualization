import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 저장할 폴더 설정
output_dir = "llm_prompts"
os.makedirs(output_dir, exist_ok=True)

# OCR 결과 불러오기
with open("data/ocr_result.txt", "r") as f:
    ocr_text = f.read().strip()

# 단일 프롬프트 설정
prompt_name = "전문가수준요청"
prompt_text = f" 'Attention is all you need'라는 문장은 인공지능 역사에서 전환점으로 평가받는다. 이 문장의 의미, 해당 논문이 기존 RNN 기반 모델과 어떤 차별점을 가졌는지, 그리고 이후 BERT와 GPT 같은 모델에 어떤 영향을 미쳤는지 단계별로 설명하고, 마지막엔 본인의 의견도 간단히 덧붙여줘: \"{ocr_text}\""

# LLM 응답 받기
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt_text}],
    temperature=0.7,
)
llm_output = response.choices[0].message.content.strip()

# 결과 저장
result = {
    "prompt_name": prompt_name,
    "prompt": prompt_text,
    "response": llm_output
}
output_path = os.path.join(output_dir, f"{prompt_name}.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"'{prompt_name}' 결과가 {output_path} 에 저장되었습니다.")