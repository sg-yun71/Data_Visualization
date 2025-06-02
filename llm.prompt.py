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
prompt_name = "간단한 번역요청"
prompt_text = f"다음 문장을 이해할 수 있게 번역해줘: \"{ocr_text}\""

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