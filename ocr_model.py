from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# 이미지 로드
image = Image.open("data/sample.png")

# TrOCR 모델 로딩
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# OCR 추론
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# OCR 결과 저장
with open("data/ocr_result.txt", "w") as f:
    f.write(generated_text)

print("OCR 결과:", generated_text)