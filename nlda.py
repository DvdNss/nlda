from language_detection.language_detector import load_model, detect_language

# Load language detector
language_detector = load_model('language_detection/model/lid.176.ftz')

# Detect language
result = detect_language(['this is a test'], language_detector=language_detector)
print(result['stats'])