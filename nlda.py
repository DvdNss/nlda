from language_detection.language_detector import load_model, detect_language

ld_path = 'language_detection/model/lid.176.ftz'

# Load language detector
language_detector = load_model(ld_path)

# Detect language
result = detect_language(['this is a test', 'mi chiamo DAvide sono francese'], language_detector=language_detector)
print(result['stats'])

# TODO: translation and paraphrasing
