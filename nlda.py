from language_detection.language_detector import load_model as load_detector, detect_language
from helsinki_translation.helsinki_translation import load_model as load_translator, translate
from helsinki_translation.languages import Language

ld_path = 'language_detection/model/lid.176.ftz'

# Load language detector
language_detector = load_detector(ld_path)

# Detect language
result = detect_language(['this is a test', 'mi chiamo davide sono francese'], language_detector=language_detector)
print(result['stats'])

# Load translator
tokenizer, translator = load_translator(source=Language.English, target=Language.French)

# Translate some text
translation = translate(source='Hi, my name is David.', tokenizer=tokenizer, model=translator)
print(translation)

# TODO: paraphrasing
