import logging

import pandas

from helsinki_translation.helsinki_translation import load_model as load_translator, translate
from helsinki_translation.languages import Language
from language_detection.language_detector import load_model as load_detector, detect_language_from_df

logging.basicConfig(level=logging.INFO)

# Load language detector
language_detector = load_detector()

# Load DataFrame
dataset = pandas.read_csv('df.tsv', sep='\t').astype(str)

# Detect language
dataset, stats = detect_language_from_df(dataset, language_detector=language_detector)
logging.info(f" Dataset stats are the following:\n{stats}")

target_languages = ['en', 'fr', 'it']

# Add id column to track translations
dataset['id'] = ''
data_to_add = []

for index, row in dataset.iterrows():
    row['id'] = index
    data_to_add.append([])

# Load translator
tokenizer, translator = load_translator(source=Language.English, target=Language.French)

# Translate some text
translation = translate(source='Hi, my name is David.', tokenizer=tokenizer, model=translator)
print(translation)
