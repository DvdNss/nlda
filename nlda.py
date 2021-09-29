import logging

import pandas
from tqdm import tqdm

from helsinki_translation.helsinki_translation import load_model as load_translator, translate
from helsinki_translation.languages import Language
from language_detection.language_detector import load_model as load_detector, detect_language_from_df

logging.basicConfig(level=logging.INFO)
logging.info('Currently running NLDA. \n')

# Load language detector
language_detector = load_detector()

# Load DataFrame
dataset = pandas.read_csv('df.tsv', sep='\t').astype(str)

# Detect language
dataset, stats = detect_language_from_df(dataset, language_detector=language_detector)
logging.info(f" Dataset stats are the following:\n{stats}\n")

target_languages = ['en', 'fr', 'it']

# Add id column to track translations
dataset['id'] = ''
for index, row in dataset.iterrows():
    row['id'] = index

data_to_add = []
id_count = 0
translate_targets = False

logging.info(f' Translating your dataset to {target_languages}. Please wait...')
for source_language in tqdm(stats.keys()):
    for target_language in target_languages:
        if source_language != target_language:

            # Load source-target pair model
            try:
                tokenizer, translator = load_translator(source=source_language, target=target_language)

                for index, row in dataset.iterrows():
                    if row['language'] == source_language:
                        data_to_add.append([
                            translate(source=row['source_text'], tokenizer=tokenizer, model=translator),
                            translate(source=row['target_text'], tokenizer=tokenizer,
                                      model=translator) if translate_targets is True else row['target_text'],
                            target_language,
                            row['id']
                        ])
            except:
                pass

print(data_to_add)
