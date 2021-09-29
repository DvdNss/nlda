import logging
from typing import List

import pandas
from tqdm import tqdm

from helsinki_translation.helsinki_translation import load_model as load_translator, translate
from language_detection.language_detector import load_model as load_detector, detect_language_from_df


def translate_dataset(path: str, target_languages: List[str], translate_targets: bool = False,
                      aug_data_path: str = 'augmented_data.tsv', source_col: str = 'source_text',
                      target_col: str = 'target_text'):
    """
    Translate 2 columns datasets to given languages.

    :param path: path to dataset
    :param target_languages: list of languages codes (ex: 'fr' for french)
    :param translate_targets: whether to translate targets or not
    :param aug_data_path: path of augmented file
    :param source_col: source column
    :param target_col: target column
    :return:
    """

    # Init logging
    logging.basicConfig(level=logging.INFO)
    logging.info('Currently running NLDA. \n')

    # Load language detector
    language_detector = load_detector()

    # Load DataFrame
    dataset = pandas.read_csv(path, sep='\t', usecols=[source_col, target_col]).astype(str)

    # Detect language
    dataset, stats = detect_language_from_df(dataset, language_detector=language_detector)
    logging.info(f" Dataset stats are the following:\n{stats}\n")

    # Add id column to track translations
    dataset['id'] = ''
    for index, row in dataset.iterrows():
        row['id'] = index

    # Declare variables
    data_to_add = []

    # Start data augmentation
    logging.info(f' Translating your dataset to {target_languages}. Please wait...')
    logging.basicConfig(level=logging.NOTSET)

    for source_language in tqdm(stats.keys()):
        for target_language in target_languages:
            if source_language != target_language:

                # try/except in case language pair is unavailable
                try:
                    # Load source-target pair model
                    tokenizer, translator = load_translator(source=source_language, target=target_language)

                    # Augment data
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
    print('')

    # Convert augmented data list to dataframe and append it to the not augmented one
    new_dataset = pandas.DataFrame(data_to_add, columns=['source_text', 'target_text', 'language', 'id'])
    augmented_dataset = dataset.append(new_dataset, ignore_index=True).sort_values('id').reset_index(drop=True)

    # Write augmented data to file
    augmented_dataset.to_csv(aug_data_path, sep='\t')
    logging.info(f' Augmented dataset has been saved to {aug_data_path}. ')
