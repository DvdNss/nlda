from nlda import translate_dataset

# Dataset informations
path_to_dataset = 'example.tsv'
target_languages = ['en', 'fr']

# Augment dataset using translation with default arguments
translate_dataset(path_to_dataset, target_languages=target_languages)
