import pandas as pd
import streamlit as st

from language_detection.language_detector import load_model as load_detector, detect_language_from_df
from nlda import translate_dataframe


@st.cache
def df_to_file(df: pd.DataFrame):
    return df.to_csv(sep='\t').encode('utf-8')


@st.cache
def load_ld():
    return load_detector(path='nlda/language_detection/model/lid.176.ftz')


# Page header
st.title('NLDA - Natural Language Data Augmentation')
st.write('Natural Language Augmentation made easy.')

# Load language detector
language_detector = load_ld()

# Dataset uploader
uploaded_file = st.file_uploader('Upload your dataset. ', type=['.csv', '.tsv'])
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file, sep='\t').astype(str)
    file_details = {"FileName": uploaded_file.name, "FileSize": len(dataset), "Columns": [c for c in dataset.columns]}

    # Write dataset informations
    st.write(file_details)

    # Display dataset
    st.write(dataset)

    # Apply language detection
    dataset, stats = detect_language_from_df(dataset, language_detector=language_detector)
    st.write('Here are the languages statistics about your dataset: ')
    st.write(stats)

    # Choose target languages
    languages = {
        'Oromo': 'om',
        'Abkhazian': 'ab',
        'Afar': 'aa',
        'Afrikaans': 'af',
        'Albanian': 'sq',
        'Amharic': 'am',
        'Arabic': 'ar',
        'Armenian': 'hy',
        'Assamese': 'as',
        'Aymara': 'ay',
        'Azerbaijani': 'az',
        'Bashkir': 'ba',
        'Basque': 'eu',
        'Bengali': 'bn',
        'Bhutani': 'dz',
        'Bihari': 'bh',
        'Bislama': 'bi',
        'Breton': 'br',
        'Bulgarian': 'bg',
        'Burmese': 'my',
        'Byelorussian': 'be',
        'Cambodian': 'km',
        'Catalan': 'ca',
        'Chinese': 'zh',
        'Corsican': 'co',
        'Croatian': 'hr',
        'Czech': 'cs',
        'Danish': 'da',
        'Dutch': 'nl',
        'English': 'en',
        'Esperanto': 'eo',
        'Estonian': 'et',
        'Faeroese': 'fo',
        'Fiji': 'fj',
        'Finnish': 'fi',
        'French': 'fr',
        'Frisian': 'fy',
        'Galician': 'gl',
        'Georgian': 'ka',
        'German': 'de',
        'Greek': 'el',
        'Greenlandic': 'kl',
        'Guarani': 'gn',
        'Gujarati': 'gu',
        'Hausa': 'ha',
        'Hebrew': 'he',
        'Hindi': 'hi',
        'Hungarian': 'hu',
        'Icelandic': 'is',
        'Indonesian': 'id',
        'Interlingua': 'ia',
        'Interlingue': 'ie',
        'Inupiak': 'ik',
        'Inuktitut, Eskimo': 'iu',
        'Irish': 'ga',
        'Italian': 'it',
        'Japanese': 'ja',
        'Javanese': 'jw',
        'Kannada': 'kn',
        'Kashmiri': 'ks',
        'Kazakh': 'kk',
        'Kinyarwanda': 'rw',
        'Kirghiz': 'ky',
        'Kirundi': 'rn',
        'Korean': 'ko',
        'Kurdish': 'ku',
        'Laothian': 'lo',
        'Latin': 'la',
        'Latvian, Lettish': 'lv',
        'Lingala': 'ln',
        'Lithuanian': 'lt',
        'Macedonian': 'mk',
        'Malagasy': 'mg',
        'Malay': 'ms',
        'Malayalam': 'ml',
        'Maltese': 'mt',
        'Maori': 'mi',
        'Marathi': 'mr',
        'Moldavian': 'mo',
        'Mongolian': 'mn',
        'Nauru': 'na',
        'Nepali': 'ne',
        'Norwegian': 'no',
        'Occitan': 'oc',
        'Oriya': 'or',
        'Pashto, Pushto': 'ps',
        'Persian': 'fa',
        'Polish': 'pl',
        'Portuguese': 'pt',
        'Punjabi': 'pa',
        'Quechua': 'qu',
        'Rhaeto, Romance': 'rm',
        'Romanian': 'ro',
        'Russian': 'ru',
        'Samoan': 'sm',
        'Sangro': 'sg',
        'Sanskrit': 'sa',
        'ScotsGaelic': 'gd',
        'Serbian': 'sr',
        'SerboCroatian': 'sh',
        'Sesotho': 'st',
        'Setswana': 'tn',
        'Shona': 'sn',
        'Sindhi': 'sd',
        'Singhalese': 'si',
        'Siswati': 'ss',
        'Slovak': 'sk',
        'Slovenian': 'sl',
        'Somali': 'so',
        'Spanish': 'es',
        'Sudanese': 'su',
        'Swahili': 'sw',
        'Swedish': 'sv',
        'Tagalog': 'tl',
        'Tajik': 'tg',
        'Tamil': 'ta',
        'Tatar': 'tt',
        'Tegulu': 'te',
        'Thai': 'th',
        'Tibetan': 'bo',
        'Tigrinya': 'ti',
        'Tonga': 'to',
        'Tsonga': 'ts',
        'Turkish': 'tr',
        'Turkmen': 'tk',
        'Twi': 'tw',
        'Uigur': 'ug',
        'Ukrainian': 'uk',
        'Urdu': 'ur',
        'Uzbek': 'uz',
        'Vietnamese': 'vu',
        'Volapuk': 'vo',
        'Welch': 'cy',
        'Wolof': 'wo',
        'Xhosa': 'xh',
        'Yiddish': 'yi',
        'Yoruba': 'yo',
        'Zhuang': 'za',
        'Zulu': 'zu'
    }
    selected_languages = st.multiselect("Select your target languages: ", options=languages)
    st.write(f'You selected {selected_languages} languages. ')

    # Augment dataset
    doAugment = st.button('Augment dataset')
    if doAugment:
        for index, item in enumerate(selected_languages):
            selected_languages[index] = languages[item]

        # Build augmented dataset
        with st.spinner('Augmenting your dataset... This might take a while...'):
            augmented_dataset = translate_dataframe(dataset, stats, selected_languages)
            file = df_to_file(augmented_dataset)
        st.success('Augmented dataset is ready! Click the download button below. ')

        # Display augmented dataset and download button
        st.write(augmented_dataset)
        st.download_button('Download augmented dataset', data=file,
                           file_name='augmented_dataset.tsv')
