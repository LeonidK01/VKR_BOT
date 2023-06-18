import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
import re

def preprocess_text(text):
    text = text.replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('[^a-zA-Zа-яА-Я]+', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.lower()
    text = "".join(text)
    return text.strip()

def lem(text):


# Скачивание необходимых ресурсов NLTK
#     nltk.download('punkt')
#     nltk.download('stopwords')

# Инициализация анализатора и загрузка стоп-слов
    mystem = Mystem()
    russian_stopwords = stopwords.words("russian")

    tokens = word_tokenize(text, language="russian")

    # Удаление стоп-слов
    tokens = [token for token in tokens if token not in russian_stopwords]

    # Лемматизация
    lemmas = [mystem.lemmatize(token)[0] for token in tokens]
    return [preprocess_text(' '.join(lemmas))]

#print("Вчера вечером я гулял около опушки и увидел там спящего енота www.len.ru, и завтра вечером я тоже туда пойду\n",lem("Вчера вечером я гулял около опушки и увидел там спящего енота , и завтра вечером я тоже туда пойду"))
# Вывод лемм
