import pandas as pd
import nltk
# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

from sklearn.model_selection import train_test_split
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from nltk.corpus import wordnet

# Inicializar recursos de NLTK
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Leer el archivo CSV
df = pd.read_csv('Datos/texto.csv', delimiter=';')

# Dividir los datos por cada valor único en 'label_text'
resultados_train = []
resultados_test = []

for label in df['label_text'].unique():
    # Filtrar las filas por el valor de 'label_text'
    df_filtrado = df[df['label_text'] == label]
    
    # Limitar a 5,000 filas 
    df_filtrado = df_filtrado.head(5000)
    
    # Dividir en train (2/3) y test (1/3)
    train, test = train_test_split(df_filtrado, test_size=0.33, random_state=42)
    resultados_train.append(train)
    resultados_test.append(test)

# Combinar los conjuntos
df_train = pd.concat(resultados_train)
df_test = pd.concat(resultados_test)

# Función para obtener la etiqueta POS adecuada para WordNetLemmatizer
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Función para preprocesar el texto
def preprocess_text(text):
    tokens = wordpunct_tokenize(text.lower())  # Convertir a minúsculas y tokenizar
    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(word))
        for word in tokens
        if word not in stop_words and word.isalnum()  # Eliminar stopwords y caracteres no alfanuméricos
    ]
    return " ".join(tokens)

# Preprocesar solo el conjunto de entrenamiento
print("Inicio Preprocesado del Train")
df_train['text'] = df_train['text'].apply(preprocess_text)
print("Fin Preprocesado del Train")

# OPCIONAL: Preprocesar el conjunto de test, no lo vamos a hacer ya que justo el modelo que haremos tiene que clasificar
procesar_test = True  # Cambiar a True si quieres preprocesar el test
if procesar_test:
    print("Inicio Preprocesado del Test")
    df_test['text'] = df_test['text'].apply(preprocess_text)
    print("Fin Preprocesado del Test")

# Eliminar la columna 'label_text' antes de guardar
df_train = df_train.drop(columns=['label_text'])
df_test = df_test.drop(columns=['label_text'])

# Guardar los resultados en archivos CSV
df_train.to_csv('Datos/texto_train-True.csv', index=False)
df_test.to_csv('Datos/texto_test-True.csv', index=False)

print("Proceso completado y datos guardados.")
