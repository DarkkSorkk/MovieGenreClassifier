import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Carregando datasets...")
# Carregue seus datasets
df_treino = pd.read_csv('df_treino.csv')
df_test = pd.read_csv('df_test.csv')

print("Preparando os dados...")
# Combine movie_name e synopsis em uma única feature para treino e teste
df_treino['Text'] = df_treino['movie_name'] + ' ' + df_treino['synopsis']
df_test['Text'] = df_test['movie_name'] + ' ' + df_test['synopsis']

print("Realizando a vetorização TF-IDF...")
# Extração de características com TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(df_treino['Text'])
y_train = df_treino['genre']
X_test = vectorizer.transform(df_test['Text'])

print("Treinando o modelo de Random Forest...")
# Treinar o modelo de Random Forest com verbose=2
clf = RandomForestClassifier(verbose=2)
clf.fit(X_train, y_train)

print("Fazendo previsões e calculando probabilidades...")
# Fazer previsões e calcular probabilidades
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

print("Adicionando previsões e probabilidades ao conjunto de testes...")
# Adicionar previsões e probabilidades ao df_test
df_test['predicted_genre'] = y_pred
df_test['prediction_probability'] = [max(proba) for proba in y_proba]

print("Salvando o conjunto de testes atualizado...")
# Salvar o df_test atualizado em um novo arquivo CSV
df_test.to_csv('df_test_with_predictions.csv', index=False)

print("Processo concluído.")

# Salvar o modelo
joblib.dump(clf, 'movie_genre_classifier.pkl')
# Salvar o vetorizador
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Modelo e vetorizador salvos com sucesso.")
