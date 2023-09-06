import joblib

# Carregar o modelo e o vetorizador treinados
clf = joblib.load('movie_genre_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Função para fazer previsões
def predict_genre(title, synopsis):
    # Combinar o título e a sinopse
    combined_text = title + ' ' + synopsis
    
    # Transformar o texto usando o vetorizador TF-IDF
    X = vectorizer.transform([combined_text])
    
    # Fazer a previsão usando o modelo treinado
    predicted_genre = clf.predict(X)
    prediction_probability = max(clf.predict_proba(X)[0])
    
    return predicted_genre[0], prediction_probability

# Solicitar ao usuário o título e a sinopse
title = input("Insira o título do filme: ")
synopsis = input("Insira uma curta sinopse: ")

# Fazer a previsão e imprimir o resultado
genre, probability = predict_genre(title, synopsis)
print(f"O gênero previsto é {genre} com uma probabilidade de {probability:.2f}.")
