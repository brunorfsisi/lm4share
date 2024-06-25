import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Função para gerar embeddings
def generate_embeddings(text, model):
    words = text.split()
    embeddings = model.encode(words)
    return embeddings, words

# Função para reduzir a dimensionalidade para visualização
def reduce_dimensionality(embeddings):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

# Função para plotar os embeddings com melhor visualização do texto
def plot_embeddings(reduced_embeddings, words):
    plt.figure(figsize=(12, 8))
    
    # Usar uma paleta de cores para os pontos
    palette = sns.color_palette("hsv", len(words))
    
    for i, word in enumerate(words):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], color=palette[i], s=100, alpha=0.5)
        plt.annotate(word, 
                     (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center', 
                     fontsize=8, 
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    plt.title("Embeddings Visualizados em 2D")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.grid(True)
    st.pyplot(plt)

# Aplicativo Streamlit
st.title("Exploração de Modelos LLM com Embeddings")

st.write("Digite uma frase ou texto para gerar embeddings e visualizar em uma nuvem de pontos.")

# Input do usuário para a frase ou texto
user_input = st.text_area("Digite aqui o texto:", "")

if user_input:
    model_choice = st.selectbox("Escolha o modelo de embeddings", ["distilbert-base-nli-mean-tokens", "paraphrase-MiniLM-L6-v2"])
    model = SentenceTransformer(model_choice)

    with st.spinner("Gerando embeddings..."):
        embeddings, words = generate_embeddings(user_input, model)

    with st.spinner("Reduzindo dimensionalidade para visualização..."):
        reduced_embeddings = reduce_dimensionality(embeddings)

    st.write("Nuvem de pontos dos embeddings:")
    plot_embeddings(reduced_embeddings, words)

