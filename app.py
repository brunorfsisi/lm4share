import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns

# Texto padrão a ser carregado no campo de texto
default_text = """O que são LLMs?
Os Modelos de Linguagem de Grande Escala, conhecidos como LLMs (Large Language Models), são tipos avançados de modelos de inteligência artificial treinados para entender e gerar linguagem natural. Eles são chamados "de grande escala" porque são treinados em vastas quantidades de dados textuais e possuem um número muito grande de parâmetros, o que lhes permite capturar nuances e complexidades da linguagem humana.

Funcionamento dos LLMs
Os LLMs utilizam redes neurais profundas, especificamente redes transformadoras (Transformers), que são altamente eficazes para tarefas de processamento de linguagem natural (NLP). Durante o treinamento, esses modelos aprendem padrões, contextos e significados das palavras a partir de grandes corpora de texto. Isso permite que eles gerem respostas coerentes e contextualmente relevantes para uma ampla gama de perguntas e comandos.

Aplicações dos LLMs
Geração de Texto: Produção de texto fluente e coerente para diversos fins, como redação de artigos, respostas a perguntas e criação de conteúdo.
Compreensão de Texto: Análise e interpretação de textos, extraindo informações relevantes e respondendo a consultas baseadas no conteúdo.
Tradução Automática: Conversão de texto de um idioma para outro, mantendo a precisão e o contexto.
Assistentes Virtuais: Implementação em chatbots e assistentes virtuais para fornecer suporte ao cliente, responder a perguntas frequentes e realizar tarefas baseadas em comandos de texto.
Elaboração de Relatórios: Geração automática de relatórios detalhados a partir de dados fornecidos, ajudando na tomada de decisões e na comunicação empresarial.

Vantagens dos LLMs
Precisão e Relevância: Capacidade de gerar texto que é contextualmente relevante e preciso, reduzindo a necessidade de revisão manual.
Eficiência: Automatização de tarefas repetitivas de escrita e análise de texto, economizando tempo e recursos.
Flexibilidade: Aplicação em uma ampla gama de setores e usos, desde atendimento ao cliente até criação de conteúdo e análise de dados.

Desafios dos LLMs
Necessidade de Dados de Qualidade: Para manter a precisão e relevância, os LLMs requerem grandes volumes de dados de alta qualidade.
Capacidade Computacional: Treinar e executar LLMs exige recursos computacionais significativos, o que pode ser uma barreira para algumas organizações.
Bias e Ética: Lidar com os vieses presentes nos dados de treinamento e garantir que os modelos gerem respostas éticas e imparciais é um desafio contínuo.

Os LLMs representam um avanço significativo na capacidade das máquinas de entender e gerar linguagem natural, proporcionando benefícios substanciais para diversas aplicações empresariais e tecnológicas."""

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
    st.pyplot(plt.gcf())

# Função para gerar uma frase representativa a partir dos clusters
def generate_representative_sentence(embeddings, words, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(embeddings)
    clusters = kmeans.predict(embeddings)
    
    representative_words = []
    for cluster in range(num_clusters):
        cluster_indices = np.where(clusters == cluster)[0]
        cluster_words = [words[i] for i in cluster_indices]
        # Escolher a palavra mais central no cluster como representante
        cluster_center = kmeans.cluster_centers_[cluster]
        distances = np.linalg.norm(embeddings[cluster_indices] - cluster_center, axis=1)
        representative_word = cluster_words[np.argmin(distances)]
        representative_words.append(representative_word)
    
    return " ".join(representative_words)

# Aplicativo Streamlit
st.title("Exploração de Modelos LLM com Embeddings")

st.write("Digite uma frase ou texto para gerar embeddings e visualizar em uma nuvem de pontos.")

# Input do usuário para a frase ou texto
user_input = st.text_area("Texto em linguagem natural :", default_text, height=400)

if user_input:
    model_choice = st.selectbox("Escolha o modelo de embeddings", ["distilbert-base-nli-mean-tokens", "paraphrase-MiniLM-L6-v2"])
    model = SentenceTransformer(model_choice)

    with st.spinner("Gerando embeddings..."):
        embeddings, words = generate_embeddings(user_input, model)

    with st.spinner("Reduzindo dimensionalidade para visualização..."):
        reduced_embeddings = reduce_dimensionality(embeddings)

    st.write("Nuvem de pontos dos embeddings:")
    plot_embeddings(reduced_embeddings, words)

    st.write("Gerando frase representativa a partir dos clusters:")
    num_clusters = st.slider("Número de clusters:", 2,20, 5)
    representative_sentence = generate_representative_sentence(embeddings, words, num_clusters)
    st.write(f"Frase representativa: {representative_sentence}")

