#!/usr/bin/env python
# coding: utf-8

# # Instalação
# 
# Libs necessárias: pip install -r requirements.txt

# In[80]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import warnings


# # 0. Lendo o arquivo

# In[81]:


df = pd.read_csv( 'dataset/top10K-TMDB-movies.csv' )
df.shape


# # 1. Pré-processamento da base de dados

# ## 1.1 Valores Ausentes

# ### 1.1.1 Ver se há algum valor ausente no DataFrame
# 
# Retorna `True` se houver pelo menos um valor ausente.

# In[82]:


df.isnull().values.any()


# ### 1.1.2 Contar quantos valores ausentes há em cada coluna

# In[83]:


df.isnull().sum()


# ### 1.1.3 Exibir linhas que possuem valores ausentes

# In[84]:


df[ df.isnull().any( axis = 1 ) ]


# ### 1.1.4 Removendo linhas com valores ausentes

# In[85]:


df.dropna( subset = [ 'overview' ], inplace = True )
df.reset_index( drop = True, inplace = True )


# ## 1.2 Valores Duplicados

# ### 1.2.1 Ver se há alguma linha duplicada
# 
# Retorna `True` se houver ao menos uma linha duplicada

# In[86]:


df.duplicated().any()


# ### 1.2.2 Contar o número de linhas duplicadas

# In[87]:


df.duplicated().sum()


# ### 1.2.3 Ver quais são as linhas duplicadas

# In[88]:


df[ df.duplicated() ]


# # 2. Seleção de Atributos

# In[89]:


df = df[ [ 'id', 'title', 'overview', "genre" ] ]
df.shape


# In[90]:


df[ 'tags' ] = df[ 'overview' ] + df[ 'genre' ]
df.shape


# In[91]:


df_new = df.drop( columns = [ 'overview', 'genre' ] )
df_new.shape


# # 2. Similaridade de Cossenos
# 
# Imagine que cada sinopse de filme, depois de ser transformada em números (seja por Bag of Words ou TF-IDF), se torna um vetor em um espaço com muitas dimensões (onde cada dimensão corresponde a uma palavra do vocabulário).
# 
# ## 2.1 **A Ideia Central**
# 
# A similaridade de cossenos **não mede a distância** entre as pontas desses vetores, mas sim o ângulo entre eles.
# 
# - Se dois vetores apontam para **direções muito parecidas**, o ângulo entre eles é pequeno, e a similaridade de cossenos é alta (próxima de 1). Isso sugere que as sinopses usam palavras/termos de forma parecida, indicando temas semelhantes.
# - Se dois vetores apontam para **direções completamente diferentes** (são ortogonais, formam um ângulo de 90 graus), a similaridade de cossenos é 0. Isso sugere que as sinopses tratam de assuntos muito distintos, com poucas palavras-chave em comum.
# - Se dois vetores apontam para **direções opostas**, o ângulo é de 180 graus, e a similaridade de cossenos é -1. Na prática, com vetores de texto baseados em contagens (BoW) ou TF-IDF, que geralmente não têm valores negativos, a similaridade varia entre 0 e 1.
# 
# ## 2.2 **Exemplo**
# 
# Vamos simplificar muito e imaginar um vocabulário minúsculo com apenas 3 palavras: "ação", "comédia", "drama".
# 
# Agora, vamos representar 3 filmes com vetores baseados na contagem dessas palavras em suas (hipotéticas) sinopses:
# 
# - Filme A: "Muita ação e um pouco de drama."
#     - Vetor A = [ação: 2, comédia: 0, drama: 1] -> [2, 0, 1]
# - Filme B: "Pura ação!"
#     - Vetor B = [ação: 1, comédia: 0, drama: 0] -> [1, 0, 0]
# - Filme C: "Uma comédia dramática."
#     - Vetor C = [ação: 0, comédia: 1, drama: 1] -> [0, 1, 1]
# 
# Agora, vamos calcular a similaridade de cossenos (sem entrar nos detalhes matemáticos exatos aqui, apenas a intuição):
# 
# - Similaridade(A, B):
#     - Ambos têm "ação". O Filme A também tem "drama", o Filme B não.
#     - Os vetores [2, 0, 1] e [1, 0, 0] apontam em direções relativamente parecidas (ambos têm um forte componente na dimensão "ação").
#     - A similaridade de cossenos será alta, mas não 1 (porque A também tem "drama").
# - Similaridade(A, C):
#     - Ambos têm "drama". O Filme A tem "ação", o Filme C tem "comédia".
#     - Os vetores [2, 0, 1] e [0, 1, 1] compartilham a dimensão "drama", mas divergem nas outras ("ação" vs "comédia").
#     - A similaridade de cossenos será média-baixa.
# - Similaridade(B, C):
#     - Não compartilham nenhuma palavra do nosso vocabulário ("ação" vs "comédia", "drama").
#     - Os vetores [1, 0, 0] e [0, 1, 1] apontam em direções muito diferentes.
#     - A similaridade de cossenos será próxima de 0.
# 
# ## 2.3 **Matematicamente**
# 
# A fórmula é:
# 
# $$
# \text{similaridade(A,B)} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}
# $$
# 
# Onde:
# - $ \mathbf{A} \cdot \mathbf{B} $ é o produto escalar dos vetores (soma da multiplicação de cada componente correspondente).
# - $ \|\mathbf{A}\|$ e $ \|\mathbf{B}\| $ são as magnitudes (ou "comprimentos") dos vetores.

# # 3. Vetorização das Sinopses
# 
# Para que os algoritmos de machine learning compreendam e processem dados textuais, como as sinopses ou tags dos filmes, precisamos primeiro convertê-los em vetores numéricos. Bag of Words (BoW) e TF-IDF são duas abordagens fundamentais e amplamente utilizadas para essa tarefa de **vetorização**.

# ## 3.1 Bag of Worlds (BoW)
# 
# * **Ideia Central:** Representar um texto pela frequência (contagem) de suas palavras, ignorando a ordem ou a gramática.
# * **Como Funciona:**
#     1.  **Construção do Vocabulário:** Primeiro, cria-se um vocabulário que contém todas as palavras únicas encontradas em *todos* os textos do conjunto de dados (corpus).
#     2.  **Criação do Vetor:** Para cada texto individual, cria-se um vetor numérico. O tamanho desse vetor é igual ao tamanho do vocabulário.
#     3.  **Contagem:** Cada posição no vetor corresponde a uma palavra específica do vocabulário. O valor nessa posição é simplesmente quantas vezes aquela palavra aparece no texto que está sendo vetorizado.
# * **Analogia:** Imagine que você joga todas as palavras de um documento dentro de uma "sacola" (bag). O BoW apenas conta quantas vezes cada palavra única está presente nessa sacola.
# * **Características:**
#     * Simples e fácil de entender.
#     * Perde a informação sobre a ordem das palavras.
#     * Trata todas as palavras da mesma forma; palavras muito frequentes (como artigos ou preposições, se não removidas por *stop words*) podem dominar o vetor, mesmo sem carregar tanto significado distintivo.
# * **Implementação:** Em Scikit-learn, é feito usando o `CountVectorizer`.

# In[92]:


bow_vectorizer = CountVectorizer( max_features = 10000, stop_words = 'english' )
bow_matrix = bow_vectorizer.fit_transform( df_new[ 'tags' ].values.astype( 'U' ) ).toarray()

cosine_sim_bow = cosine_similarity( bow_matrix )


# ## 3.2 TF-IDF (Term Frequency-Inverse Document Frequency)
# 
# * **Ideia Central:** É uma evolução do BoW que tenta dar mais importância às palavras que são significativas para um texto específico, ao mesmo tempo que diminui o peso das palavras que são muito comuns em *todos* os textos. O peso de cada palavra reflete sua importância naquele documento dentro do contexto de toda a coleção.
# * **Como Funciona:** Calcula um score para cada palavra em cada documento, baseado em dois fatores:
#     1.  **TF (Term Frequency - Frequência do Termo):** Mede quantas vezes uma palavra aparece no documento atual. Quanto mais vezes aparece, maior o TF (geralmente normalizado). Indica a importância da palavra *dentro* daquele documento.
#     2.  **IDF (Inverse Document Frequency - Frequência Inversa do Documento):** Mede o quão rara ou comum a palavra é em *toda a coleção* de documentos. É calculado como o logaritmo da divisão do número total de documentos pelo número de documentos que contêm a palavra ($ \log(\frac{N}{df_t}) $). Palavras que aparecem em muitos documentos terão um IDF baixo (ex: "filme", "ele"), enquanto palavras que aparecem em poucos documentos terão um IDF alto (ex: "multiverso", "hobbit"). Indica a informatividade ou poder discriminatório da palavra.
#     3.  **Score TF-IDF:** É o produto $ TF \times IDF $. Uma palavra terá um score TF-IDF alto se for frequente em um documento específico (alto TF) mas rara no conjunto geral de documentos (alto IDF).
# * **Características:**
#     * Dá mais peso a palavras que são boas indicadoras do conteúdo específico de um documento.
#     * Reduz o peso de palavras muito comuns que não ajudam a distinguir entre documentos.
#     * Geralmente leva a melhores resultados do que o BoW em tarefas de busca, classificação e cálculo de similaridade.
# * **Implementação:** Em Scikit-learn, é feito usando o `TfidfVectorizer`.

# In[93]:


tfidf_vectorizer = TfidfVectorizer( max_features = 10000, stop_words = 'english' )
tfidf_matrix = tfidf_vectorizer.fit_transform( df_new[ 'tags' ].values.astype( 'U' ) ).toarray()

cosine_sim_tfidf = cosine_similarity( tfidf_matrix )


# # 4. Sistema de Recomendação

# In[94]:


# Para facilitar a busca pelo filme, criamos uma Série que mapeia títulos para índices
# Isso é mais eficiente que procurar no DataFrame toda vez
indices = pd.Series( df.index, index = df[ 'title' ] ).drop_duplicates()


def recommend_by_synopsis( titulo, cosine_sim_matrix, data = df, mapping = indices, top_n = 10 ):
    """
    Gera recomendações de filmes baseadas na similaridade de sinopses,
    retornando títulos e suas pontuações de similaridade.

    Args:
        titulo (str): O título do filme base para a recomendação.
        cosine_sim_matrix (np.array): A matriz de similaridade de cossenos pré-calculada.
        data (pd.DataFrame): O DataFrame contendo os dados dos filmes.
        mapping (pd.Series): Mapeamento de títulos para índices.
        top_n (int): Número de recomendações.

    Returns:
        list: Uma lista de tuplas (titulo_recomendado, similaridade),
              ou uma mensagem de erro (str).
    """
    try:
        # Obter o índice do filme que corresponde ao título
        if titulo not in mapping:
            # Tenta encontrar correspondência ignorando maiúsculas/minúsculas e espaços extras
            matches = [ t for t in mapping.index if t.lower().strip() == titulo.lower().strip() ]
            if not matches:
                # Se ainda não encontrou, retorna erro
                raise KeyError
            # Pega o primeiro título correspondente (pode haver duplicatas exatas no índice se não tratadas)
            actual_title = matches[ 0 ]
            idx = mapping[ actual_title ]
            print( f"Nota: Buscando por '{actual_title}' (correspondência encontrada para '{titulo}')" )
        else:
            idx = mapping[ titulo ]  # Pega o índice diretamente se o título for exato

        # Obter as pontuações de similaridade de todos os filmes com este filme
        sim_scores = list( enumerate( cosine_sim_matrix[ idx ] ) )

        # Ordenar os filmes com base nas pontuações de similaridade (decrescente)
        sim_scores = sorted( sim_scores, key = lambda x: x[ 1 ], reverse = True )

        # Obter as pontuações e índices dos 'top_n' filmes mais similares (ignorando o próprio filme)
        sim_scores = sim_scores[ 1:top_n + 1 ]  # Pega do segundo até top_n+1

        # Criar a lista de tuplas (título, score)
        recomendacoes = [ ]
        for index, score in sim_scores:
            # Verifica se o índice é válido para evitar erros
            if index < len( data ):
                titulo_filme = data[ 'title' ].iloc[ index ]
                recomendacoes.append( (titulo_filme, score) )
            else:
                print( f"Aviso: Índice {index} fora dos limites do DataFrame. Ignorando." )

        return recomendacoes

    except KeyError:
        # Mensagem de erro mais informativa
        suggestion = [ t for t in mapping.index if titulo.lower() in t.lower() ][ :5 ]
        error_msg = f"Erro: Filme '{titulo}' não encontrado no dataset."
        if suggestion:
            error_msg += f" Você quis dizer algum destes? {suggestion}"
        return error_msg
    except Exception as e:
        return f"Ocorreu um erro inesperado: {e}"


# ## 4.1 Aplicação

# In[95]:


filme_exemplo = "The Dark Knight Rises"
filme_exemplo = "Bohemian Rhapsody"

print( f"\n--- Recomendações para '{filme_exemplo}' (BoW) ---" )
recomendacoes = recommend_by_synopsis( filme_exemplo, cosine_sim_matrix = cosine_sim_bow, top_n = 5 )

if isinstance( recomendacoes, list ):
    for i, (filme, score) in enumerate( recomendacoes ):
        print( f"{i + 1}. {filme} (Similaridade: {score:.3f})" )
else:
    print( recomendacoes )

print( f"\n--- Recomendações para '{filme_exemplo}' (TF-IDF) ---" )
recomendacoes_2 = recommend_by_synopsis( filme_exemplo, cosine_sim_matrix = cosine_sim_tfidf, top_n = 5 )

if isinstance( recomendacoes_2, list ):
    for i, (filme, score) in enumerate( recomendacoes_2 ):
        print( f"{i + 1}. {filme} (Similaridade: {score:.3f})" )
else:
    print( recomendacoes_2 )


# # 5. Sentence-BERT
# 
# Sentence-BERT é uma técnica desenvolvida para superar uma limitação dos modelos BERT originais em tarefas de comparação de sentenças. Enquanto o BERT puro é poderoso, ele não gera, por padrão, *embeddings* (vetores) de sentenças que sejam semanticamente significativos e diretamente comparáveis usando similaridade de cossenos. Comparar a similaridade entre N sentenças exigiria N*(N-1)/2 passagens pelo modelo BERT, o que é computacionalmente inviável para grandes conjuntos de dados.
# 
# **O Objetivo do Sentence-BERT:**
# 
# O objetivo principal é criar vetores de tamanho fixo (embeddings) para sentenças individuais, de forma que a similaridade semântica entre as sentenças se traduza na proximidade desses vetores no espaço vetorial (geralmente medido pela similaridade de cossenos).
# 
# **Como Ele Consegue Isso:**
# 
# 1.  **Base Pré-treinada:** Utiliza um modelo transformer pré-treinado (como BERT, RoBERTa, MPNet, etc.) como sua base.
# 2.  **Pooling:** Adiciona uma operação de *pooling* (como tirar a média dos vetores de saída de todas as palavras da sentença - MEAN Pooling) sobre a saída do modelo base. Isso transforma a sequência de vetores de palavras em um único vetor de tamanho fixo que representa a sentença inteira.
# 3.  **Fine-tuning Específico (A Chave):** A etapa crucial é o *fine-tuning*. O Sentence-BERT é treinado usando arquiteturas de redes **siamesas** ou **triplet**.
#     * **Redes Siamesas:** O modelo processa duas sentenças simultaneamente e ajusta seus pesos para que o cálculo de similaridade (ex: cosseno) entre os embeddings resultantes se aproxime de um valor desejado (ex: 1 para pares sinônimos, 0 para pares não relacionados).
#     * **Redes Triplet:** O modelo vê três sentenças (âncora, positiva, negativa) e aprende a fazer o embedding da âncora ser mais próximo do embedding da positiva do que do embedding da negativa.
#     * Esse fine-tuning ensina o modelo a produzir embeddings que realmente refletem o significado semântico para fins de comparação.
# 
# **Uso (Inferência):**
# 
# Uma vez que o modelo está fine-tunado (ou quando usamos um modelo pré-treinado já fine-tunado com essa técnica, como `all-mpnet-base-v2` ou `paraphrase-multilingual-mpnet-base-v2`), podemos simplesmente passar **uma única sentença** por ele para obter seu embedding. Isso é muito rápido e eficiente. Com todos os embeddings gerados, podemos calcular similaridades (ex: cosseno) entre quaisquer pares de vetores de forma quase instantânea.
# 
# **Resultado:**
# 
# Obtemos representações vetoriais densas e semanticamente ricas para cada sentença (ou, no nosso caso, para cada `tag` de filme), permitindo comparações de significado muito mais eficazes do que com métodos tradicionais como Bag-of-Words ou TF-IDF puro.

# In[96]:


# --- Configurações do Sentence-BERT ---
MODEL_NAME_SBERT = 'all-mpnet-base-v2'
# Arquivo para salvar/carregar os embeddings calculados (economiza tempo)
EMBEDDINGS_FILE_SBERT = 'movie_tags_embeddings_sbert.npy'

# --- Gerar ou Carregar Embeddings ---
embeddings_sbert = None  # Inicializa como None

# Tenta carregar embeddings se o arquivo já existir
if os.path.exists( EMBEDDINGS_FILE_SBERT ):
    print( f"Tentando carregar embeddings pré-calculados de '{EMBEDDINGS_FILE_SBERT}'..." )
    try:
        embeddings_sbert = np.load( EMBEDDINGS_FILE_SBERT )
        print( f"Embeddings carregados com sucesso. Shape: {embeddings_sbert.shape}" )
        # Validação rápida: o número de linhas no embedding deve ser igual ao número de filmes em df_new
        if embeddings_sbert.shape[ 0 ] != len( df_new ):
            print(
                    f"Aviso: Número de embeddings ({embeddings_sbert.shape[ 0 ]}) "
                    f"não corresponde ao tamanho do DataFrame df_new ({len( df_new )})." )
            print( "Os embeddings serão gerados novamente." )
            embeddings_sbert = None  # Força a regeração
    except Exception as e:
        print( f"Erro ao carregar o arquivo de embeddings '{EMBEDDINGS_FILE_SBERT}': {e}" )
        print( "Os embeddings serão gerados novamente." )
        embeddings_sbert = None


# In[97]:


# Se os embeddings não foram carregados (arquivo não existe ou houve erro/incompatibilidade)
if embeddings_sbert is None:
    try:
        # Carregar o modelo SentenceTransformer
        # A primeira vez pode levar tempo para baixar o modelo
        model_sbert = SentenceTransformer( MODEL_NAME_SBERT )

        # Preparar a lista de textos a serem codificados (a coluna 'tags')
        # Usamos df_new['tags'] que foi pré-processada (astype('U')) implicitamente antes,
        # mas pegar do df original e garantir tipo string é mais seguro.
        # Usar .tolist() para passar uma lista de strings para o encode.
        print( "Preparando a lista de 'tags' para codificação..." )
        tags_list = df_new[ 'tags' ].astype( str ).tolist()  # Garante que são strings

        # Gerar os embeddings
        print( "Gerando embeddings com Sentence-BERT... (Isso pode levar vários minutos)" )
        embeddings_sbert = model_sbert.encode( tags_list, show_progress_bar = True )

        print( f"Embeddings gerados com sucesso! Shape: {embeddings_sbert.shape}" )  # Ex: (9986, 768)

        # Salvar os embeddings calculados para não precisar gerar de novo
        try:
            print( f"Salvando embeddings em '{EMBEDDINGS_FILE_SBERT}' para uso futuro..." )
            np.save( EMBEDDINGS_FILE_SBERT, embeddings_sbert )
        except Exception as e:
            print( f"Atenção: Não foi possível salvar os embeddings em '{EMBEDDINGS_FILE_SBERT}'. Erro: {e}" )

    except Exception as e:
        print( f"Erro crítico durante o processo do Sentence-BERT: {e}" )
        print( "Verifique sua conexão com a internet se for a primeira vez baixando o modelo." )
        embeddings_sbert = None  # Garante que não tentará usar embeddings


# In[98]:


# Calcular Similaridade de Cossenos (usando Embeddings SBERT)
cosine_sim_sbert = None
if embeddings_sbert is not None:
    try:
        print( "\nCalculando a matriz de similaridade de cossenos a partir dos embeddings SBERT..." )
        cosine_sim_sbert = cosine_similarity( embeddings_sbert )
        print( f"Matriz de similaridade (Sentence-BERT) calculada. Shape: {cosine_sim_sbert.shape}" )
    except Exception as e:
        print( f"Erro ao calcular a similaridade de cossenos com embeddings SBERT: {e}" )
else:
    print( "\nNão foi possível calcular a similaridade SBERT pois os embeddings não estão disponíveis." )


# ## 5.1 Aplicação

# In[99]:


if cosine_sim_sbert is not None:
    print( f"\n--- Recomendações para '{filme_exemplo}' (Sentence-BERT) ---" )

    recomendacoes_sbert = recommend_by_synopsis(
            filme_exemplo,
            cosine_sim_matrix = cosine_sim_sbert,
            data = df,
            mapping = indices,
            top_n = 5
    )

    if isinstance( recomendacoes_sbert, list ):
        if not recomendacoes_sbert:
            print( "Nenhuma recomendação encontrada." )
        else:
            for i, (filme, score) in enumerate( recomendacoes_sbert ):
                print( f"{i + 1}. {filme} (Similaridade Semântica: {score:.3f})" )
    else:
        # Imprime a mensagem de erro retornada pela função
        print( recomendacoes_sbert )

else:
    print( "Não é possível gerar recomendações com Sentence-BERT devido a erros na geração/cálculo." )


# # 6. Clustering de Filmes

# In[100]:


warnings.filterwarnings( "ignore", category = FutureWarning, module = 'sklearn.cluster._kmeans' )
warnings.filterwarnings( "ignore", category = UserWarning, module = 'sklearn.cluster._kmeans' )

# Verificar se temos os embeddings SBERT disponíveis
if 'embeddings_sbert' not in globals() or embeddings_sbert is None:
    print( "Erro: Embeddings SBERT ('embeddings_sbert') não encontrados." )
    print( "Por favor, execute a célula anterior (Seção 5) para gerar ou carregar os embeddings." )
else:
    print( f"Usando embeddings SBERT com shape: {embeddings_sbert.shape}" )

    # --- Passo 6.1: Escolhendo o Número de Clusters (K) ---

    # Abordagem 1: Método do Cotovelo (Elbow Method)
    # Calcula a inércia (soma das distâncias quadráticas intra-cluster) para diferentes valores de K.
    # Procuramos um 'cotovelo' no gráfico, onde a redução da inércia começa a diminuir drasticamente.

    print( "\nCalculando inércia para o Método do Cotovelo..." )
    inertia = [ ]
    # Definir um range de K para testar (ajuste conforme necessário)
    # Para 10k filmes, testar até 50 ou 100 pode ser razoável
    k_range = range( 2, 51, 2 )  # Ex: De 2 a 50, de 2 em 2

    try:
        for k in k_range:
            kmeans_elbow = KMeans(
                    n_clusters = k, random_state = 42, n_init = 10 )  # n_init=10 é o padrão e recomendado
            kmeans_elbow.fit( embeddings_sbert )
            inertia.append( kmeans_elbow.inertia_ )
            print( f"  K={k}, Inércia={kmeans_elbow.inertia_:.2f}" )

        # Plotar o gráfico do cotovelo
        plt.figure( figsize = (10, 6) )
        plt.plot( k_range, inertia, marker = 'o' )
        plt.title( 'Método do Cotovelo (Elbow Method)' )
        plt.xlabel( 'Número de Clusters (K)' )
        plt.ylabel( 'Inércia (WSS - Within-Cluster Sum of Squares)' )
        plt.xticks( k_range )
        plt.grid( True )
        plt.show()

        print( "\nAnalise o gráfico acima para identificar um 'cotovelo'." )
        print(
                "O 'cotovelo' sugere um valor de K onde adicionar mais clusters não melhora significativamente a coesão intra-cluster." )

    except Exception as e:
        print( f"\nErro ao calcular o Método do Cotovelo: {e}" )
        print( "Verifique se os embeddings são válidos." )

    # Abordagem 2: Pontuação de Silhueta (Opcional, pode ser lento)
    # Mede quão similar um objeto é ao seu próprio cluster comparado a outros clusters.
    # Valores próximos de 1 são melhores. Procuramos um pico no gráfico.
    # NOTA: Calcular silhouette score para muitos pontos e Ks pode ser demorado.
    # calculate_silhouette = False # Mude para True se quiser calcular
    calculate_silhouette = True  # Vamos tentar calcular, pode demorar um pouco

    if calculate_silhouette:
        print( "\nCalculando Pontuação de Silhueta (pode ser demorado)..." )
        silhouette_scores = [ ]
        # Usar um range menor para K na silhueta devido ao custo computacional
        k_range_silhouette = range( 2, 21, 2 )  # Ex: De 2 a 20, de 2 em 2

        try:
            for k in k_range_silhouette:
                kmeans_silhouette = KMeans( n_clusters = k, random_state = 42, n_init = 10 )
                cluster_labels_temp = kmeans_silhouette.fit_predict( embeddings_sbert )
                # sample_size reduz o número de pontos para acelerar o cálculo da silhueta
                # Usar None para calcular em todos os dados (mais preciso, mais lento)
                score = silhouette_score(
                        embeddings_sbert, cluster_labels_temp, metric = 'cosine', sample_size = 2000,
                        random_state = 42 )
                silhouette_scores.append( score )
                print( f"  K={k}, Silhouette Score={score:.3f}" )

            # Plotar o gráfico da silhueta
            plt.figure( figsize = (10, 6) )
            plt.plot( k_range_silhouette, silhouette_scores, marker = 'o' )
            plt.title( 'Pontuação Média de Silhueta' )
            plt.xlabel( 'Número de Clusters (K)' )
            plt.ylabel( 'Silhouette Score Médio' )
            plt.xticks( k_range_silhouette )
            plt.grid( True )
            plt.show()

            print( "\nProcure por um pico no gráfico de silhueta, indicando um K com clusters bem definidos." )

        except Exception as e:
            print( f"\nErro ao calcular a Pontuação de Silhueta: {e}" )

    # --- Passo 6.2: Treinar K-Means com o K Escolhido ---

    # **DECISÃO IMPORTANTE:** Analise os gráficos acima (cotovelo e/ou silhueta)
    # e escolha um valor para K que pareça razoável.
    # Se não houver um ponto claro, pode ser necessário experimentar ou usar conhecimento do domínio.
    # Vamos escolher um valor exemplo, **VOCÊ DEVE AJUSTAR ESTE VALOR!**
    K_ESCOLHIDO = 20  # <--- AJUSTE AQUI COM BASE NOS GRÁFICOS OU SUA INTUIÇÃO

    print( f"\nTreinando K-Means final com K={K_ESCOLHIDO}..." )

    try:
        kmeans = KMeans( n_clusters = K_ESCOLHIDO, random_state = 42, n_init = 10 )
        kmeans.fit( embeddings_sbert )
        print( "Treinamento K-Means concluído." )

        # --- Passo 6.3: Obter e Adicionar Rótulos dos Clusters ---
        cluster_labels = kmeans.labels_
        print( f"Rótulos dos clusters gerados (Array de {len( cluster_labels )} elementos)" )

        # Adicionar os rótulos ao DataFrame original ('df') para análise
        # Certifique-se que o índice do DataFrame 'df' ainda está alinhado com os 'embeddings_sbert'
        # (Deve estar se você seguiu os passos anteriores de reset_index)
        df[ 'cluster_sbert' ] = cluster_labels
        # Opcional: adicionar também ao df_new se for usá-lo
        if 'df_new' in globals():
            df_new[ 'cluster_sbert' ] = cluster_labels

        print( f"Coluna 'cluster_sbert' adicionada ao DataFrame 'df'." )

        # --- Passo 6.4: Analisar os Clusters ---

        print( "\nAnálise Inicial dos Clusters:" )

        # Contar quantos filmes caíram em cada cluster
        print( "\nDistribuição de filmes por cluster:" )
        print( df[ 'cluster_sbert' ].value_counts().sort_index() )

        # Ver exemplos de filmes de alguns clusters específicos
        print( "\nExemplos de filmes por cluster:" )
        # Mude os números dos clusters para ver exemplos diferentes
        clusters_para_ver = [ 0, 1, K_ESCOLHIDO // 2, K_ESCOLHIDO - 1 ]  # Pega alguns clusters exemplo

        for cluster_num in clusters_para_ver:
            if cluster_num < K_ESCOLHIDO:  # Verifica se o cluster existe
                print( f"\n--- Cluster {cluster_num} ---" )
                # Pega os primeiros 5-10 filmes do cluster
                filmes_no_cluster = df[ df[ 'cluster_sbert' ] == cluster_num ][ 'title' ].head( 10 ).tolist()
                if filmes_no_cluster:
                    for filme_titulo in filmes_no_cluster:
                        print( f"  - {filme_titulo}" )
                else:
                    print( "  (Nenhum filme encontrado neste cluster - pode indicar problema)" )
            else:
                print( f"\nCluster {cluster_num} não existe (K={K_ESCOLHIDO})." )

        print( "\n--- Fim da Seção 6: Clustering de Filmes ---" )
        print( "Próximo passo: Usar esses clusters para fazer recomendações." )

    except Exception as e:
        print( f"\nErro ao treinar o K-Means final ou analisar clusters: {e}" )


# # 7. Recomendação por Cluster

# In[102]:


# In[ ]: # Nova Célula - Recomendação por Cluster

# 7. Sistema de Recomendação Baseado em Clusters SBERT

print( "\n--- Iniciando Seção 7: Recomendação Baseada em Cluster ---" )

# Verificar se a coluna de cluster e a coluna de score existem no DataFrame df
if 'cluster_sbert' not in df.columns:
    print( "Erro: Coluna 'cluster_sbert' não encontrada no DataFrame 'df'." )
    print( "Execute a célula anterior (Seção 6) para gerar os clusters." )
elif 'score' not in df.columns:
    print( "Aviso: Coluna 'score' (Weighted Rating) não encontrada no DataFrame 'df'." )
    print( "As recomendações do cluster não serão ordenadas por qualidade/popularidade." )
    # Criar score padrão para evitar erros, mas recomendações serão baseadas na ordem do DF
    df[ 'score' ] = 0.0  # Ou algum outro valor padrão, como df['vote_average']
    recommend_by_score = False
else:
    print( "Coluna 'cluster_sbert' e 'score' encontradas. Pronto para recomendar por cluster." )
    recommend_by_score = True


def recomendar_por_cluster( titulo, data = df, mapping = indices, top_n = 10 ):
    """
    Gera recomendações de filmes encontrando outros filmes no mesmo cluster
    do filme de entrada, ordenados por 'score' (Weighted Rating).

    Args:
        titulo (str): O título do filme base para a recomendação.
        data (pd.DataFrame): DataFrame com colunas 'title', 'cluster_sbert', 'score'.
        mapping (pd.Series): Mapeamento de títulos para índices.
        top_n (int): Número de recomendações a serem retornadas.

    Returns:
        list: Uma lista de títulos de filmes recomendados,
              ou uma mensagem de erro (str).
    """
    try:
        # Obter o índice do filme que corresponde ao título
        if titulo not in mapping:
            matches = [ t for t in mapping.index if t.lower().strip() == titulo.lower().strip() ]
            if not matches: raise KeyError
            actual_title = matches[ 0 ]
            idx = mapping[ actual_title ]
        else:
            idx = mapping[ titulo ]

        # Verificar se o índice é válido e se a coluna cluster existe
        if idx >= len( data ) or 'cluster_sbert' not in data.columns:
            raise ValueError( "Índice do filme inválido ou coluna de cluster ausente." )

        # Obter o número do cluster do filme de entrada
        input_movie_cluster = data.iloc[ idx ][ 'cluster_sbert' ]
        print( f"Filme '{data.iloc[ idx ][ 'title' ]}' pertence ao Cluster: {input_movie_cluster}" )

        # Filtrar o DataFrame para obter outros filmes do mesmo cluster
        # Exclui o próprio filme de entrada usando o índice
        cluster_movies = data[ (data[ 'cluster_sbert' ] == input_movie_cluster) & (data.index != idx) ]

        if cluster_movies.empty:
            return f"Nenhum outro filme encontrado no Cluster {input_movie_cluster} para recomendar."

        # Ordenar os filmes do cluster pelo 'score' (Weighted Rating) em ordem decrescente
        if recommend_by_score and 'score' in cluster_movies.columns:
            cluster_movies_sorted = cluster_movies.sort_values( 'score', ascending = False )
        else:
            # Se não for ordenar por score, apenas pega os primeiros que aparecerem
            cluster_movies_sorted = cluster_movies

        # Selecionar os top_n títulos
        recommendations = cluster_movies_sorted[ 'title' ].head( top_n ).tolist()

        return recommendations

    except KeyError:
        # Mensagem de erro mais informativa (reutilizada da função anterior)
        suggestion = [ t for t in mapping.index if titulo.lower() in t.lower() ][ :5 ]
        error_msg = f"Erro: Filme '{titulo}' não encontrado no dataset."
        if suggestion: error_msg += f" Você quis dizer algum destes? {suggestion}"
        return error_msg
    except ValueError as ve:
        print( f"Erro ao processar filme/cluster: {ve}" )
        return "Não foi possível gerar recomendações devido a um erro interno."
    except Exception as e:
        return f"Ocorreu um erro inesperado durante a recomendação por cluster: {e}"


# ## 7.1 Aplicação

# In[103]:


if 'cluster_sbert' in df.columns:  # Só executa se os clusters foram gerados
    filme_exemplo_cluster = "The Dark Knight Rises"  # Use o mesmo filme ou outro

    print( f"\n--- Recomendações por Cluster para '{filme_exemplo_cluster}' ---" )
    recomendacoes_cluster = recomendar_por_cluster( filme_exemplo_cluster, top_n = 10 )  # Pega top 10 do cluster

    if isinstance( recomendacoes_cluster, list ):
        if not recomendacoes_cluster:
            print( "Nenhuma recomendação encontrada." )
        else:
            for i, filme in enumerate( recomendacoes_cluster ):
                # Opcional: buscar e mostrar o score do filme recomendado
                # score_rec = df.loc[df['title'] == filme, 'score'].iloc[0]
                # print(f"{i + 1}. {filme} (Score: {score_rec:.2f})")
                print( f"{i + 1}. {filme}" )
    else:
        # Imprime a mensagem de erro retornada pela função
        print( recomendacoes_cluster )

    # Exemplo 2
    filme_exemplo_cluster_2 = "Avatar"
    print( f"\n--- Recomendações por Cluster para '{filme_exemplo_cluster_2}' ---" )
    recomendacoes_cluster_2 = recomendar_por_cluster( filme_exemplo_cluster_2, top_n = 10 )

    if isinstance( recomendacoes_cluster_2, list ):
        if not recomendacoes_cluster_2:
            print( "Nenhuma recomendação encontrada." )
        else:
            for i, filme in enumerate( recomendacoes_cluster_2 ):
                print( f"{i + 1}. {filme}" )
    else:
        print( recomendacoes_cluster_2 )
else:
    print( "\nNão é possível gerar recomendações por cluster pois a coluna 'cluster_sbert' não foi criada." )

