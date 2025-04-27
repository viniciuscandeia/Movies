#!/usr/bin/env python
# coding: utf-8

# # Instalação
# 
# Libs necessárias: pip install -r requirements.txt

# In[24]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# # 0. Lendo o arquivo

# In[25]:


df = pd.read_csv( 'dataset/top10K-TMDB-movies.csv' )
df.shape


# # 1. Pré-processamento da base de dados

# ## 1.1 Valores Ausentes

# ### 1.1.1 Ver se há algum valor ausente no DataFrame
# 
# Retorna `True` se houver pelo menos um valor ausente.

# In[26]:


df.isnull().values.any()


# ### 1.1.2 Contar quantos valores ausentes há em cada coluna

# In[27]:


df.isnull().sum()


# ### 1.1.3 Exibir linhas que possuem valores ausentes

# In[28]:


df[ df.isnull().any( axis = 1 ) ]


# ### 1.1.4 Removendo linhas com valores ausentes

# In[29]:


df.dropna( subset = [ 'overview' ], inplace = True )
df.reset_index( drop = True, inplace = True )


# ## 1.2 Valores Duplicados

# ### 1.2.1 Ver se há alguma linha duplicada
# 
# Retorna `True` se houver ao menos uma linha duplicada

# In[30]:


df.duplicated().any()


# ### 1.2.2 Contar o número de linhas duplicadas

# In[31]:


df.duplicated().sum()


# ### 1.2.3 Ver quais são as linhas duplicadas

# In[32]:


df[ df.duplicated() ]


# # 2. Seleção de Atributos

# In[33]:


df = df[ [ 'id', 'title', 'overview', "genre" ] ]
df.shape


# In[34]:


df[ 'tags' ] = df[ 'overview' ] + df[ 'genre' ]
df.shape


# In[35]:


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

# ## 3.1 Bag of Worlds (BoW)
# 
# Cria um vocabulário com todas as palavras únicas presentes nas sinopses. Para cada sinopse, ele cria um vetor onde cada posição corresponde a uma palavra do vocabulário, e o valor nessa posição é simplesmente a contagem de quantas vezes aquela palavra aparece na sinopse.
# 

# In[36]:


bow_vectorizer = CountVectorizer( max_features = 10000, stop_words = 'english' )
bow_matrix = bow_vectorizer.fit_transform( df_new[ 'tags' ].values.astype( 'U' ) ).toarray()

cosine_sim_bow = cosine_similarity( bow_matrix )


# ## 3.2 TF-IDF (Term Frequency-Inverse Document Frequency)
# 
# Começa contando a frequência das palavras (TF - Term Frequency), mas depois ajusta esse valor com base na frequência inversa do documento (IDF - Inverse Document Frequency). O IDF diminui o peso de palavras que são muito comuns em todas as sinopses (como "filme", "vida", etc., mesmo que não sejam stop words) e aumenta o peso de palavras que são frequentes em poucas sinopses (palavras mais específicas e potencialmente mais descritivas).

# In[37]:


tfidf_vectorizer = TfidfVectorizer( max_features = 10000, stop_words = 'english' )
tfidf_matrix = tfidf_vectorizer.fit_transform( df_new[ 'tags' ].values.astype( 'U' ) ).toarray()

cosine_sim_tfidf = cosine_similarity( tfidf_matrix )


# # 4. Sistema de Recomendação

# In[38]:


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

# In[41]:


filme_exemplo = "The Dark Knight Rises"

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

# In[ ]:




