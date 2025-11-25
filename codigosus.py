import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURAÇÃO INICIAL
# ==============================================================================
st.set_page_config(page_title="Sistema de Recomendação", layout="wide")
st.title("📊 Sistema de Recomendação de Pedidos")

# ==============================================================================
# 1. CARREGAMENTO DOS DADOS (USANDO PARQUET)
# ==============================================================================
@st.cache_data
def load_data():
    try:
        # Lê os arquivos Parquet
        df_users = pd.read_parquet('base_usuarios.parquet')
        df_estab = pd.read_parquet('base_estabelecimentos.parquet')
        df_pedidos = pd.read_parquet('base_pedidos.parquet')
        return df_users, df_pedidos, df_estab
    except Exception as e:
        return None, None, e

st.write("--- Iniciando Processamento ---")
df_users, df_pedidos, df_estab_info = load_data()

if df_users is None:
    st.error(f"ERRO CRÍTICO: Falha ao carregar arquivos. Detalhes: {df_estab_info}")
    st.stop()
else:
    st.success("Arquivos carregados com sucesso!")

# ==============================================================================
# 2. PRÉ-PROCESSAMENTO
# ==============================================================================

# Filtra apenas pedidos entregues
df_pedidos_validos = df_pedidos[df_pedidos['status_pedido'] == 'ENTREGUE'].copy()

# Divisão Treino/Teste
train_data, test_data = train_test_split(df_pedidos_validos, test_size=0.2, random_state=42)

st.write(f"**Dados divididos:** {len(train_data)} pedidos para treino, {len(test_data)} para teste.")

# ==============================================================================
# 3. PREPARAÇÃO DA MATRIZ (SEM CÁLCULO PESADO IMEDIATO)
# ==============================================================================

st.info("Preparando matrizes esparsas...")

# Cria matriz User-Item (Linhas=Usuários, Colunas=Estabelecimentos)
# Usamos crosstab aqui, mas em bases muito grandes recomenda-se criar a csr_matrix diretamente das coordenadas
train_user_item = pd.crosstab(train_data['usuario_id'], train_data['estabelecimento_id'])
train_sparse = csr_matrix(train_user_item.values)

# OTIMIZAÇÃO CRÍTICA: Transposta para calcular similaridade entre ITENS (Estabelecimentos)
# Isso é necessário para Item-Item CF. 
item_user_matrix = train_sparse.T 

# NÃO calculamos "cosine_similarity(train_sparse)" globalmente aqui.
# Isso criaria uma matriz densa gigante que derruba o servidor (Erro EOF).

# ==============================================================================
# 4. FUNÇÃO DE RECOMENDAÇÃO (CÁLCULO SOB DEMANDA)
# ==============================================================================

def get_recs_item_item(user_id, k=5):
    """
    Gera recomendações baseadas em Item-Item Similarity.
    Calcula similaridade apenas para os itens que o usuário interagiu, economizando RAM.
    """
    # Cold Start
    if user_id not in train_user_item.index:
        return train_data['estabelecimento_id'].value_counts().head(k).index.tolist()
    
    # 1. Pega o histórico do usuário (Vetor de 0s e 1s)
    user_idx = train_user_item.index.get_loc(user_id)
    user_vector = train_sparse[user_idx, :].toarray().flatten()
    
    # 2. Identifica itens que o usuário já interagiu (índices das colunas)
    interacted_items_indices = np.where(user_vector > 0)[0]
    
    scores = {}
    
    # 3. Calcula similaridade APENAS para os itens relevantes
    # Em vez de computar uma matriz N x N gigante, calculamos pontualmente
    if len(interacted_items_indices) > 0:
        # Calcula similaridade entre todos os itens e os itens que o usuário gostou
        # Isso ainda pode ser pesado, então usamos o scikit-learn de forma vetorizada se possível
        # Para economizar memória, limitamos a lógica simplificada:
        
        # Computa a similaridade de cosseno entre a matriz de itens e os itens alvo
        # item_user_matrix é (N_items x N_users)
        # sub_matrix é (N_interacted x N_users)
        sub_matrix = item_user_matrix[interacted_items_indices]
        
        # Similaridade (N_interacted x N_items) - Muito menor que (N_items x N_items)
        sim_subset = cosine_similarity(sub_matrix, item_user_matrix)
        
        # Soma as similaridades ponderadas pelo histórico (aqui histórico é 1)
        # axis=0 soma as colunas, resultando em um score para cada item
        summed_scores = sim_subset.sum(axis=0)
        
        # Cria série para facilitar ordenação
        scores_series = pd.Series(summed_scores, index=train_user_item.columns)
        
        # Remove itens já vistos
        items_seen = train_user_item.columns[interacted_items_indices]
        scores_series = scores_series.drop(items_seen, errors='ignore')
        
        return scores_series.sort_values(ascending=False).head(k).index.tolist()
    
    return []

# ==============================================================================
# 5. AVALIAÇÃO DE DESEMPENHO
# ==============================================================================

if st.button("Executar Avaliação de Desempenho"):
    # Reduzimos a amostra para garantir que rode na memória da nuvem
    sample_size = 100 
    
    st.warning(f"Executando avaliação em uma amostra de {sample_size} usuários para evitar sobrecarga de memória...")
    
    with st.spinner("Calculando métricas..."):
        k_list = [3, 5, 10]
        results = []
        test_users = test_data['usuario_id'].unique()

        if len(test_users) > sample_size:
            test_users = np.random.choice(test_users, sample_size, replace=False)

        global_precision = {k: [] for k in k_list}
        global_recall = {k: [] for k in k_list}

        progress_bar = st.progress(0)
        
        for i, user in enumerate(test_users):
            itens_reais = test_data[test_data['usuario_id'] == user]['estabelecimento_id'].unique()
            
            # Chama a nova função otimizada
            recs = get_recs_item_item(user, k=max(k_list))
            
            for k in k_list:
                recs_k = recs[:k]
                acertos = len(set(recs_k) & set(itens_reais))
                precision = acertos / k
                recall = acertos / len(itens_reais) if len(itens_reais) > 0 else 0
                global_precision[k].append(precision)
                global_recall[k].append(recall)
            
            # Atualiza barra de progresso
            progress_bar.progress((i + 1) / len(test_users))

        for k in k_list:
            results.append({
                'Top N': k, 
                'Precision': np.mean(global_precision[k]), 
                'Recall': np.mean(global_recall[k])
            })

        df_results = pd.DataFrame(results)
        
        st.subheader("Performance do Modelo")
        st.dataframe(df_results)

        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df_results))
        width = 0.35

        ax.bar(x - width/2, df_results['Precision'], width, label='Precision', color='#4285F4')
        ax.bar(x + width/2, df_results['Recall'], width, label='Recall', color='#34A853')

        ax.set_xlabel('Top N')
        ax.set_ylabel('Score')
        ax.set_title('Precisão vs Recall')
        ax.set_xticks(x)
        ax.set_xticklabels(df_results['Top N'])
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        st.pyplot(fig)

        # Exemplo Prático
        st.write("---")
        if len(test_users) > 0:
            exemplo_user = test_users[0]
            recs_ids = get_recs_item_item(exemplo_user, k=3)
            
            # Busca nomes
            nomes_recs = df_estab_info[df_estab_info['estabelecimento_id'].isin(recs_ids)]['categoria_estabelecimento'].tolist()
            
            st.write(f"Exemplo: Para o usuário **{exemplo_user}**, o modelo sugere: **{nomes_recs}**")
