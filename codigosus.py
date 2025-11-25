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
# 1. CARREGAMENTO DOS DADOS (AGORA TUDO EM PARQUET)
# ==============================================================================
@st.cache_data
def load_data():
    try:
        # LENDO TODOS OS ARQUIVOS EM FORMATO PARQUET
        df_users = pd.read_parquet('base_usuarios.parquet')
        df_estab = pd.read_parquet('base_estabelecimentos.parquet')
        df_pedidos = pd.read_parquet('base_pedidos.parquet')
        
        return df_users, df_pedidos, df_estab
    except FileNotFoundError as e:
        return None, None, e

st.write("--- Iniciando Processamento ---")
# df_estab_info contém os dados dos estabelecimentos
df_users, df_pedidos, df_estab_info = load_data()

# Verificação de erro no carregamento
if df_users is None:
    st.error(f"ERRO CRÍTICO: Um dos arquivos Parquet não foi encontrado: {df_estab_info}")
    st.warning("Verifique se você converteu e fez o upload dos arquivos **base_usuarios.parquet**, **base_estabelecimentos.parquet** e **base_pedidos.parquet**.")
    st.stop()
else:
    st.success("Arquivos carregados com sucesso! (Todos em Parquet)")

# ==============================================================================
# 2. PRÉ-PROCESSAMENTO
# ==============================================================================

# Filtra apenas pedidos entregues
df_pedidos_validos = df_pedidos[df_pedidos['status_pedido'] == 'ENTREGUE'].copy()

# Divisão Treino (80%) e Teste (20%)
train_data, test_data = train_test_split(df_pedidos_validos, test_size=0.2, random_state=42)

st.write(f"**Dados divididos:** {len(train_data)} pedidos para treino, {len(test_data)} para teste.")

# ==============================================================================
# 3. TREINAMENTO DO MODELO
# ==============================================================================
st.info("Construindo matriz de similaridade...")

try:
    # Cria matriz User-Item apenas com dados de TREINO
    train_user_item = pd.crosstab(train_data['usuario_id'], train_data['estabelecimento_id'])
    train_sparse = csr_matrix(train_user_item.values)

    # Calcula similaridade
    train_similarity = cosine_similarity(train_sparse)
    
    # OTIMIZAÇÃO DE MEMÓRIA:
    # Não converter para DataFrame denso imediatamente se for muito grande
    # Mantemos em numpy array para economizar RAM
    df_train_sim = pd.DataFrame(train_similarity, index=train_user_item.index, columns=train_user_item.index)

except Exception as e:
    st.error(f"Erro de Memória ou Cálculo: {e}")
    st.warning("A base de dados pode ser grande demais para este ambiente gratuito.")
    st.stop()
# ==============================================================================
# 4. FUNÇÃO DE RECOMENDAÇÃO
# ==============================================================================

def get_recs(user_id, k=5):
    """Gera recomendações baseadas no histórico de treino"""
    # Cold Start: Se o usuário não existe no treino, recomenda os mais populares
    if user_id not in train_user_item.index:
        top_pop = train_data['estabelecimento_id'].value_counts().head(k).index.tolist()
        return top_pop
    
    # Lógica de similaridade ponderada
    user_history = train_user_item.loc[user_id]
    
    estabs_comprados = user_history[user_history > 0].index.tolist()
    scores = pd.Series(dtype=float)
    
    for estab_id in estabs_comprados:
        if estab_id in df_train_sim.columns:
            sim_scores = df_train_sim[estab_id]
            weighted_scores = sim_scores * user_history[estab_id]
            scores = scores.add(weighted_scores, fill_value=0)
            
    scores = scores.drop(estabs_comprados, errors='ignore')
    return scores.sort_values(ascending=False).head(k).index.tolist()

# ==============================================================================
# 5. AVALIAÇÃO DE DESEMPENHO E VISUALIZAÇÃO
# ==============================================================================

if st.button("Executar Avaliação de Desempenho"):
    with st.spinner("Calculando métricas (isso pode levar alguns segundos)..."):
        k_list = [3, 5, 10]
        results = []
        test_users = test_data['usuario_id'].unique()

        # Amostra para performance se a base for muito grande
        if len(test_users) > 500:
            test_users = np.random.choice(test_users, 500, replace=False)

        global_precision = {k: [] for k in k_list}
        global_recall = {k: [] for k in k_list}

        for user in test_users:
            itens_reais = test_data[test_data['usuario_id'] == user]['estabelecimento_id'].unique()
            recs = get_recs(user, k=max(k_list))
            
            for k in k_list:
                recs_k = recs[:k]
                acertos = len(set(recs_k) & set(itens_reais))
                precision = acertos / k
                recall = acertos / len(itens_reais) if len(itens_reais) > 0 else 0
                global_precision[k].append(precision)
                global_recall[k].append(recall)

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
            recs_ids = get_recs(exemplo_user, k=3)
            # Verifica se o ID existe na base de estabelecimentos antes de buscar o nome
            nomes_recs = df_estab_info[df_estab_info['estabelecimento_id'].isin(recs_ids)]['categoria_estabelecimento'].tolist()
            
            st.write(f"Exemplo: Para o usuário **{exemplo_user}**, o modelo sugere: **{nomes_recs}**")
