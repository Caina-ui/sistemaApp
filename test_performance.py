import pandas as pd
import numpy as np
import time
import psutil
import os
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import gc

def print_memory_usage(step_name):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"💾 [{step_name}] Uso de RAM: {mem_info.rss / 1024 / 1024:.2f} MB")

print("="*60)
print("🚀 INICIANDO TESTE DE PERFORMANCE E CARGA")
print("="*60)
print_memory_usage("Início")

# ---------------------------------------------------------
# 1. CARREGAMENTO
# ---------------------------------------------------------
start_time = time.time()
try:
    df_users = pd.read_parquet('base_usuarios.parquet')
    df_estab = pd.read_parquet('base_estabelecimentos.parquet')
    df_pedidos = pd.read_parquet('base_pedidos.parquet')
    print(f"✅ Arquivos carregados em {time.time() - start_time:.2f}s")
except FileNotFoundError:
    print("❌ ERRO: Arquivos .parquet não encontrados. Rode o script de conversão primeiro.")
    exit()

print_memory_usage("Após Load")

# ---------------------------------------------------------
# 2. PRÉ-PROCESSAMENTO
# ---------------------------------------------------------
df_pedidos_validos = df_pedidos[df_pedidos['status_pedido'] == 'ENTREGUE'].copy()
train_data, _ = train_test_split(df_pedidos_validos, test_size=0.2, random_state=42)
print(f"📊 Linhas de treino: {len(train_data)}")

# ---------------------------------------------------------
# 3. SIMULAÇÃO DA COMPILAÇÃO DA MATRIZ (A parte crítica)
# ---------------------------------------------------------
print("\n--- 🔨 Construindo Matriz (Otimizada) ---")
t0_build = time.time()

# Lógica idêntica ao seu codigosus.py otimizado
try:
    # Crosstab
    train_user_item = pd.crosstab(train_data['usuario_id'], train_data['estabelecimento_id'])
    
    # CSR Matrix com FLOAT32 (Otimização chave)
    train_sparse = csr_matrix(train_user_item.values, dtype=np.float32)
    
    # Similaridade Item-Item (Transposta)
    # Aqui é onde o pico de memória acontece
    item_sim_matrix = cosine_similarity(train_sparse.T)
    
    # Mapeamentos
    estab_ids = train_user_item.columns
    estab_to_idx = {estab_id: i for i, estab_id in enumerate(estab_ids)}
    idx_to_estab = {i: estab_id for i, estab_id in enumerate(estab_ids)}
    
    # Força limpeza do que não é mais necessário
    del train_sparse
    gc.collect()
    
    print(f"✅ Matriz construída em {time.time() - t0_build:.2f}s")
    print(f"📐 Dimensão da Matriz: {item_sim_matrix.shape}")
    print(f"🔢 Tipo de dado da Matriz: {item_sim_matrix.dtype}") # Deve ser float32

except Exception as e:
    print(f"❌ ERRO CRÍTICO DE MEMÓRIA/CÁLCULO: {e}")
    exit()

print_memory_usage("Após Matriz Completa")

# ---------------------------------------------------------
# 4. TESTE DE RECOMENDAÇÃO (Velocidade de resposta)
# ---------------------------------------------------------
print("\n--- ⚡ Testando Velocidade de Recomendação ---")

# Função otimizada (cópia do seu código)
def get_recs_fast_test(user_id, k=5):
    if user_id not in train_user_item.index:
        return []
    
    user_history = train_user_item.loc[user_id]
    interacted_estabs = user_history[user_history > 0].index.tolist()
    
    if not interacted_estabs: return []

    total_scores = np.zeros(item_sim_matrix.shape[0], dtype=np.float32)
    
    for estab_id in interacted_estabs:
        if estab_id in estab_to_idx:
            idx = estab_to_idx[estab_id]
            total_scores += item_sim_matrix[idx]

    for estab_id in interacted_estabs:
        if estab_id in estab_to_idx:
            total_scores[estab_to_idx[estab_id]] = -1

    if k >= len(total_scores):
        top_indices = np.argsort(total_scores)[::-1]
    else:
        top_indices = np.argpartition(total_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(total_scores[top_indices])[::-1]]
    
    return [idx_to_estab[i] for i in top_indices if total_scores[i] > 0]

# Pega um usuário real para testar
test_user = train_data['usuario_id'].iloc[0]
print(f"👤 Testando para usuário: {test_user}")

t0_rec = time.time()
recs = get_recs_fast_test(test_user, k=5)
t_end_rec = time.time()

print(f"✅ Recomendações geradas: {recs}")
print(f"⏱️ Tempo de inferência: {(t_end_rec - t0_rec) * 1000:.2f} ms") # em milissegundos

print("\n" + "="*60)
print("🏁 RESULTADO DO DIAGNÓSTICO")
print("="*60)

process = psutil.Process(os.getpid())
final_ram = process.memory_info().rss / 1024 / 1024

if final_ram > 800:
    print(f"⚠️ PERIGO: Uso de RAM ({final_ram:.2f} MB) está próximo do limite de 1GB do Streamlit Cloud.")
    print("   Sugestão: Reduza o tamanho da base de pedidos (ex: filtre apenas o último ano).")
else:
    print(f"🟢 SUCESSO: Uso de RAM ({final_ram:.2f} MB) está seguro para o plano gratuito.")
