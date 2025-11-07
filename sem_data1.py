import pandas as pd
import numpy as np
from scipy.stats import skewnorm

# ======================
# 1. 生成潜变量得分 (Latent Scores)
# ======================
np.random.seed(42)
n = 438  # 样本量

# 生成基础潜变量（正态分布）
latent_scores = pd.DataFrame({
    'A': skewnorm.rvs(-1, loc=3.8, scale=0.8, size=n),  # 轻度左偏
    'B': skewnorm.rvs(1, loc=3.2, scale=1.0, size=n),   # 轻度右偏
    'C': np.random.normal(3.5, 0.9, n),
    'D': skewnorm.rvs(0.5, loc=3.1, scale=1.1, size=n),
    'M': np.random.normal(4.0, 0.7, n)
})

# ======================
# 2. 生成测量项 (指标)
# ======================
def create_items(latent, loadings, noise_level=0.2):
    """创建带噪声的观测项"""
    items = {}
    for i, loading in enumerate(loadings, 1):
        base = latent * loading
        noise = np.random.normal(0, noise_level*abs(base).std(), n)
        items[f'item{i}'] = np.clip(base + noise, 1, 5)
    return pd.DataFrame(items)

# 自变量ABCD (因子载荷0.6-0.8)
data = pd.DataFrame()
for var in ['A','B','C','D']:
    df = create_items(latent_scores[var], 
                    loadings=[0.7, 0.75, 0.65], 
                    noise_level=0.25)
    data = pd.concat([data, df.add_prefix(f'{var}')], axis=1)

# 调节变量M (4个指标)
m_loadings = [0.68, 0.72, 0.65, 0.7]
m_items = create_items(latent_scores['M'], m_loadings)
data = pd.concat([data, m_items.add_prefix('M')], axis=1)

# ======================
# 3. 构建结构模型
# ======================
# 中介变量E的生成 (受ABCD影响)
E = (
    0.3*latent_scores['A'] + 
    0.25*latent_scores['B'] + 
    0.2*latent_scores['C'] + 
    0.15*latent_scores['D'] + 
    np.random.normal(0, 0.8, n)
)
E += 0.1*(latent_scores['A']*latent_scores['M'])  # 调节效应

# 结果变量F的生成 (受E影响)
F = 0.6*E + np.random.normal(0, 0.7, n)

# 添加E和F的测量项
e_items = create_items(E, [0.75, 0.8, 0.72], noise_level=0.2)
f_items = create_items(F, [0.82, 0.78, 0.75], noise_level=0.18)
data = pd.concat([data, e_items.add_prefix('E'), f_items.add_prefix('F')], axis=1)

# ======================
# 4. 数据后处理
# ======================
def post_process(x):
    """添加人类响应特征"""
    x = np.round(x).astype(int)
    x = np.clip(x, 1, 5)
    
    # 添加5%的极端响应
    mask = np.random.choice([0,1], size=len(x), p=[0.95,0.05])
    x = np.where(mask, np.random.choice([1,5], len(x)), x)
    
    # 添加10%的中间响应
    mask = np.random.choice([0,1], size=len(x), p=[0.9,0.1])
    x = np.where(mask, 3, x)
    
    return x

data = data.apply(post_process)

# ======================
# 5. 保存数据
# ======================
data.to_csv("pls_sem_simulation_data.csv", index=False)
print("数据生成完成！文件已保存为 pls_sem_simulation_data.csv")
