#imports
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

epsilon_total = 1.0

epsilon_mi   = 0.4   # MI computation
epsilon_corr = 0.4   # correlation computation
# remaining 0.2 can be used for elbow selection (EM)

def dp_mutual_information(mi_values, epsilon, delta=1e-5):
    """
    mi_values: array of MI scores (already computed)
    """
    sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, size=len(mi_values))
    return mi_values + noise


def mi_for_all(df_name = df, lc = label_column, title = 'complete dataset'):
    y = df_name.iloc[:, lc]
    #print(f'label is {y}')
    x = df_name.drop(df_name.columns[lc], axis=1)
    mi_scores = mutual_info_classif(x, y, discrete_features=True)
    mi_scores= mi_scores / np.log(2)

    mi_df = pd.DataFrame({"Feature": x.columns, "MI_Score": mi_scores}).sort_values(by="MI_Score", ascending=False)

    return x,y,mi_df

x,y,mi_df= mi_for_all()


parts = np.array_split(np.arange(df.shape[1]), number_of_sets, axis=0)
for i, part in enumerate(parts):
    print(f"Part {i}: {part}")

target_set = None

# Find which part has the label column and get that part
for part in parts:
    if label_column in part:
        target_set = df.iloc[:, part]
        break

if target_set is not None:
    print(f"Target set shape: {target_set.shape}")
else:
    print(f"Label column {label_column} not found in any part")


#make label first column on target_set and the last column will be the lowst mi on target_set
label_col = df.columns[label_column]
col_data = target_set.pop(label_col)  # Remove the column
target_set.insert(0, label_col, col_data)  # Insert it at position 0


# Get MI scores and find lowest MI column
a,b,target_mi_df = mi_for_all(df_name=target_set, lc=0, title = 'in target set')

lowest_mi_column = target_mi_df.iloc[-1]['Feature']
print(f"Column with lowest MI: {lowest_mi_column}")

# Calculate Spearman correlations
lowest_mi_data = target_set[lowest_mi_column]
print(f"lowest_mi_data:{lowest_mi_data}")

#x_ranked = x.rank(method='average').astype(int)
#lowest_mi_ranked = lowest_mi_data.rank(method='average').astype(int)

correlations = x.corrwith(lowest_mi_data, method='spearman')

# Convert to DataFrame and sort
corr_df = pd.DataFrame({
    'Feature': correlations.index,
    'Spearman_Correlation': correlations.abs().values
}).sort_values('Spearman_Correlation', ascending=False)

# Remove the column that correlates with itself
corr_df = corr_df[corr_df['Feature'] != lowest_mi_column]



import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

# =========================================================
# 1. Privacy budget configuration
# =========================================================

EPSILON_TOTAL = 1.0
DELTA = 1e-5

EPSILON_MI = 0.5
EPSILON_CORR = 0.5


# =========================================================
# 2. Utility functions
# =========================================================

def gaussian_noise(scale, size):
    return np.random.normal(0, scale, size=size)


def gaussian_sigma(epsilon, delta):
    return np.sqrt(2 * np.log(1.25 / delta)) / epsilon


# =========================================================
# 3. DP Mutual Information
# =========================================================

def dp_mutual_information(X, y, epsilon, delta, number_of_sets):
    """
    Computes DP Mutual Information using Gaussian mechanism
    """
    # Raw MI (never shared)
    mi_raw = mutual_info_classif(
        X,
        y,
        discrete_features="auto",
        n_neighbors=number_of_sets,
        random_state=42
    )

    # Gaussian noise
    sigma = gaussian_sigma(epsilon, delta)
    mi_dp = mi_raw + gaussian_noise(sigma, size=len(mi_raw))

    return mi_dp


# =========================================================
# 4. DP Spearman Correlation
# =========================================================

def dp_spearman_correlations(X, target_feature, epsilon, delta):
    """
    Computes DP Spearman correlations to the target feature
    """
    corr_raw = []

    target_values = X[:, target_feature]

    for j in range(X.shape[1]):
        corr, _ = spearmanr(X[:, j], target_values)
        if np.isnan(corr):
            corr = 0.0
        corr_raw.append(corr)

    corr_raw = np.array(corr_raw)

    # Gaussian noise (Spearman corr ∈ [-1, 1])
    sigma = gaussian_sigma(epsilon, delta)
    corr_dp = corr_raw + gaussian_noise(sigma, size=len(corr_raw))

    # Optional clipping for stability (post-processing)
    corr_dp = np.clip(corr_dp, -1.0, 1.0)

    return corr_dp


# =========================================================
# 5. MAIN PIPELINE
# =========================================================

def dp_feature_selection_pipeline(
    df,
    X,
    y,
    label_column,
    number_of_sets
):
    """
    Returns corr_df with DP-noised correlations
    """

    # -----------------------------------------
    # Feature names
    # -----------------------------------------
    feature_names = [
        col for i, col in enumerate(df.columns)
        if i != label_column
    ]

    # -----------------------------------------
    # Step 1: DP Mutual Information
    # -----------------------------------------
    mi_dp = dp_mutual_information(
        X,
        y,
        epsilon=EPSILON_MI,
        delta=DELTA,
        number_of_sets=number_of_sets
    )

    # -----------------------------------------
    # Step 2: Select target feature (DP-safe)
    # -----------------------------------------
    target_feature_idx = int(np.argmin(mi_dp))

    # -----------------------------------------
    # Step 3: DP Spearman correlations
    # -----------------------------------------
    corr_dp = dp_spearman_correlations(
        X,
        target_feature=target_feature_idx,
        epsilon=EPSILON_CORR,
        delta=DELTA
    )

    # -----------------------------------------
    # Step 4: Build corr_df (DP output)
    # -----------------------------------------
    corr_df = pd.DataFrame({
        "Feature": feature_names,
        "Spearman_Correlation": corr_dp
    })

    corr_df = corr_df.sort_values(
        "Spearman_Correlation",
        ascending=False
    ).reset_index(drop=True)

    return corr_df


# =========================================================
# 6. RUN
# =========================================================

corr_df = dp_feature_selection_pipeline(
    df=df,
    X=X,
    y=y,
    label_column=label_column,
    number_of_sets=number_of_sets
)

corr_df
