corr_df = corr_df.reset_index(drop=True)
corr_df['diffrence'] = corr_df['Spearman_Correlation'].diff(1)
corr_df['change'] = corr_df['diffrence'].diff(1)
print(corr_df)
print(np.argmin(corr_df['change']))

valid_df = corr_df[corr_df["change"].notna()]
n = len(valid_df)
start_idx = int(n * 0.10)
end_idx = int(n * 0.50)

elbow_index = (valid_df.iloc[start_idx:end_idx]["change"].idxmin()) -1
print(elbow_index)

def exponential_mechanism_position(corr_df, elbow_index, epsilon):
    d = len(corr_df)
    K = d // 5

    indices = np.arange(len(corr_df))
    mask = np.abs(indices - elbow_index) <= K
    candidate_indices = indices[mask]

    utilities = -np.abs(candidate_indices - elbow_index)
    
    # Exponential mechanism probabilities
    scores = np.exp((epsilon * utilities) / 2)
    probabilities = scores / scores.sum()

    sampled_index = np.random.choice(candidate_indices, p=probabilities)

    return sampled_index, probabilities, candidate_indices

epsilon = 1.0 # privacy budget for elbow selection

dp_elbow, probs, candidate_indices = exponential_mechanism_position( corr_df, elbow_index, epsilon )

print("DP elbow index:", dp_elbow)
print("Selected feature:", corr_df.loc[dp_elbow, "Feature"])


prob_df = corr_df.copy()
prob_df["probability"] = 0.0 # initialize all to zero
prob_df.iloc[candidate_indices, prob_df.columns.get_loc("probability")] = probs
prob_df["distance_to_elbow"] = abs(prob_df.index - elbow_index)

prob_df.sort_values("probability", ascending=False)