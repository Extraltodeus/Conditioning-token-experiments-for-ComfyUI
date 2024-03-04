import torch

def get_closest_token_euclidean(single_weight, all_weights, token_index=0):
    distances = torch.norm(all_weights - single_weight.unsqueeze(0), dim=1, p=2)
    sorted_distances, sorted_ids = torch.sort(distances)
    best_id = sorted_ids.tolist()[token_index]
    return best_id
def get_closest_token_manhattan(single_weight, all_weights, token_index=0):
    distances = torch.norm(all_weights - single_weight.unsqueeze(0), dim=1, p=1)
    sorted_distances, sorted_ids = torch.sort(distances)
    best_id = sorted_ids.tolist()[token_index]
    return best_id
def get_closest_token_dot_product(single_weight, all_weights, token_index=0):
    # Assuming all_weights is a 2D tensor and single_weight is 1D
    similarities = torch.matmul(all_weights, single_weight.unsqueeze(1)).squeeze()
    sorted_similarities, sorted_ids = torch.sort(similarities, descending=True)
    best_id = sorted_ids.tolist()[token_index]
    return best_id

def get_closest_token_pearson_correlation(single_weight, all_weights, token_index=0):
    # Normalize the vectors
    mean_centered_all_weights = all_weights - all_weights.mean(dim=2, keepdim=True)
    mean_centered_single_weight = single_weight - single_weight.mean(dim=1, keepdim=True)
    std_all_weights = mean_centered_all_weights.std(dim=2, keepdim=True)
    std_single_weight = mean_centered_single_weight.std(dim=1, keepdim=True)
    
    # Compute Pearson correlation
    correlations = (torch.bmm(mean_centered_all_weights, mean_centered_single_weight.unsqueeze(2)).squeeze() / 
                    (std_all_weights * std_single_weight * (all_weights.shape[2] - 1)))
    sorted_correlations, sorted_ids = torch.sort(correlations.squeeze(), descending=True)
    best_id = sorted_ids.tolist()[token_index]
    return best_id

def get_closest_token_jaccard_similarity(single_weight, all_weights, token_index=0):
    # Binarize the weights to convert to sets. This might require a threshold.
    # Here, assuming a simple threshold at 0 to create binary vectors.
    binary_single_weight = (single_weight > 0).float()
    binary_all_weights = (all_weights > 0).float()
    
    # Calculate Jaccard similarity as intersection over union.
    intersection = torch.min(binary_single_weight, binary_all_weights).sum(dim=1)
    union = torch.max(binary_single_weight, binary_all_weights).sum(dim=1)
    jaccard_similarity = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    sorted_similarities, sorted_ids = torch.sort(jaccard_similarity, descending=True)
    best_id = sorted_ids.tolist()[token_index]
    return best_id
def get_closest_token_mahalanobis(single_weight, all_weights, token_index=0):
    # Normally, you'd compute a covariance matrix and its inverse, but for simplicity:
    # Assuming identity covariance matrix, the Mahalanobis distance simplifies to Euclidean distance.
    distances = torch.norm(all_weights - single_weight.unsqueeze(0), dim=2, p=2)
    sorted_distances, sorted_ids = torch.sort(distances)
    best_id = sorted_ids.tolist()[token_index]
    return best_id
def get_closest_token_hamming(single_weight, all_weights, token_index=0):
    # Convert weights to binary
    binary_single_weight = (single_weight > 0).int()
    binary_all_weights = (all_weights > 0).int()
    
    # Calculate Hamming distance
    differences = torch.abs(binary_single_weight - binary_all_weights).sum(dim=2)
    sorted_differences, sorted_ids = torch.sort(differences)
    best_id = sorted_ids.tolist()[token_index]
    return best_id

correlation_functions = {
    "euclidean": get_closest_token_euclidean,
    "manhattan": get_closest_token_manhattan,
    "dot_product": get_closest_token_dot_product,
    "pearson_correlation":get_closest_token_pearson_correlation,
    "jaccard_similarity": get_closest_token_jaccard_similarity,
    "mahalanobis": get_closest_token_mahalanobis,
    "hamming": get_closest_token_hamming
}
