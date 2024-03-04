import torch

def get_closest_token_cosine_similarities(single_weight, all_weights, return_scores=False):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    scores = cos(all_weights, single_weight.unsqueeze(0).to(all_weights.device))
    sorted_scores, sorted_ids = torch.sort(scores, descending=True)
    best_id_list = sorted_ids.tolist()
    if not return_scores:
        return best_id_list
    scores_list = sorted_scores.tolist()
    return best_id_list, scores_list

def get_closest_token_euclidean(single_weight, all_weights):
    single_weight = single_weight.to(all_weights.device)
    distances = torch.norm(all_weights - single_weight.unsqueeze(0), dim=1, p=2)
    sorted_distances, sorted_ids = torch.sort(distances)
    best_id_list = sorted_ids.tolist()
    return best_id_list

def get_closest_token_manhattan(single_weight, all_weights):
    single_weight = single_weight.to(all_weights.device)
    distances = torch.norm(all_weights - single_weight.unsqueeze(0), dim=1, p=1)
    sorted_distances, sorted_ids = torch.sort(distances)
    best_id_list = sorted_ids.tolist()
    return best_id_list

def get_closest_token_jaccard_similarity(single_weight, all_weights, return_scores=False):
    single_weight = single_weight.to(all_weights.device)
    binary_single_weight = (single_weight > 0).float()
    binary_all_weights = (all_weights > 0).float()
    
    # Calculate Jaccard similarity as intersection over union.
    intersection = torch.min(binary_single_weight, binary_all_weights).sum(dim=1)
    union = torch.max(binary_single_weight, binary_all_weights).sum(dim=1)
    jaccard_similarity = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    sorted_similarities, sorted_ids = torch.sort(jaccard_similarity, descending=True)
    best_id_list = sorted_ids.tolist()
    if not return_scores:
        return best_id_list
    scores_list = sorted_similarities.tolist()
    return best_id_list, scores_list

def get_closest_token_mahalanobis(single_weight, all_weights):
    single_weight = single_weight.to(all_weights.device)
    distances = torch.norm(all_weights - single_weight.unsqueeze(0), dim=1, p=2)
    sorted_distances, sorted_ids = torch.sort(distances)
    best_id_list = sorted_ids.tolist()
    return best_id_list

def get_closest_token_hamming(single_weight, all_weights):
    single_weight = single_weight.to(all_weights.device)
    # Convert weights to binary
    binary_single_weight = (single_weight > 0).int()
    binary_all_weights = (all_weights > 0).int()
    
    # Calculate Hamming distance
    differences = torch.abs(binary_single_weight - binary_all_weights).sum(dim=1)
    sorted_differences, sorted_ids = torch.sort(differences)
    best_id_list = sorted_ids.tolist()
    return best_id_list

correlation_functions = {
    "cosine_similarities":get_closest_token_cosine_similarities,
    "euclidean": get_closest_token_euclidean,
    "manhattan": get_closest_token_manhattan,
    "jaccard": get_closest_token_jaccard_similarity,
    "mahalanobis": get_closest_token_mahalanobis,
    "hamming": get_closest_token_hamming
}
