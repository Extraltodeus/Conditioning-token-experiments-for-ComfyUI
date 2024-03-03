import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import comfy.model_management as model_management

def linear_regression(tensors):
    tensors = tensors.to(device=model_management.get_torch_device())
    X = torch.arange(tensors.size(0)).unsqueeze(1).float().to(device=model_management.get_torch_device())
    X = torch.cat([X, torch.ones(X.size(0), 1, device=model_management.get_torch_device())], dim=1)  # Ensure constant term is also on GPU
    Y = tensors.view(tensors.size(0), -1).float().to(device=model_management.get_torch_device())  # Ensure Y is on GPU
    beta = torch.linalg.inv(X.T @ X) @ X.T @ Y
    next_X = torch.tensor([[tensors.size(0), 1.0]], device=model_management.get_torch_device())
    return (next_X @ beta).view_as(tensors[0])


def autoregression(tensors):
    series = tensors.view(tensors.size(0), -1).float()
    predicted_next = torch.zeros_like(series[0])
    for i in range(series.size(1)):
        X = series[:-1, i:i+1]
        Y = series[1:, i]
        beta = torch.linalg.solve(X.T @ X + 0.0001 * torch.eye(1, device=tensors.device), X.T @ Y.unsqueeze(1))
        predicted_next[i] = series[-1, i:i+1] @ beta
    return predicted_next.view_as(tensors[0])

def random_forest(tensors):
    tensors=tensors.to(device='cpu')
    X = np.arange(tensors.size(0)).reshape(-1, 1)
    Y = tensors.view(tensors.size(0), -1).numpy()
    predicted_tensors = []
    for i in range(Y.shape[1]):
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, Y[:, i])
        preds = model.predict(np.array([[tensors.size(0)]]))
        predicted_tensors.append(preds)
    return torch.tensor(np.column_stack(predicted_tensors)).view_as(tensors[0])

# def weighted_moving_average(tensors, weights=None):
#     if weights is None:
#         weights = torch.linspace(1, 2, steps=tensors.shape[0]).to(device=tensors.device)
#     weights = weights / weights.sum()
#     return torch.sum(tensors * weights.view(-1, 1, 1), dim=0)
def weighted_moving_average(tensors, weights=None):
    if weights is None:
        weights = torch.linspace(1, 2, steps=tensors.size(0), device=tensors.device)
    normalized_weights = weights / weights.sum()
    weighted_sum = torch.sum(tensors * normalized_weights.view(-1, 1, 1, 1), dim=0, keepdim=True)
    return weighted_sum[0]

def principal_component_analysis(tensors, n_components=1):
    tensors=tensors.to(device='cpu')
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)
    data = tensors.view(tensors.shape[0], -1).numpy()
    scaled_data = scaler.fit_transform(data)
    pca.fit(scaled_data)
    transformed = pca.transform(scaled_data)
    inverse_transform = pca.inverse_transform(np.append(transformed, [transformed[-1]], axis=0))
    last_tensor = scaler.inverse_transform(inverse_transform)[-1]
    return torch.tensor(last_tensor).view(tensors[0].shape)

def exponential_smoothing(tensors, alpha=0.5):
    smoothed = torch.zeros_like(tensors)
    smoothed[0] = tensors[0]
    for i in range(1, tensors.shape[0]):
        smoothed[i] = alpha * tensors[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed[-1]

def polynomial_regression(tensors, degree=2):
    x = torch.arange(0, tensors.shape[0], device=tensors.device).unsqueeze(-1).float()
    X = torch.pow(x, torch.arange(degree, -1, -1, device=tensors.device).float())
    Y = tensors.view(tensors.shape[0], -1).float()
    solution = torch.linalg.lstsq(X, Y).solution
    new_x = torch.tensor([tensors.shape[0]], device=tensors.device).float()
    new_X = torch.pow(new_x, torch.arange(degree, -1, -1, device=tensors.device).float())
    return torch.matmul(new_X, solution).view(tensors[0].shape)

# 'autoregression': autoregression,
# 'random_forest': random_forest,

methods = {
    'linear_regression': linear_regression,
    'weighted_moving_average': weighted_moving_average,
    'principal_component_analysis': principal_component_analysis,
    'exponential_smoothing': exponential_smoothing,
    'polynomial_regression': polynomial_regression,
}

def dispatch_tensors(tensors, method):
    if method in methods:
        return methods[method](tensors)
    else:
        raise ValueError(f"Unknown method: {method}")

# random_tensors = torch.stack([torch.randn(1, 77, 2048) for _ in range(12)]).to(device=model_management.get_torch_device())
# for method in list(methods.keys()):
#     print(method)
#     result = dispatch_tensors(random_tensors, method)
#     print(f'Result for {method}:', result.shape)
# input()