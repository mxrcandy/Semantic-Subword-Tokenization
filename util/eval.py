from typing import List
import numpy as np

def compute_hr_at_k(predictions: List[List[str]], ground_truth: List[List[str]], k: List[int]) -> dict:
    """
    计算在不同k值下的Hit Rate (HR)。
    
    :param predictions: 模型预测的推荐列表，每个元素是一个用户的推荐列表。
    :param ground_truth: 每个用户的真实标签集合。
    :param k: 一个整数列表，表示要计算HR的k值。
    :return: 在所有k值下的平均HR。
    """
    hr_at_k = []
    
    for k_value in k:
        hits = 0
        total_users = len(predictions)
        
        for user_predictions, user_ground_truth in zip(predictions, ground_truth):
            if len(user_predictions) >= k_value:
                top_k_predictions = set(user_predictions[:k_value])
                if top_k_predictions & set(user_ground_truth):
                    hits += 1
            else:
                raise ValueError(f"Predictions length {len(user_predictions)} is less than k value {k_value}.")
        
        hr_at_k.append(hits / total_users if total_users > 0 else 0.0)

    metrics = {f"HR@{k_value}": hr_at_k[idx] for idx, k_value in enumerate(k)}
    return metrics

def compute_ndcg_at_k(predictions: List[List[str]], ground_truth: List[List[str]], k: List[int]) -> dict:
    """
    计算在不同k值下的Normalized Discounted Cumulative Gain (NDCG)。
    
    :param predictions: 模型预测的推荐列表，每个元素是一个用户的推荐列表。
    :param ground_truth: 每个用户的真实标签集合。
    :param k: 一个整数列表，表示要计算NDCG的k值。
    :return: 在所有k值下的平均NDCG。
    """
    ndcg_at_k = []
    
    for k_value in k:
        total_ndcg = 0.0
        total_users = len(predictions)
        
        for user_predictions, user_ground_truth in zip(predictions, ground_truth):
            if len(user_predictions) >= k_value:
                top_k_predictions = user_predictions[:k_value]
                dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k_predictions) if item in set(user_ground_truth))
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k_value, len(user_ground_truth))))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                total_ndcg += ndcg
            else:
                raise ValueError(f"Predictions length {len(user_predictions)} is less than k value {k_value}.")
        
        ndcg_at_k.append(total_ndcg / total_users if total_users > 0 else 0.0)

    metrics = {f"NDCG@{k_value}": ndcg_at_k[idx] for idx, k_value in enumerate(k)}
    return metrics