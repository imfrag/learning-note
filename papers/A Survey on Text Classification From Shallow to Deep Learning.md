## Evaluation Metrics

### Single-label metrics

1.**Accuracy and Error Rate**. Accuracy and Error Rate are the fundamental metrics for a classification model.
$$
Accuracy=\frac{TP+TN}{N}, ErrorRate = 1 - Accuracy = \frac{FP + FN}{N}
$$
TP: True Positive; TN: True Negative; FP: False Positive; FN: False Negative.

2.**Precision, Recall and F1**.These are vital metrics utilized for ==unbalanced== test sets regardless of the standard type and error rate.
$$
Precision=\frac{TP}{TP+FP}, Recall = \frac{TP}{TP+FN}, \frac{1}{F1}=\frac{1}{2}(\frac{1}{Precision} + \frac{1}{Recall})
$$
For the multi-class classification problem, the precision and recall value of each class can be calculated separately, and then the performance of the individual an whole can be analyzed.

3.**Exact Match (EM)**. The EM is a metric for QA tasks measuring the prediction that matches all the ground-truth answers precisely.

4.**Mean Reciprocal Rank (MRR)**. The MRR is usually applied for assessing the performance of ranking algorithms on QA and Information Retrieval (IR) tasks.
$$
MRR = \frac{1}{Q}\sum_{i=1}^{Q}{\frac{1}{rank_i}}
$$
5.**Hamming-loss (HL)**. The [HL][217] assesses the score of misclassified instance-label pairs where a related label is omitted or an unrelated is predicted.

### Multi-label metrics

1.**Micro - F1**. The [Micro-F1][218] is a measure that considers the overall accuracy and recall of all labels.
$$
Micro-F1=\frac{2P_t\times R_t}{P + R}
$$
where:
$$
P=\frac{\sum_{t\in S}{TP_t}}{\sum_{t\in S}{TP_t + FP_t}} ,R=\frac{\sum_{t\in S}{TP_t}}{\sum_{t\in S}{TP_t + FN_t}}
$$
2.**Precision at Top K (P@K)**. The P@K is the precision at the top k. For P@K, each text has a set of $\scr L$ ground truth labels $L_t=\langle l_0,l_1,l_2 \dots,l_{\scr L - 1} \rangle$, in order of decreasing probability $P_t=[p_0, p_1, p_2 \dots, p_{Q-1}]$. The precision at k is
$$
P@K=\frac1k\sum_{j=0}^{min({\scr L}, k) -1}{rel_{L_i}(P_t(j))}
$$
where
$$
rel_L(p) = \begin{cases}
1, p\in L \\
0, otherwise
\end{cases}
$$
3.**Normalized Discounted Cummulated Gains (NDCG@K)**.
$$
NDCG@K=\frac{1}{IDCG(L_i, k)}\sum_{j=0}^{n-1}\frac{{rel_{L_i}(P_t(j))}}{ln(j + 1)}
$$
where
$$
n=min(max(|P_i|, |L_i|), k)
$$


## References

[ 217 ]: # "Improved boosting algorithms using confidence-rated predictions"
[ 218 ]: # "Introduction to information retrieval"

