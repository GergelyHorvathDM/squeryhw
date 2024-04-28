from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold


def compute_acc_rec_prec_f1_with_cv(model, X, y, cv_n=10):
    metrics = {"accuracy": accuracy_score, "recall": recall_score, "precision": precision_score, "f1": f1_score}

    means = {}
    sdevs = {}

    cv = StratifiedKFold(n_splits=cv_n)

    for metric in metrics:
        scorer = make_scorer(metrics[metric])
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
        means[metric] = scores.mean()*100
        sdevs[metric] = scores.std()*100
        print(f"\t{metric.capitalize()} score:", scores.mean()*100, "+/-", scores.std()*100)

    return means, sdevs