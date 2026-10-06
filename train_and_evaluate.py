import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import joblib

def main():
    print("=" * 60)
    print("Step 1: Loading Dataset...")
    print("=" * 60)
    
    csv_path = "data/OnlineNewsPopularity.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    print(f"Loaded {df.shape[0]} records with {df.shape[1]} raw columns.")
    
    # Define Target: Engagement threshold at median shares (1,400)
    # 1 = High Engagement (Popular), 0 = Standard Engagement (Not Popular)
    median_shares = df['shares'].median()
    print(f"Target Threshold (Median Shares): {median_shares}")
    df['is_popular'] = (df['shares'] >= 1400).astype(int)
    
    # Class distribution
    class_counts = df['is_popular'].value_counts().to_dict()
    print(f"Class Distribution: {class_counts} (~{class_counts[1] / len(df) * 100:.1f}% positive)")

    # Select clean, intuitive, and highly explainable features
    # 1. Content structure & length
    # 2. Media richness
    # 3. Content channel (one-hot encoded in dataset)
    # 4. Publication timing
    # 5. Sentiment & Subjectivity
    feature_cols = [
        'n_tokens_title',
        'n_tokens_content',
        'num_hrefs',
        'num_imgs',
        'num_videos',
        'num_keywords',
        'data_channel_is_lifestyle',
        'data_channel_is_entertainment',
        'data_channel_is_bus',
        'data_channel_is_socmed',
        'data_channel_is_tech',
        'data_channel_is_world',
        'is_weekend',
        'global_sentiment_polarity',
        'title_sentiment_polarity',
        'title_subjectivity',
        'kw_avg_avg'
    ]
    
    print("\nSelected Feature Subset for Clean Interpretability:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")
        
    # Handle missing / infinite values if any
    X = df[feature_cols].copy().fillna(0)
    y = df['is_popular'].values
    
    # Precompute summary statistics for EDA in Streamlit
    # Channel distribution and popularity rate
    channels = {
        'Lifestyle': 'data_channel_is_lifestyle',
        'Entertainment': 'data_channel_is_entertainment',
        'Business': 'data_channel_is_bus',
        'Social Media': 'data_channel_is_socmed',
        'Tech': 'data_channel_is_tech',
        'World': 'data_channel_is_world'
    }
    
    channel_stats = []
    for ch_name, ch_col in channels.items():
        subset = df[df[ch_col] == 1]
        if len(subset) > 0:
            channel_stats.append({
                'channel': ch_name,
                'count': int(len(subset)),
                'avg_shares': float(round(subset['shares'].mean(), 1)),
                'median_shares': float(subset['shares'].median()),
                'popular_pct': float(round(subset['is_popular'].mean() * 100, 2))
            })
            
    # Weekend vs Weekday popularity
    timing_stats = {
        'Weekday': {
            'count': int((df['is_weekend'] == 0).sum()),
            'avg_shares': float(round(df[df['is_weekend'] == 0]['shares'].mean(), 1)),
            'popular_pct': float(round(df[df['is_weekend'] == 0]['is_popular'].mean() * 100, 2))
        },
        'Weekend': {
            'count': int((df['is_weekend'] == 1).sum()),
            'avg_shares': float(round(df[df['is_weekend'] == 1]['shares'].mean(), 1)),
            'popular_pct': float(round(df[df['is_weekend'] == 1]['is_popular'].mean() * 100, 2))
        }
    }
    
    # Overall summary stats
    eda_summary = {
        'total_records': int(len(df)),
        'total_features': int(len(feature_cols)),
        'median_shares': float(median_shares),
        'mean_shares': float(round(df['shares'].mean(), 1)),
        'class_balance': {
            'Popular (>= 1400)': int(class_counts.get(1, 0)),
            'Not Popular (< 1400)': int(class_counts.get(0, 0))
        },
        'channel_stats': channel_stats,
        'timing_stats': timing_stats,
        'feature_stats': {
            col: {
                'min': float(round(X[col].min(), 3)),
                'max': float(round(X[col].max(), 3)),
                'mean': float(round(X[col].mean(), 3)),
                'median': float(round(X[col].median(), 3))
            } for col in feature_cols
        }
    }
    
    print("\n" + "=" * 60)
    print("Step 2: Train-Test Split & Feature Scaling...")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n" + "=" * 60)
    print("Step 3: Training & Evaluating Machine Learning Models...")
    print("=" * 60)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1)
    }
    
    comparison_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"--> Training {name}...")
        # For tree models, unscaled or scaled works, but scaled is consistent
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Calculate ROC curve points (subsampled for small JSON size)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        indices = np.linspace(0, len(fpr) - 1, 50, dtype=int)
        roc_pts = [{'fpr': float(round(fpr[idx], 4)), 'tpr': float(round(tpr[idx], 4))} for idx in indices]
        
        # Feature importance / coefficients
        importance_dict = {}
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = {feature_cols[i]: float(round(importances[i], 4)) for i in range(len(feature_cols))}
        elif hasattr(model, 'coef_'):
            coefs = model.coef_[0]
            importance_dict = {feature_cols[i]: float(round(coefs[i], 4)) for i in range(len(feature_cols))}
            
        comparison_results[name] = {
            'Accuracy': float(round(acc, 4)),
            'Precision': float(round(prec, 4)),
            'Recall': float(round(rec, 4)),
            'F1_Score': float(round(f1, 4)),
            'ROC_AUC': float(round(auc, 4)),
            'Confusion_Matrix': cm,
            'ROC_Curve': roc_pts,
            'Feature_Importances': importance_dict
        }
        trained_models[name] = model
        print(f"    Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}% | ROC-AUC: {auc:.4f}")
        
    print("\n" + "=" * 60)
    print("Step 4: Saving Trained Models and Metadata...")
    print("=" * 60)
    os.makedirs("models", exist_ok=True)
    
    # Save best model (Random Forest)
    best_model_name = "Random Forest"
    joblib.dump(trained_models[best_model_name], "models/best_model.joblib")
    joblib.dump(trained_models, "models/all_models.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
        
    with open("models/model_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
        
    with open("models/eda_summary.json", "w") as f:
        json.dump(eda_summary, f, indent=2)
        
    # Also save a small sample test set for quick testing
    test_sample = pd.DataFrame(X_test, columns=feature_cols).head(20)
    test_sample['actual_popular'] = y_test[:20]
    test_sample.to_csv("data/sample_test.csv", index=False)
    
    print("✓ Successfully saved best_model.joblib, all_models.joblib, scaler.joblib")
    print("✓ Successfully saved feature_names.json, model_comparison.json, eda_summary.json")
    print("✓ Successfully generated data/sample_test.csv")
    print("\nAll pipeline steps completed successfully!")

if __name__ == "__main__":
    main()
