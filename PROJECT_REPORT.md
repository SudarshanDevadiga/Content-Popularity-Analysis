# Case Study No. 124: Content Popularity Analysis
## Final Project Report

---

### Project Title
**Supervised Machine Learning System for Digital Content Engagement Forecasting and Article Optimization**

---

## 1. Problem Definition & Formulation

### 1.1 Context and Real-World Motivation
In modern digital publishing (media portals, news agencies, content marketing blogs), content editors and digital strategists face intense competition for reader attention. Thousands of articles are drafted daily, requiring editorial decisions regarding headline composition, length, multimedia inclusion, topic tagging, and release timing.

Currently, editorial decisions often rely on intuition or post-publication analytics (e.g., observing click counts hours after an article goes live). This project builds a **pre-publication machine-learning estimation system** that assesses draft article attributes prior to publication and predicts whether the article is likely to attract high reader engagement.

### 1.2 Formal Machine Learning Problem Definition
* **Learning Task**: Supervised Binary Classification.
* **Input ($\mathbf{x} \in \mathbb{R}^d$)**: Pre-publication article attributes including headline length, word count, media count (images/videos), content channel/category, keyword popularity, and publishing day.
* **Target ($y \in \{0, 1\}$)**:
  * **$y = 1$ (High Engagement / Popular)**: Social media shares $\ge 1,400$
  * **$y = 0$ (Standard Engagement / Not Popular)**: Social media shares $< 1,400$

### 1.3 Target Justification
Social media shares follow an extreme right-skewed Power-Law distribution (a handful of articles receive hundreds of thousands of shares, while the majority cluster near the median). Formulating this as a median-threshold binary classification:
1. Yields a balanced class distribution (~53.4% positive class vs. 46.6% negative class), avoiding severe class imbalance penalties.
2. Avoids regression sensitivity to extreme virality outliers.
3. Delivers a clean, actionable binary recommendation for content teams: *"Is this article above or below our viral benchmark?"*

---

## 2. Dataset & Documentation

### 2.1 Dataset Source
* **Dataset Name**: Online News Popularity Dataset
* **Source**: UCI Machine Learning Repository (Acquired from Mashable articles published between January 2013 and January 2015).
* **Reference**: K. Fernandes, P. Vinagre, and P. Cortez. *A Proactive Intelligent Decision Support System for Predicting the Popularity of Online News*. EPIA 2015.

### 2.2 Data Description & Variables
* **Total Instances**: 39,644 articles
* **Total Raw Attributes**: 61 attributes (non-predictive identifiers like URL and timedelta, plus 58 content/timing/sentiment attributes and 1 target `shares`).

### 2.3 Curated Feature Subset for High Explainability
To maintain model transparency and practical utility for demonstration, 17 core predictive features were selected across 5 logical categories:

| Feature Name | Description | Type | Mean / Range |
| :--- | :--- | :--- | :--- |
| `n_tokens_title` | Number of words in the headline | Numerical | Mean: 10.4 words (Range: 2–23) |
| `n_tokens_content` | Number of words in the article body | Numerical | Mean: 546.5 words (Range: 0–8,474) |
| `num_hrefs` | Number of hyperlinks in the article | Numerical | Mean: 10.9 links |
| `num_imgs` | Number of images embedded | Numerical | Mean: 4.5 images |
| `num_videos` | Number of embedded videos | Numerical | Mean: 1.2 videos |
| `num_keywords` | Number of keywords in metadata | Numerical | Mean: 7.2 keywords |
| `data_channel_is_lifestyle` | Channel: Lifestyle | Binary | 5.3% of articles |
| `data_channel_is_entertainment`| Channel: Entertainment | Binary | 17.8% of articles |
| `data_channel_is_bus` | Channel: Business | Binary | 15.8% of articles |
| `data_channel_is_socmed` | Channel: Social Media | Binary | 5.9% of articles |
| `data_channel_is_tech` | Channel: Technology | Binary | 18.5% of articles |
| `data_channel_is_world` | Channel: World News | Binary | 21.3% of articles |
| `is_weekend` | Published on Saturday or Sunday | Binary | 13.0% of articles |
| `global_sentiment_polarity` | Overall polarity of article text | Numerical | Range: -1.0 (neg) to +1.0 (pos) |
| `title_sentiment_polarity` | Sentiment polarity of headline | Numerical | Range: -1.0 to +1.0 |
| `title_subjectivity` | Subjectivity level of headline | Numerical | Range: 0.0 (factual) to 1.0 (opinion) |
| `kw_avg_avg` | Average popularity of targeted keywords| Numerical | Mean: 3,136 shares |

### 2.4 Data Quality Observations & Handling
1. **Column Spacing**: Raw column names contained leading whitespace (e.g., `' shares'`), which was stripped.
2. **Identifiers Dropped**: Non-predictive fields (`url`, `timedelta`) were excluded to prevent target leakage and spurious correlations.
3. **Missing Values**: Handled using numerical imputation (median/zero substitution where appropriate).
4. **Outliers**: Features such as word count and keyword averages were standardized using `StandardScaler` to ensure zero-mean and unit-variance.

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Key Findings & Visual Insights

#### 1. Content Category / Channel Dynamics
* **Social Media Articles**: Highest proportion of high engagement (**65.6%** above median), driven by visual and easily digestible formats.
* **Technology**: Second most consistent category (**55.4%** high engagement), benefiting from passionate tech readerships.
* **World News**: Lowest engagement rate (**32.8%** high engagement). Routine political updates rarely generate massive cross-platform sharing unless breaking or sensational.

#### 2. The "Weekend Effect"
* Articles published on **Weekends (Sat-Sun)** achieved a **63.8%** high-engagement rate, compared to **51.8%** for weekday articles.
* **Justification**: On weekends, total publication volume drops sharply (publishers reduce output), meaning less clutter in user newsfeeds. Furthermore, readers have leisure time to consume and share articles.

#### 3. Multimedia Richness
* Articles containing **2 to 5 images** and at least **1 embedded video** demonstrated a 15–20% higher probability of exceeding 1,400 shares than text-only articles.

#### 4. Headline Length
* Headlines between **9 and 12 words** showed the highest probability of engagement. Headlines that are too short (<5 words) fail to convey curiosity or value, while headlines that are too long (>16 words) suffer from lower click-through on mobile displays.

---

## 4. Preprocessing Pipeline

1. **Target Binarization**:
   $$y_i = \begin{cases} 1 & \text{if } \text{shares}_i \ge 1,400 \\ 0 & \text{if } \text{shares}_i < 1,400 \end{cases}$$
2. **Data Partitioning**:
   * Stratified Train-Test Split: **80% Training (31,715 samples)** and **20% Testing (7,929 samples)**.
   * Stratification ensures the exact ~53.4% / 46.6% class ratio is preserved across both splits.
3. **Feature Normalization**:
   * Scaled using Scikit-Learn's `StandardScaler`:
     $$z = \frac{x - \mu}{\sigma}$$
   * Scaler fitted strictly on the training set and applied to the test set to avoid data leakage.

---

## 5. Model Development & Comparison

Three representative machine learning algorithms with distinct inductive biases were trained and evaluated:
1. **Logistic Regression**: Linear baseline model with L2 regularization ($C=1.0$).
2. **Decision Tree Classifier**: Non-linear tree model with depth constraint ($\text{max\_depth}=6, \text{min\_samples\_leaf}=20$) to prevent overfitting.
3. **Random Forest Classifier**: Ensemble of 100 bagged decision trees ($\text{max\_depth}=12, \text{min\_samples\_leaf}=10$) aggregating predictions across decorrelated feature subspaces.

### 5.1 Experimental Results (Test Set: 7,929 Samples)

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 64.71% | 65.65% | 71.05% | 68.24% | 0.6883 |
| **Decision Tree** | 64.23% | 66.05% | 67.83% | 66.93% | 0.6796 |
| **Random Forest (Best)** | **65.22%** | **65.79%** | **72.54%** | **69.00%** | **0.7088** |

### 5.2 Model Selection Justification
* **Random Forest** achieved the best overall performance with **65.22% Accuracy** and **0.7088 ROC-AUC**.
* It captured complex non-linear interactions (e.g., how the combination of keyword index and weekend timing amplifies reach) without succumbing to single-tree variance.
* **Recall is 72.54%**, which is particularly valuable for editorial teams who want to avoid missing potential viral hits.

---

## 6. Evaluation & Error Analysis

### 6.1 Confusion Matrix (Random Forest on Test Set)
* **True Negatives ($TN$)**: 2,126 (correctly predicted standard articles)
* **False Positives ($FP$)**: 1,572 (predicted viral, but did not reach 1,400 shares)
* **False Negatives ($FN$)**: 1,162 (predicted standard, but went viral)
* **True Positives ($TP$)**: 3,069 (correctly identified high engagement articles)

### 6.2 Top Predictive Features
1. `kw_avg_avg` (Keyword average historical popularity): The single strongest predictor. High-interest search queries strongly correlate with social sharing.
2. `is_weekend`: Publishing timing significantly shifts the baseline probability.
3. `data_channel_is_world` & `data_channel_is_socmed`: Channel category heavily shapes reader interest.
4. `n_tokens_content`: Article depth and substantive length.
5. `num_imgs` & `num_hrefs`: Visual richness and cross-referencing.

### 6.3 Project Limitations
* **Pre-Publication Information Boundary**: The model only analyzes variables observable *before* publication. It cannot foresee external post-publication viral shocks (e.g., an influencer retweeting the link, a breaking news event shifting public interest, or platform algorithm tweaks).
* **Textual Nuance**: The dataset relies on aggregate sentiment and metadata rather than full-context semantic transformers (e.g., BERT), keeping it lightweight and fast for production deployment.

---

## 7. Streamlit Application Overview

A fully functional, interactive Streamlit application was created (`app.py`).
* **Section 1: Problem Definition**: Outlines the business context, objectives, and dataset metrics.
* **Section 2: Exploratory Data Analysis**: Visualizes channel rates, weekend effect, and feature distributions.
* **Section 3: Model Comparison & Results**: Interactive metric tables, selectable confusion matrix heatmaps, and feature importance rankings.
* **Section 4: Live Popularity Predictor**: Allows users to input headline length, word count, media assets, channel, timing, and sentiment to receive an instant engagement prediction, confidence score, and targeted optimization tips.

---

## 8. Conclusion & Strategic Recommendations

1. **Strategic Scheduling**: Schedule key in-depth or lifestyle content for weekend mornings to capture higher reader attention with lower competition.
2. **Multimedia Standards**: Require minimum multimedia standards (at least 2 relevant images and structured links) for all editorial drafts.
3. **Keyword Optimization**: Use keyword trends to tag articles with high-affinity search queries before scheduling.
4. **Feasibility**: The developed solution provides a lightweight, highly interpretable machine-learning system that can be seamlessly embedded into an editorial Content Management System (CMS).
