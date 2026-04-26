# Business Case Analysis

## 50 Stores,5 Promotions

# B1. Problem Formulation 

## (a) ML Problem Formulation 

This is a **supervised, multi-class classification problem**. The goal is to learn a mapping from store and context features to the promotion type that maximises items sold, enabling the marketing team to deploy the optimal promotion for each store each month.

| ML Component | Definition for This Problem |

| **Target Variable (y)** | Optimal promotion type — one of: Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, Loyalty Points Bonus (5 classes) |
| **Input Features (X)** | Store size, location type (urban/semi-urban/rural), monthly footfall, local competition density, customer demographics (age, income bracket), month/season, is_weekend, is_festival flag |
| **Problem Type** | Multi-class classification — five discrete promotion labels, one chosen per store per month |
| **Justification** | Classification is preferred over regression because the output is a discrete decision (which promotion to run), not a continuous value. The model learns which promotion historically generated the highest items sold under similar conditions, then recommends it |


## (b) Items Sold vs. Total Sales Revenue as Target Variable 

Using total sales revenue as the target variable introduces a confound: revenue is the product of items sold and price. Two promotions may sell the same volume, but one sells higher-priced items, making revenue misleading about which promotion drove customer engagement. Conversely, a deep discount may inflate revenue purely through price reduction while depressing margins.

| Metric | Total Sales Revenue | Items Sold (Volume) |

| Captures promotion pull? | Indirectly — conflated with price | Directly — pure demand signal |
| Sensitive to markdowns? | Yes — deep discounts distort it | No — price-independent |
| Reflects footfall conversion? | Partially | Yes — each sale = one unit |
| Actionable for marketing? | Less — pricing team owns it | Yes — marketing drives volume |

**Broader Principle — Target Variable Selection:**

The target variable must be the most direct, least-confounded proxy for the business objective. Revenue is a downstream outcome influenced by factors (pricing strategy, product mix, margin policy) outside marketing's control. Items sold isolates the demand-generation effect of the promotion — the causal mechanism the model is meant to learn. This illustrates the principle: **choose targets that are causally close to the intervention being optimised, and strip out effects that your intervention does not control.**

> **Key Insight:** If the model were trained on revenue, it would implicitly favour promotions run during high-ASP (average selling price) periods — e.g., winter coats over summer accessories — rewarding seasonality rather than promotion effectiveness. Items sold removes this artefact.


## (c) Global vs. Store-Segmented Modelling Strategy

A junior analyst's proposal of a single global model assumes homogeneity: that the effect of each promotion type is the same across all 50 stores. This is unlikely — an urban flagship store with 10,000 monthly footfall and affluent demographics will respond very differently to a Loyalty Points Bonus than a small rural outlet with 400 monthly visitors.

**Proposed Alternative: Hierarchical / Segmented Modelling**

| Strategy | Description | Best For |

| **Segment-level models** | Train one model per location type (urban, semi-urban, rural). Each model learns promotion responses within its segment. | When segments are large enough (≥200 records each) and location type is the key heterogeneity driver |
| **Store-level models with global prior** | Train a global model, then fine-tune per store using local data (hierarchical Bayesian or transfer learning). Global model provides stability; local fine-tuning captures idiosyncrasies. | When individual stores have limited history (cold-start problem). Balances bias and variance |
| **Mixed-effects model** | Include store_id as a random effect. The global fixed effects capture average promotion impact; random effects capture store-level deviations. | When store-level differences are modest departures from a common pattern rather than fundamentally different dynamics |

> **Recommended Approach:** A two-tier strategy: (1) Train separate Random Forest classifiers for each `location_type` segment (urban / semi-urban / rural). This accounts for the largest source of heterogeneity. (2) Include `store_id` as a high-cardinality categorical feature (target-encoded) within each segment model to capture residual store-level effects. This gives the model the ability to personalise further without requiring 50 separate models with thin training data.


# B2. Data and EDA Strategy

## (a) Data Joins, Grain, and Aggregations

The raw data arrives in four separate tables. The join strategy below assembles them into a single analytical dataset with **one row per (store × month)**, which is the natural grain for a monthly promotion-allocation decision.

**Step 1 — Define the Spine (Grain):**

Generate a complete (store_id × year_month) grid covering every store for every month in scope. This ensures stores with no transactions in a given month are represented (and can be flagged as zero-footfall months rather than missing).

**Step 2 — Join Tables:**

| Join | Keys | Type | Result |

| Spine ← Transactions | store_id, year_month | LEFT JOIN then aggregate | Aggregated sales metrics per store-month (see Step 3 below) |
| ← Store Attributes | store_id | LEFT JOIN (many-to-one) | Adds store_size, location_type, demographics — static per store |
| ← Promotion Details | promotion_type | LEFT JOIN (many-to-one) | Adds promotion metadata (discount_depth, mechanic_type). Promotion is determined per store per month by the transactions table |
| ← Calendar | date (exploded to year_month) | LEFT JOIN then aggregate | Adds is_weekend count, is_festival flag (any festival day in the month), and month/quarter/season labels |

**Step 3 — Aggregations Before Modelling:**

- **Total items_sold** per store per month (target variable and feature for lag engineering)
- **Dominant promotion_type** per store per month (mode of daily promotion, or as provided by the promotion schedule)
- **Monthly footfall** — sum or average daily footfall transactions
- **festival_days_in_month** — count of festival days (not just binary flag) for richer signal
- **Lag features** — items_sold in t-1 and t-2 months, rolling 3-month average, to capture momentum
- **competition_density** — taken as-is from store attributes (assumed static or slowly varying)

> **Grain: One Row = One Store × One Month.** This grain matches the decision cadence: marketing assigns promotions at the start of each month per store. A row encodes the store's context (size, location, footfall, competition) and the calendar context (month, festivals, weekends), and the label is which promotion was run and how many items were sold.


## (b) EDA Strategy — Four Key Analyses

The following four analyses are performed before modelling, each with a specific decision impact:

| # | Analysis / Chart | What to Look For | Decision Impact |

| **1** | **Promotion × Items Sold Box Plot** (faceted by location_type) | Which promotion type produces the highest median items sold in each location segment? Are there promotions that consistently underperform? | Informs whether location-segmented models are warranted. If BOGO dominates in urban but underperforms in rural, a global model will average these effects and mislead |
| **2** | **Heatmap: Avg. Items Sold by Month × Promotion Type** | Identify seasonality interactions — does Flat Discount work better in January (post-festive clearance) but Loyalty Points work better mid-year? | Drives feature engineering decisions: month, quarter, and season must be included as features. May reveal that certain promotions should be locked to certain seasons |
| **3** | **Footfall vs. Items Sold Scatter** (coloured by promotion) | Is the relationship linear? Do high-footfall stores benefit more from certain promotion types (conversion-rate vs. awareness effect)? | Guides whether footfall × promotion interaction terms are needed. May justify a log-transform of footfall if the relationship is sub-linear |
| **4** | **Correlation Heatmap** (numeric features vs. items_sold) | Check for multicollinearity (e.g., store_size and footfall likely correlated). Identify which numeric features have the strongest individual signal. | Features with correlation > 0.8 to each other may need to be de-duplicated (keep one or apply PCA). Weak-signal features (\|r\| < 0.05) are candidates for removal |

> **Bonus EDA — Promotion Rotation Frequency:** Plot a Sankey or sequence diagram showing how often stores change promotion month-to-month. If stores use the same promotion for 6+ consecutive months, this introduces temporal autocorrelation that a naive i.i.d. model will ignore. This finding would motivate adding a `months_on_same_promo` lag feature.


## (c) Handling 80% No-Promotion Transactions

If 80% of transactions carry no promotion, three problems arise:

- **Class imbalance in classification:** The "no promotion" class dominates, and the model may achieve high accuracy simply by predicting no promotion for every store — a degenerate solution.
- **Biased feature importance:** Features associated with the no-promotion baseline will appear artificially important.
- **Baseline conflation:** Items sold under no promotion reflects organic demand, not promotion lift. Training on these rows without a flag teaches the model the wrong signal.

**Mitigation Strategy:**

| Step | Action | Rationale |

| **1 — Reframe the Problem** | Restrict the modelling dataset to months where a promotion was actively run. Treat no-promotion months as the baseline and model lift above baseline instead. | The business question is which promotion to deploy — implicitly assuming one will be deployed. No-promotion rows are not decision-relevant. |
| **2 — Add a 'no_promo_baseline' Feature** | For each store, compute average items_sold in no-promotion months. Include this as a feature in promotion months. | Gives the model a store-specific demand floor, helping it learn genuine lift rather than conflating baseline demand with promotion effect. |
| **3 — Class Weights / SMOTE (if needed)** | If no-promotion rows must be included (e.g., to predict whether to promote at all), apply `class_weight='balanced'` in sklearn or SMOTE oversampling for minority promotion classes. | Prevents the majority class from dominating gradient updates during training. |
| **4 — Stratified CV** | Ensure each cross-validation fold preserves the promotion class distribution using `StratifiedKFold`. | Prevents folds where minority promotion classes are absent, which inflates CV accuracy. |


# B3. Model Evaluation and Deployment

## (a) Train-Test Split and Evaluation Metrics

The dataset spans three years across 50 stores. With time-ordered, store-level monthly data, a random split would cause data leakage — models trained on December 2024 data would be tested on January 2022 data, making test performance optimistically biased. The correct approach is a **temporal split**.

**Temporal Split Design:**

- **Sort** all rows by year_month ascending
- **Train set:** Months 1–30 (the first 2.5 years, e.g., Jan 2022 – Jun 2024). All 50 stores present.
- **Test set:** Months 31–36 (the final 6 months, e.g., Jul – Dec 2024). The model never sees these during training.
- **Walk-forward validation** (preferred): Instead of a single split, use expanding windows — train on months 1–12, validate on 13; train on 1–24, validate on 25; etc. Average performance across folds. This is more robust than a single temporal holdout.

> **Why Not Random Split?** Random splitting treats each (store × month) row as i.i.d. — independent and identically distributed. But monthly store performance is autocorrelated (January 2023 depends on December 2022), and future months cannot be used to predict past months in production. A random split inflates test accuracy by allowing the model to "borrow" from adjacent months that it would never see in deployment.

**Evaluation Metrics:**

| Metric | Formula | Business Interpretation |

| **Weighted F1-Score** | Harmonic mean of precision and recall, weighted by class support | Primary metric. Accounts for class imbalance across 5 promotion types. A model that always recommends BOGO scores poorly here. |
| **Per-Class Recall (Sensitivity)** | TP / (TP + FN) per promotion type | Ensures no promotion type is systematically ignored. If recall for "Category Offer" is 0.10, the model never recommends it — a deployment risk. |
| **Top-1 Accuracy** | % of store-months where recommended promotion = historically best promotion | Simple stakeholder-facing metric. Complements F1 for executive reporting. |
| **Regret / Opportunity Cost** | (items_sold under best promo) − (items_sold under recommended promo), averaged | Business-value metric. A wrong recommendation that costs 5 items is less damaging than one that costs 200. Converts model errors into revenue terms. |


## (b) Explaining Different Recommendations for Store 12 in December vs. March

The model recommends Loyalty Points Bonus for Store 12 in December and Flat Discount in March. To investigate and communicate this, we use feature importance analysis combined with counterfactual reasoning.

**Step 1 — Global Feature Importance (Random Forest):**

Extract the Gini-based feature importances from the trained Random Forest. This tells the marketing team which features most commonly drive the promotion decision across all stores and months — e.g., "festival flag is the most important feature, followed by footfall and month."

**Step 2 — Local Explanation with SHAP (SHapley Additive exPlanations):**

For each prediction, compute SHAP values to explain why the model chose that specific promotion for Store 12 in December vs. March. SHAP decomposes the model's output into the additive contribution of each input feature for that exact row.

| Feature | Store 12 — December | SHAP Impact (Dec) | Store 12 — March | SHAP Impact (Mar) |

| month | 12 (December) | ↑ strongly favours Loyalty Points | 3 (March) | ↑ favours Flat Discount |
| is_festival | 1 (festive season) | ↑ boosts loyalty-based promotions | 0 (no festival) | ↔ neutral |
| footfall | High (8,200) | ↑ loyalty works better with repeat visitors | Moderate (4,100) | ↓ Flat Discount attracts new shoppers |
| competition_density | Low (2 nearby stores) | ↑ less need for aggressive discounting | High (7 nearby stores) | ↑ discount needed to compete |

**Step 3 — Counterfactual Explanation for the Marketing Team:**

> **Plain-English Communication to Stakeholders:**
>
> In December, Store 12 benefits from high festive footfall and low local competition — conditions under which loyalty rewards retain existing customers who are already visiting frequently. The model learned from historical data that Loyalty Points generate 18% more items sold than Flat Discounts in these conditions.
>
> In March, footfall drops and seven nearby competitors run promotions simultaneously. Historical data shows that Flat Discounts are the strongest footfall driver in high-competition months — they attract price-sensitive customers who might otherwise go elsewhere. The model is not "changing its mind" arbitrarily; it is responding to the changed input context, just as a human expert would.

This illustrates the principle that feature importance operates at two levels: **global importance** tells us what generally matters most, while **local SHAP values** tell us why this specific decision was made — both are needed for trust and adoption by the marketing team.


## (c) End-to-End Deployment Process

The model must generate monthly promotion recommendations for all 50 stores at the start of each month, without retraining. Below is the full deployment pipeline.

**Phase 1 — Model Serialisation (One-Time):**

- After final training, serialise the trained pipeline (preprocessor + model) using `joblib.dump('rf_pipeline.pkl')`
- Serialise the **label encoder** and **feature schema** (expected column names and dtypes) separately to validate incoming data at inference time
- Store all artefacts in a **versioned model registry** (e.g., MLflow, AWS S3 with version tags). Retain previous versions for rollback

**Phase 2 — Monthly Inference Pipeline (Recurring):**

- **Data ingestion:** At month-start (e.g., 1st of each month), a scheduled job pulls the previous month's transactions, footfall, and calendar data from the data warehouse
- **Feature construction:** Run the same feature engineering script used during training (year, month, lag features, festival flags, competition density) on the 50-store batch. Output: a 50-row DataFrame with identical columns to the training feature matrix
- **Schema validation:** Assert that all expected columns are present and within valid ranges. Reject and alert if missing features exceed 10% of rows
- **Inference:** Load the serialised pipeline and call `pipeline.predict(X_new)` to generate one promotion label per store. Optionally call `predict_proba` to attach confidence scores
- **Output delivery:** Write recommendations to a database table or dashboard (e.g., Tableau, Power BI) accessible to the marketing team, with confidence scores and top-2 alternatives for human override

**Phase 3 — Monitoring and Retraining Triggers:**

| Monitoring Signal | Detection Method | Threshold / Action |

| **Data drift** | Track distribution of each input feature (footfall, competition_density, month) using Population Stability Index (PSI) or KS test each month | PSI > 0.2 on any key feature → alert data engineering team; investigate source change |
| **Prediction drift** | Monitor distribution of recommended promotion types month-over-month. Flag if one class drops to < 5% of recommendations unexpectedly | Sudden class collapse → likely a feature pipeline bug or a data ingestion failure upstream |
| **Performance degradation** | Each month, compare actual items_sold (once available, ~30-day lag) against model-recommended promotion outcomes. Track rolling Regret metric | Rolling 3-month Regret increases > 15% above training-time Regret → trigger retraining |
| **Retraining cadence** | Scheduled quarterly retraining using an expanding window (all available history). Evaluate candidate model on a 2-month holdout before promoting to production | Quarterly is sufficient for stable retail; accelerate to monthly if performance degrades faster |

> **Deployment Architecture Summary:** Serialised model → versioned model registry → monthly scheduled inference job → schema validation → batch predict (50 stores) → write to marketing dashboard → monitor data drift + rolling regret → trigger retraining when thresholds breached. All steps are logged with timestamps for auditability.