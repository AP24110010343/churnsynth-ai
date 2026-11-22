import numpy as np
import pandas as pd

def generate_baseline_subscription_data(n_samples=4000, random_seed=42):
    """
    Generates a realistic retail subscription box customer lifecycle dataset.
    Reflects complex non-linear correlations between subscription features and churn.
    """
    np.random.seed(random_seed)

    tiers = ['Basic', 'Standard', 'Premium']
    tier_probs = [0.45, 0.35, 0.20]
    tier_choices = np.random.choice(tiers, size=n_samples, p=tier_probs)

    categories = ['Beauty & Care', 'Gourmet Snacks', 'Eco Home', 'Fitness & Wellness', 'Book & Craft']
    category_probs = [0.30, 0.25, 0.15, 0.18, 0.12]
    cat_choices = np.random.choice(categories, size=n_samples, p=category_probs)

    # Base pricing based on tier with natural variation
    tier_price_map = {'Basic': 19.99, 'Standard': 34.99, 'Premium': 54.99}
    monthly_fee = np.array([tier_price_map[t] + np.random.normal(0, 2.5) for t in tier_choices])
    monthly_fee = np.clip(monthly_fee, 12.0, 65.0).round(2)

    # Tenure in months: exponential distribution (many new, fewer long-term)
    tenure_months = np.random.exponential(scale=14.0, size=n_samples).astype(int) + 1
    tenure_months = np.clip(tenure_months, 1, 48)

    # Discount applied during signup (more common in Basic, less in Premium)
    discount_prob = np.where(tier_choices == 'Basic', 0.55, np.where(tier_choices == 'Standard', 0.35, 0.20))
    discount_applied = (np.random.rand(n_samples) < discount_prob).astype(int)

    # App logins per month (engagement)
    base_logins = np.random.poisson(lam=8.0, size=n_samples) + (tenure_months * 0.2)
    app_logins = np.clip(base_logins, 0, 30).astype(int)

    # Skipped boxes ratio: customers skipping boxes often churn
    skipped_ratio = np.random.beta(a=1.5, b=5.0, size=n_samples)
    skipped_ratio = np.clip(skipped_ratio, 0.0, 0.8).round(3)

    # Customer service tickets raised (0 to 7)
    cs_tickets = np.random.poisson(lam=1.2, size=n_samples)
    cs_tickets = np.clip(cs_tickets, 0, 7)

    # Delivery delay incidents experienced
    delay_incidents = np.random.poisson(lam=0.8, size=n_samples)
    delay_incidents = np.clip(delay_incidents, 0, 5)

    # Engagement score (composite 0 - 100)
    engagement_score = (
        app_logins * 2.5 
        - (skipped_ratio * 35.0) 
        - (cs_tickets * 4.0) 
        + np.where(tier_choices == 'Premium', 15.0, 5.0)
        + np.random.normal(45, 10, size=n_samples)
    )
    engagement_score = np.clip(engagement_score, 5.0, 98.0).round(1)

    # Churn probability computation using realistic logistic formula
    # High churn drivers: low tenure, high skipped ratio, high CS tickets, delivery delays, low engagement
    logit = (
        - 1.2
        - (tenure_months * 0.06)
        + (skipped_ratio * 3.8)
        + (cs_tickets * 0.55)
        + (delay_incidents * 0.40)
        - (engagement_score * 0.045)
        + (discount_applied * 0.45) # discount hunters churn faster
        + np.where(tier_choices == 'Basic', 0.35, -0.2)
    )
    
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churned = (np.random.rand(n_samples) < churn_prob).astype(int)

    customer_ids = [f"SUB-{10000 + i}" for i in range(n_samples)]

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'subscription_tier': tier_choices,
        'box_category': cat_choices,
        'tenure_months': tenure_months,
        'monthly_fee': monthly_fee,
        'discount_applied': discount_applied,
        'app_logins_per_month': app_logins,
        'skipped_boxes_ratio': skipped_ratio,
        'cs_tickets_raised': cs_tickets,
        'delivery_delays_count': delay_incidents,
        'engagement_score': engagement_score,
        'churned': churned
    })

    return df

if __name__ == '__main__':
    df = generate_baseline_subscription_data(4000)
    df.to_csv('baseline_subscription_data.csv', index=False)
    print(f"Generated {len(df)} baseline records.")
    print(f"Overall Churn Rate: {df['churned'].mean():.2%}")
    print(df.head())
