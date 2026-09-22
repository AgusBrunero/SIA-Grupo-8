import ast
import matplotlib.pyplot as plt
import pandas as pd

def analyze_continuous_columns(df: pd.DataFrame):
    discrete_cols = ['quantity_purchased', 'items_viewed_before_purchase']
    for col in discrete_cols:
        fraud_rate = df.groupby(col)['flagged_fraud'].mean().sort_index()
        plt.figure(figsize=(8, 4))
        plt.plot(fraud_rate.index, fraud_rate.values, marker='o')
        plt.title(f'Fraud Rate vs {col}')
        plt.xlabel(col)
        plt.ylabel('Fraud Rate')
        plt.grid(True)
        plt.show()

    continuous_cols = [
        'amount_usd', 'session_duration_seconds', 'days_since_last_purchase',
        'account_age_days', 'time_since_last_login_s', 'big_model_fraud_probability'
    ]
    for col in continuous_cols:
        binned_data = pd.qcut(df[col], q=20, duplicates='drop')
        fraud_rate = df.groupby(binned_data)['flagged_fraud'].mean()

        x_vals = [interval.mid for interval in fraud_rate.index]

        plt.figure(figsize=(8, 4))
        plt.plot(x_vals, fraud_rate.values, marker='o')
        plt.title(f'Fraud Rate vs {col} (Binned)')
        plt.xlabel(col)
        plt.ylabel('Fraud Rate')
        plt.grid(True)
        plt.show()

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['device_screen_resolution'] = pd.to_numeric(df['device_screen_resolution']) // 1_000_000
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day_name()
    return df

def analyze_hour(df: pd.DataFrame):
    fraud_rate=df.groupby('hour')['flagged_fraud'].mean().sort_index()
    plt.bar(fraud_rate.index, fraud_rate.values)
    plt.xticks(range(24), range(24))
    plt.xlabel('Hour of day')
    plt.ylabel('Fraud Rate (%)')
    plt.show()
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    fraud_rate=df.groupby('day')['flagged_fraud'].mean().reindex(day_order)
    plt.bar(fraud_rate.index,fraud_rate.values)
    plt.xlabel('Day of week')
    plt.ylabel('Fraud Rate (%)')
    plt.xticks(rotation=45)
    plt.show()
    fraud_rate=((df.groupby(['day','hour'])['flagged_fraud'].mean()
                .unstack(fill_value=0)
                .reindex(day_order))
                .reindex(columns=range(24),fill_value=0))
    plt.figure(figsize=(14, 5))
    plt.imshow(
        fraud_rate,
        aspect="auto",
        cmap="Reds",
        interpolation="nearest"
    )
    plt.colorbar(label="Fraud Rate")
    plt.xticks(range(24), range(24))
    plt.yticks(range(len(day_order)), day_order)
    plt.xlabel("Hour of day")
    plt.ylabel("Day of week")
    plt.title("Fraud rate by day and hour")
    plt.show()

def analyze_screen_resolution(df: pd.DataFrame):
    summary = (
        df.groupby("device_screen_resolution")
        .agg(
            total_users=("flagged_fraud", "size"),
            frauds=("flagged_fraud", "sum"),
            fraud_rate=("flagged_fraud", "mean"),
        )
        .sort_index()
    )
    print(summary)
    fraud_rate=df.groupby('device_screen_resolution')["flagged_fraud"].mean().sort_index()
    plt.bar(fraud_rate.index.astype(str), fraud_rate.values)
    plt.xlabel('Screen Resolution (Mp)')
    plt.ylabel('Fraud Rate (%)')
    plt.show()

if __name__ == '__main__':
    df = load_dataset('./datasets/fraud_dataset.csv')
    analyze_screen_resolution(df)
    analyze_hour(df)
    analyze_continuous_columns(df)
