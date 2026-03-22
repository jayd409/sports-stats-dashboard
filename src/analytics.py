import pandas as pd
import numpy as np
from ml_utils import kmeans

def compute_all_metrics(df):
    """Compute basketball analytics: PER, clustering, team stats, and efficiency metrics."""

    # Player Efficiency Rating (simplified)
    # PER = (Points + Rebounds + Assists + Steals + Blocks - Turnovers) / Minutes * 36
    df_metrics = df.copy()
    df_metrics['PER'] = (
        (df_metrics['points'] + df_metrics['rebounds'] + df_metrics['assists'] +
         df_metrics['steals'] + df_metrics['blocks'] - df_metrics['turnovers']) / df_metrics['minutes'] * 36
    ).round(2)

    # Efficiency: Points per Turnover
    df_metrics['efficiency'] = (df_metrics['points'] / (df_metrics['turnovers'] + 0.1)).round(2)

    # Team aggregates
    team_stats = df_metrics.groupby('team').agg({
        'points': 'sum',
        'rebounds': 'sum',
        'assists': 'sum',
        'player': 'count'
    }).round(1)
    team_stats.columns = ['total_points', 'total_rebounds', 'total_assists', 'num_players']
    team_stats = team_stats.sort_values('total_points', ascending=False)

    # Top 10 by category
    top_scorers = df_metrics.nlargest(10, 'points')[['player', 'team', 'points']].reset_index(drop=True)
    top_rebounders = df_metrics.nlargest(10, 'rebounds')[['player', 'team', 'rebounds']].reset_index(drop=True)
    top_passers = df_metrics.nlargest(10, 'assists')[['player', 'team', 'assists']].reset_index(drop=True)

    # Position averages
    pos_stats = df_metrics.groupby('position').agg({
        'points': 'mean',
        'rebounds': 'mean',
        'assists': 'mean',
        'steals': 'mean',
        'blocks': 'mean',
        'PER': 'mean'
    }).round(2)

    # K-means clustering on [points, rebounds, assists]
    clustering_features = df_metrics[['points', 'rebounds', 'assists']].values
    labels, centroids = kmeans(clustering_features, k=3, n_iter=100, seed=42)
    df_metrics['cluster'] = labels

    # Cluster names based on centroid characteristics
    cluster_names = {}
    for i in range(3):
        pts, reb, ast = centroids[i]
        if pts > 13:
            cluster_names[i] = 'Star'
        elif pts > 8:
            cluster_names[i] = 'Contributor'
        else:
            cluster_names[i] = 'Role Player'

    df_metrics['cluster_name'] = df_metrics['cluster'].map(cluster_names)

    # Top 5 by PER
    top_per = df_metrics.nlargest(5, 'PER')[['player', 'team', 'PER', 'position']].reset_index(drop=True)

    return {
        'df_metrics': df_metrics,
        'team_stats': team_stats,
        'pos_stats': pos_stats,
        'top_scorers': top_scorers,
        'top_rebounders': top_rebounders,
        'top_passers': top_passers,
        'top_per': top_per,
        'cluster_names': cluster_names,
    }
