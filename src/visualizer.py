import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import save_dashboard

def build_dashboard(metrics):
    """Build 6-chart sports analytics dashboard."""

    sns.set_style("whitegrid")
    df_metrics = metrics['df_metrics']

    # Chart 1: Top 10 scorers (horizontal bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10 = metrics['top_scorers'].head(10)
    y_pos = np.arange(len(top_10))
    ax.barh(y_pos, top_10['points'].values, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_10['player'].values, fontsize=9)
    ax.set_xlabel('Points Scored')
    ax.set_title('Top 10 Scorers', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    for i, v in enumerate(top_10['points'].values):
        ax.text(v + 5, i, f'{v:.1f}', va='center')
    charts = {'Top Scorers': fig}

    # Chart 2: Position performance (multi-metric bar)
    fig, ax = plt.subplots(figsize=(10, 5))
    pos_stats = metrics['pos_stats']
    x = np.arange(len(pos_stats))
    width = 0.2
    ax.bar(x - width, pos_stats['points'], width, label='Points', color='coral')
    ax.bar(x, pos_stats['rebounds'], width, label='Rebounds', color='lightblue')
    ax.bar(x + width, pos_stats['assists'], width, label='Assists', color='lightgreen')
    ax.set_ylabel('Average Per Player')
    ax.set_title('Position Performance Profile', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pos_stats.index)
    ax.legend()
    charts['Position Stats'] = fig

    # Chart 3: Team total points (top 10)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_teams = metrics['team_stats'].head(10)
    ax.bar(range(len(top_teams)), top_teams['total_points'].values, color='darkgreen', alpha=0.7)
    ax.set_xticks(range(len(top_teams)))
    ax.set_xticklabels(top_teams.index, rotation=45, ha='right')
    ax.set_ylabel('Total Points')
    ax.set_title('Top 10 Teams by Total Points', fontsize=12, fontweight='bold')
    for i, v in enumerate(top_teams['total_points'].values):
        ax.text(i, v + 50, f'{v:.0f}', ha='center', fontweight='bold', fontsize=9)
    charts['Team Points'] = fig

    # Chart 4: PER distribution (histogram)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_metrics['PER'], bins=40, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=df_metrics['PER'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df_metrics["PER"].mean():.2f}')
    ax.set_xlabel('Player Efficiency Rating')
    ax.set_ylabel('Frequency')
    ax.set_title('PER Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    charts['PER Distribution'] = fig

    # Chart 5: Cluster scatter (points vs rebounds, colored by cluster)
    fig, ax = plt.subplots(figsize=(10, 5))
    cluster_colors = {0: '#ff6b6b', 1: '#4ecdc4', 2: '#ffe66d'}
    for cluster in sorted(df_metrics['cluster'].unique()):
        mask = df_metrics['cluster'] == cluster
        cluster_name = metrics['cluster_names'][cluster]
        ax.scatter(df_metrics[mask]['points'], df_metrics[mask]['rebounds'],
                   alpha=0.6, s=100, label=cluster_name, color=cluster_colors.get(cluster, 'gray'))
    ax.set_xlabel('Points Scored')
    ax.set_ylabel('Rebounds')
    ax.set_title('Player Clusters: Points vs Rebounds', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    charts['Player Clusters'] = fig

    # Chart 6: Efficiency scatter (points per turnover)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_eff = df_metrics.nlargest(20, 'efficiency')
    ax.barh(range(len(top_eff)), top_eff['efficiency'].values, color='teal')
    ax.set_yticks(range(len(top_eff)))
    ax.set_yticklabels(top_eff['player'].values, fontsize=8)
    ax.set_xlabel('Points per Turnover')
    ax.set_title('Top 20 by Efficiency (Points/Turnover)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    charts['Efficiency'] = fig

    # KPIs
    kpis = {
        'Avg Points': f"{df_metrics['points'].mean():.1f}",
        'Avg PER': f"{df_metrics['PER'].mean():.2f}",
        'Total Players': f"{len(df_metrics)}",
        'Avg Efficiency': f"{df_metrics['efficiency'].mean():.1f}",
    }

    os.makedirs('outputs', exist_ok=True)
    save_dashboard(charts, 'Sports Statistics Dashboard', 'outputs/dashboard.html', kpis=kpis)
