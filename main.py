#!/usr/bin/env python3
import sys, os
sys.path.insert(0, 'src')

from data_loader import load_data
from database import save_to_db, query
from analytics import compute_all_metrics
from visualizer import build_dashboard

def main():
    print("Sports Statistics Dashboard")
    print("=" * 50)

    print("\nLoading NBA player data...")
    df = load_data()

    print("Computing analytics...")
    metrics = compute_all_metrics(df)

    save_to_db(metrics['df_metrics'], 'players')

    # Print summary statistics
    print(f"\n  Dataset Summary    :")
    print(f"    Total Players   : {len(df):,}")
    print(f"    Total Teams     : {df['team'].nunique()}")
    print(f"    Avg Points      : {metrics['df_metrics']['points'].mean():.1f}")
    print(f"    Avg PER         : {metrics['df_metrics']['PER'].mean():.2f}")

    print(f"\n  Top 5 by PER (Player Efficiency Rating) :")
    for idx, row in metrics['top_per'].iterrows():
        print(f"    {idx+1}. {row['player']:20s} (PER: {row['PER']:.2f})")

    print("\nBuilding dashboard...")
    build_dashboard(metrics)

    # Save detailed metrics
    metrics['df_metrics'].to_csv('outputs/player_analytics.csv', index=False)
    print(f"  Analytics saved → outputs/player_analytics.csv")

    # --- SQL Analytics (SQLite) ---
    print("\n--- SQL Analytics (SQLite) ---")

    # Query 1: Top 10 players by PER
    print("\n1. Top 10 Players by PER (Player Efficiency Rating):")
    sql_per = """
    SELECT player, team, position, points, rebounds, assists, PER
    FROM players
    ORDER BY PER DESC
    LIMIT 10
    """
    result_per = query(sql_per)
    print(result_per.to_string(index=False))

    # Query 2: Team stats summary
    print("\n2. Team Stats Summary (Average per player):")
    sql_teams = """
    SELECT team, COUNT(*) as num_players,
           ROUND(AVG(points), 1) as avg_points,
           ROUND(AVG(rebounds), 1) as avg_rebounds,
           ROUND(AVG(assists), 1) as avg_assists,
           ROUND(AVG(PER), 2) as avg_PER
    FROM players
    GROUP BY team
    ORDER BY avg_points DESC
    LIMIT 15
    """
    result_teams = query(sql_teams)
    print(result_teams.to_string(index=False))

    # Query 3: Position breakdown with avg efficiency
    print("\n3. Position Breakdown (Average Efficiency):")
    sql_pos = """
    SELECT position, COUNT(*) as player_count,
           ROUND(AVG(points), 1) as avg_points,
           ROUND(AVG(rebounds), 1) as avg_rebounds,
           ROUND(AVG(assists), 1) as avg_assists,
           ROUND(AVG(PER), 2) as avg_PER
    FROM players
    GROUP BY position
    ORDER BY avg_PER DESC
    """
    result_pos = query(sql_pos)
    print(result_pos.to_string(index=False))

    print("\nDone. Open outputs/dashboard.html to view.")

if __name__ == "__main__":
    main()
