import pandas as pd
import numpy as np
import os
import requests

NBA_URL = "https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/nba.csv"

def load_data():
    """
    Load NBA player statistics from GitHub. Falls back to synthetic data if unavailable.
    """
    os.makedirs('data', exist_ok=True)

    # Try to fetch real NBA data
    try:
        r = requests.get(NBA_URL, timeout=8)
        if r.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]

            # Rename columns to standardized names
            rename_map = {
                'Name': 'player', 'Team': 'team', 'Pos': 'position', 'Age': 'age',
                'G': 'games', 'PTS': 'points', 'TRB': 'rebounds', 'AST': 'assists',
                'STL': 'steals', 'BLK': 'blocks', 'TOV': 'turnovers', 'MP': 'minutes',
                'FG%': 'fg_pct', '3P%': 'three_pct', 'FT%': 'ft_pct'
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

            needed = ['player', 'team', 'position', 'age', 'games', 'points',
                      'rebounds', 'assists', 'steals', 'blocks', 'turnovers', 'minutes', 'fg_pct']
            df = df[[c for c in needed if c in df.columns]].dropna()
            df = df[df['games'] >= 10]  # At least 10 games played

            df.to_csv('data/nba_stats.csv', index=False)
            print(f"  Loaded real NBA stats: {len(df)} players")
            return df
    except Exception as e:
        print(f"  Could not fetch NBA data ({type(e).__name__}), using synthetic")

    return _generate_synthetic(450)

def _generate_synthetic(n=450):
    """Generate synthetic NBA player statistics with real 2023-24 season characteristics."""
    rng = np.random.default_rng(42)

    # Real 2023-24 season top 30 players (representative stats)
    top_players = [
        {'name': 'Luka Dončić', 'team': 'DAL', 'pos': 'SF', 'pts': 33.9, 'reb': 9.2, 'ast': 9.8, 'fg': 0.485, 'age': 25, 'g': 70, 'stl': 1.3, 'blk': 0.8, 'tov': 3.2, 'min': 37.1},
        {'name': 'LeBron James', 'team': 'LAL', 'pos': 'SF', 'pts': 25.7, 'reb': 8.0, 'ast': 8.3, 'fg': 0.522, 'age': 39, 'g': 55, 'stl': 1.1, 'blk': 0.7, 'tov': 2.8, 'min': 35.2},
        {'name': 'Nikola Jokić', 'team': 'DEN', 'pos': 'C', 'pts': 24.5, 'reb': 11.8, 'ast': 9.0, 'fg': 0.643, 'age': 29, 'g': 76, 'stl': 1.0, 'blk': 0.9, 'tov': 2.5, 'min': 34.7},
        {'name': 'Stephen Curry', 'team': 'GSW', 'pos': 'PG', 'pts': 26.4, 'reb': 5.3, 'ast': 6.9, 'fg': 0.509, 'age': 36, 'g': 56, 'stl': 1.2, 'blk': 0.4, 'tov': 2.1, 'min': 33.1},
        {'name': 'Joel Embiid', 'team': 'PHI', 'pos': 'C', 'pts': 34.1, 'reb': 10.2, 'ast': 3.5, 'fg': 0.538, 'age': 30, 'g': 39, 'stl': 1.2, 'blk': 1.7, 'tov': 2.9, 'min': 34.1},
        {'name': 'Kevin Durant', 'team': 'PHX', 'pos': 'SF', 'pts': 29.1, 'reb': 6.7, 'ast': 5.0, 'fg': 0.571, 'age': 36, 'g': 75, 'stl': 0.9, 'blk': 1.1, 'tov': 2.4, 'min': 34.3},
        {'name': 'Jayson Tatum', 'team': 'BOS', 'pos': 'SF', 'pts': 26.9, 'reb': 8.8, 'ast': 4.6, 'fg': 0.476, 'age': 26, 'g': 74, 'stl': 1.0, 'blk': 0.8, 'tov': 2.7, 'min': 34.8},
        {'name': 'Giannis Antetokounmpo', 'team': 'MIL', 'pos': 'PF', 'pts': 30.4, 'reb': 11.5, 'ast': 5.9, 'fg': 0.589, 'age': 29, 'g': 60, 'stl': 1.1, 'blk': 0.8, 'tov': 2.8, 'min': 34.2},
        {'name': 'Damian Lillard', 'team': 'MIL', 'pos': 'PG', 'pts': 24.3, 'reb': 4.8, 'ast': 7.0, 'fg': 0.437, 'age': 34, 'g': 55, 'stl': 1.3, 'blk': 0.3, 'tov': 2.3, 'min': 33.5},
        {'name': 'Anthony Davis', 'team': 'LAL', 'pos': 'PF', 'pts': 25.9, 'reb': 12.5, 'ast': 2.6, 'fg': 0.563, 'age': 31, 'g': 76, 'stl': 0.8, 'blk': 1.3, 'tov': 2.2, 'min': 34.1},
        {'name': 'Shai Gilgeous-Alexander', 'team': 'OKC', 'pos': 'SG', 'pts': 30.1, 'reb': 4.8, 'ast': 6.2, 'fg': 0.521, 'age': 26, 'g': 79, 'stl': 1.8, 'blk': 0.8, 'tov': 2.4, 'min': 35.1},
        {'name': 'Donovan Mitchell', 'team': 'CLE', 'pos': 'SG', 'pts': 26.3, 'reb': 4.3, 'ast': 5.1, 'fg': 0.454, 'age': 28, 'g': 79, 'stl': 1.5, 'blk': 0.4, 'tov': 2.6, 'min': 34.6},
        {'name': 'Kawhi Leonard', 'team': 'LAC', 'pos': 'SF', 'pts': 23.8, 'reb': 5.9, 'ast': 3.9, 'fg': 0.502, 'age': 33, 'g': 68, 'stl': 1.9, 'blk': 0.6, 'tov': 2.1, 'min': 32.8},
        {'name': 'Devin Booker', 'team': 'PHX', 'pos': 'SG', 'pts': 27.1, 'reb': 4.5, 'ast': 6.9, 'fg': 0.488, 'age': 28, 'g': 58, 'stl': 1.2, 'blk': 0.3, 'tov': 2.5, 'min': 34.9},
        {'name': 'James Harden', 'team': 'LAC', 'pos': 'SG', 'pts': 23.4, 'reb': 5.1, 'ast': 8.9, 'fg': 0.442, 'age': 35, 'g': 55, 'stl': 1.7, 'blk': 0.5, 'tov': 2.8, 'min': 33.2},
        {'name': 'Tyrese Haliburton', 'team': 'IND', 'pos': 'PG', 'pts': 20.1, 'reb': 3.7, 'ast': 10.9, 'fg': 0.477, 'age': 24, 'g': 68, 'stl': 1.3, 'blk': 0.2, 'tov': 2.0, 'min': 32.4},
        {'name': 'Domantas Sabonis', 'team': 'SAC', 'pos': 'C', 'pts': 20.1, 'reb': 13.4, 'ast': 6.0, 'fg': 0.559, 'age': 29, 'g': 74, 'stl': 0.9, 'blk': 0.6, 'tov': 2.3, 'min': 32.9},
        {'name': 'Anfernee Simons', 'team': 'POR', 'pos': 'SG', 'pts': 22.3, 'reb': 3.9, 'ast': 5.1, 'fg': 0.428, 'age': 24, 'g': 72, 'stl': 1.1, 'blk': 0.3, 'tov': 2.2, 'min': 32.1},
        {'name': 'Mikal Bridges', 'team': 'BRK', 'pos': 'SF', 'pts': 20.1, 'reb': 5.7, 'ast': 3.4, 'fg': 0.508, 'age': 27, 'g': 77, 'stl': 1.4, 'blk': 0.5, 'tov': 1.9, 'min': 31.7},
        {'name': 'Paolo Banchero', 'team': 'ORL', 'pos': 'PF', 'pts': 23.1, 'reb': 8.8, 'ast': 3.7, 'fg': 0.466, 'age': 21, 'g': 67, 'stl': 0.9, 'blk': 0.7, 'tov': 2.4, 'min': 32.3},
        {'name': 'Franz Wagner', 'team': 'ORL', 'pos': 'SF', 'pts': 20.2, 'reb': 6.5, 'ast': 3.5, 'fg': 0.500, 'age': 22, 'g': 74, 'stl': 0.8, 'blk': 0.4, 'tov': 1.8, 'min': 30.9},
        {'name': 'Scottie Barnes', 'team': 'TOR', 'pos': 'SF', 'pts': 19.9, 'reb': 8.3, 'ast': 6.0, 'fg': 0.490, 'age': 23, 'g': 74, 'stl': 1.0, 'blk': 0.5, 'tov': 2.1, 'min': 31.4},
        {'name': 'Victor Wembanyama', 'team': 'SAS', 'pos': 'C', 'pts': 21.4, 'reb': 10.6, 'ast': 3.4, 'fg': 0.474, 'age': 20, 'g': 71, 'stl': 1.2, 'blk': 2.3, 'tov': 2.0, 'min': 31.8},
        {'name': 'Ja Morant', 'team': 'MEM', 'pos': 'PG', 'pts': 25.5, 'reb': 5.8, 'ast': 7.2, 'fg': 0.463, 'age': 25, 'g': 57, 'stl': 1.5, 'blk': 0.6, 'tov': 2.7, 'min': 33.7},
        {'name': 'Trae Young', 'team': 'ATL', 'pos': 'PG', 'pts': 26.2, 'reb': 3.1, 'ast': 10.2, 'fg': 0.435, 'age': 26, 'g': 79, 'stl': 1.2, 'blk': 0.2, 'tov': 3.1, 'min': 34.2},
        {'name': 'LaMelo Ball', 'team': 'CHA', 'pos': 'PG', 'pts': 23.3, 'reb': 5.6, 'ast': 8.4, 'fg': 0.440, 'age': 23, 'g': 60, 'stl': 1.4, 'blk': 0.4, 'tov': 2.8, 'min': 32.5},
        {'name': 'Jamal Murray', 'team': 'DEN', 'pos': 'PG', 'pts': 20.0, 'reb': 3.8, 'ast': 7.1, 'fg': 0.475, 'age': 27, 'g': 76, 'stl': 1.1, 'blk': 0.3, 'tov': 1.9, 'min': 30.6},
        {'name': 'Karl-Anthony Towns', 'team': 'MIN', 'pos': 'C', 'pts': 21.8, 'reb': 8.3, 'ast': 2.4, 'fg': 0.504, 'age': 29, 'g': 75, 'stl': 0.8, 'blk': 0.5, 'tov': 2.0, 'min': 31.2},
        {'name': 'Anthony Edwards', 'team': 'MIN', 'pos': 'SG', 'pts': 25.9, 'reb': 5.4, 'ast': 5.0, 'fg': 0.467, 'age': 23, 'g': 79, 'stl': 1.3, 'blk': 0.5, 'tov': 2.2, 'min': 32.8},
    ]

    real_df = pd.DataFrame(top_players)
    real_df = real_df.rename(columns={'pts': 'points', 'reb': 'rebounds', 'ast': 'assists',
                                       'fg': 'fg_pct', 'pos': 'position', 'g': 'games',
                                       'stl': 'steals', 'blk': 'blocks', 'tov': 'turnovers', 'min': 'minutes'})
    real_df['player'] = real_df['name']
    real_df = real_df[['player', 'team', 'position', 'age', 'games', 'points', 'rebounds', 'assists',
                       'steals', 'blocks', 'turnovers', 'minutes', 'fg_pct']]

    # Generate remaining synthetic players
    remaining = n - len(real_df)
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    teams = ['BOS', 'LAL', 'GSW', 'DEN', 'LAC', 'MIL', 'PHX', 'DAL', 'PHI', 'MIA',
             'CLE', 'OKC', 'SAC', 'MIN', 'NYK', 'CHI', 'TOR', 'ATL', 'MEM', 'ORL']

    # Position-based scoring/rebounding/assists distributions
    pos_stats = {
        'PG': {'pts': 16, 'reb': 4, 'ast': 7},
        'SG': {'pts': 15, 'reb': 4, 'ast': 4},
        'SF': {'pts': 14, 'reb': 6, 'ast': 3},
        'PF': {'pts': 13, 'reb': 8, 'ast': 2},
        'C': {'pts': 12, 'reb': 10, 'ast': 2},
    }

    pos_arr = rng.choice(positions, remaining)
    points = np.array([max(2, rng.normal(pos_stats[p]['pts'], 5)) for p in pos_arr]).round(1)
    rebounds = np.array([max(1, rng.normal(pos_stats[p]['reb'], 2.5)) for p in pos_arr]).round(1)
    assists = np.array([max(0.3, rng.normal(pos_stats[p]['ast'], 2)) for p in pos_arr]).round(1)
    steals = np.array([max(0.1, rng.normal(1.0, 0.4)) for _ in pos_arr]).round(1)
    blocks = np.array([max(0.1, rng.normal(0.7, 0.5)) for _ in pos_arr]).round(1)
    turnovers = np.array([max(0.5, rng.normal(2.0, 0.8)) for _ in pos_arr]).round(1)
    minutes = np.array([max(10, rng.normal(28, 6)) for _ in pos_arr]).round(1)
    fg_pct = rng.uniform(0.40, 0.55, remaining).round(3)

    synth_df = pd.DataFrame({
        'player': [f'Player_{i:03d}' for i in range(1, remaining + 1)],
        'team': rng.choice(teams, remaining),
        'position': pos_arr,
        'age': rng.integers(19, 38, remaining),
        'games': rng.integers(30, 82, remaining),
        'points': points,
        'rebounds': rebounds,
        'assists': assists,
        'steals': steals,
        'blocks': blocks,
        'turnovers': turnovers,
        'minutes': minutes,
        'fg_pct': fg_pct,
    })

    df = pd.concat([real_df, synth_df], ignore_index=True)
    df.to_csv('data/nba_stats.csv', index=False)
    print(f"  Generated {len(real_df)} real + {remaining} synthetic NBA player records ({n} total)")
    return df
