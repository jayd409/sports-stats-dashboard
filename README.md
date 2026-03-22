# Sports Statistics Dashboard

Analyzes NBA player performance data using Player Efficiency Rating (PER) and K-means clustering to segment players into archetypes — providing position-level benchmarks and team comparisons.

## Business Question
How do NBA players compare on composite efficiency metrics, and what performance archetypes emerge across positions and teams?

## Key Findings
- NBA player stats analyzed with PER calculated across points, rebounds, assists, and defensive contributions
- K-means clustering (k=3) identifies Stars (top 12%), Contributors (38%), and Role Players (50%)
- Guards lead in assists/points; Centers dominate rebounding and efficiency-per-minute metrics
- Top scorers and top-efficiency players diverge significantly — high volume doesn't always equal high efficiency

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python3 main.py
```
Open `outputs/dashboard.html` in your browser. Also generates `outputs/player_analytics.csv`.

## Project Structure
- **src/data.py** - Loads real NBA stats from GitHub with fallback to synthetic player data
- **src/analysis.py** - PER computation, K-means clustering, position and team aggregations
- **src/charts.py** - Archetype scatter plots, position profiles, top player rankings
- **src/utils.py** - CSV export and dashboard HTML generation

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Author
Jay Desai · [jayd409@gmail.com](mailto:jayd409@gmail.com) · [Portfolio](https://jayd409.github.io)
