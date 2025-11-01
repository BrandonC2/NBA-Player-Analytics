# PickPrizes

## Purpose:
This program gathers NBA player data as well as OVER/UNDER options from PrizePicks
to provide the best lineup (flex or power) based on EV.




1) Create a `.env` and set `THE_ODDS_API_KEY`.
2) `pip install -r requirements.txt`
3) Run a quick test:



# Pull latest PrizePicks lines
python -m src.prizepicks_connector --league NBA --out data/prizepicks_live.csv

# Build the live probability + EV board
python -m src.build_board_all \
  --date $(date +%Y-%m-%d) \
  --season 2025-26 \
  --use_prizepicks 1 \
  --merge_mode replace \
  --out board.csv

# Generate top Flex/Power combinations
python -m src.build_entries --board board.csv --entry_type flex5 --min_p 0.58 --max_candidates 40 --top_n 100 --out entries_flex5.csv

# Streamlit Dashboard
streamlit run src/dashboard.py