name: Fetch RTMS Data

on:
  workflow_dispatch:
  schedule:
    - cron: '0 19 * * *'  # 매일 04:00 KST(=19:00 UTC)

jobs:
  fetch:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - name: Run ETL
        env:
          MOLIT_API_KEY: ${{ secrets.MOLIT_API_KEY }}
        run: |
          python etl_fetch.py --months 12
      - name: Commit & Push
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -A
          git commit -m "Update data (auto)" || echo "No changes"
          git push
