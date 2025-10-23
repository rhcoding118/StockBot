To operate paper mode:
python bot.py \
  --mode paper \
  --symbols AAPL MSFT \
  --model-path ./models/model.pt \
  --poll-seconds 30 \
  --units 5

  Note that AAPL and MSFT can be switched out for more options

# Sandbox (fake money)
python bot.py --mode live --symbols AAPL MSFT --units 1

# Production (REAL money)
export ETRADE_BASE_URL="https://api.etrade.com"
python bot.py --mode live --symbols AAPL --units 1

“The bot reads your E*TRADE API credentials from environment variables at startup. If any are missing, it will throw an error.”

To train model for different symbols:

python train_model.py \
  --symbols AAPL MSFT \
  --start 2016-01-01 --end 2024-12-31 \
  --epochs 8 \
  --out-path ./models/model.pt