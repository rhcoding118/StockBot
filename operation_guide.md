To operate paper mode:
python bot.py \
  --mode paper \
  --symbols AAPL MSFT \
  --model-path ./models/model.pt \
  --poll-seconds 30 \
  --units 5

  Note that AAPL and MSFT can be switched out for more options

# Sandbox (fake money)
export ETRADE_CONSUMER_KEY=...
export ETRADE_CONSUMER_SECRET=...
export ETRADE_OAUTH_TOKEN=...
export ETRADE_OAUTH_TOKEN_SECRET=...
export ETRADE_BASE_URL="https://apisb.etrade.com"
python bot.py --mode live --symbols AAPL MSFT --units 1

# Production (REAL money)
export ETRADE_BASE_URL="https://api.etrade.com"
python bot.py --mode live --symbols AAPL --units 1

The bot asks for api stuff at the start, otherwise, it will throw an error.

