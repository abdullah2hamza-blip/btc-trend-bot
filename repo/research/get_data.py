"""Download BTCUSDT hourly candles (2018 -> last full month) from Binance's public archive into data/btc_1h.pkl."""
import io, os, zipfile, urllib.request
from datetime import date
import pandas as pd
out = os.path.join(os.path.dirname(__file__), "data"); os.makedirs(out, exist_ok=True)
frames, today = [], date.today()
for y in range(2018, today.year + 1):
    for m in range(1, 13):
        if (y, m) >= (today.year, today.month):
            break
        url = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{y}-{m:02d}.zip"
        try:
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read()))
            frames.append(pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=range(6)))
            print("got", y, m)
        except Exception as e:
            print("skip", y, m, e)
df = pd.concat(frames); df.columns = ["ts", "open", "high", "low", "close", "volume"]
df["ts"] = pd.to_datetime(df.ts.where(df.ts < 1e14, df.ts // 1000), unit="ms", utc=True)
df.drop_duplicates("ts").set_index("ts").sort_index().to_pickle(os.path.join(out, "btc_1h.pkl"))
print("saved", len(df), "bars")
