import asyncio
import io
import zipfile
from pathlib import Path
import httpx
import polars as pl
from tqdm.asyncio import tqdm

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT"
]
YEARS =[2023,2024]
MONTHS = range(1,13)

BASE_URL = "https://data.binance.vision/data"
RAW_DIR = Path("data/raw")

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]

def normalize_ts_ms(col_name:str)->pl.Expr:
    return(
        pl.when(pl.col(col_name)>10**14)
        .then(pl.col(col_name)//1000)
        .otherwise(pl.col(col_name))
    )

async def fetch_zip_csv(client:httpx.AsyncClient,url:str,sem:asyncio.Semaphore):
    async with sem:
        for attempt in range(3):
            try:
                resp=await client.get(url,timeout=30.0)

                if resp.status_code==200:
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                        csv_name =z.namelist()[0]
                        return z.read(csv_name)
                elif resp.status_code ==404:
                    return None
            except Exception:
                await asyncio.sleep(1.0*(attempt+1))
        return None

def parse_klines(csv_bytes:bytes,symbol:str,prefix:str)->pl.DataFrame:
    first_byte = csv_bytes[:1]
    has_header = not first_byte.isdigit()

    df = pl.read_csv(
        io.BytesIO(csv_bytes),
        has_header=has_header,
        new_columns=None if has_header else KLINE_COLS
    )

    if has_header:
        df=df.rename(dict(zip(df.columns,KLINE_COLS)))

    return (
        df.select([
            normalize_ts_ms("open_time").alias("open_time_ms"),
            pl.col("open").cast(pl.Float64).alias(f"{prefix}_open"),
            pl.col("high").cast(pl.Float64).alias(f"{prefix}_high"),
            pl.col("low").cast(pl.Float64).alias(f"{prefix}_low"),
            pl.col("close").cast(pl.Float64).alias(f"{prefix}_close"),
            pl.col("quote_volume").cast(pl.Float64).alias(f"{prefix}_quote_vol"),
            pl.col("count").cast(pl.Float64).alias(f"{prefix}_trade_count"),
            pl.col("taker_buy_quote_volume").cast(pl.Float64).alias(f"{prefix}_taker_buy_quote_vol"),
        ]).with_columns(pl.from_epoch(pl.col("open_time_ms") + 3600_000, time_unit="ms").alias("timestamp"),
            pl.lit(symbol).alias("symbol")
        ).drop("open_time_ms")
        )

def parse_funding(csv_bytes: bytes, symbol: str) -> pl.DataFrame:
    df=pl.read_csv(io.BytesIO(csv_bytes),has_header=True)
    ts_col="calc_time" if "calc_time" in df.columns else df.columns[0]
    rate_col = "last_funding_rate" if "last_funding_rate" in df.columns else df.columns[2]
    return (
        df.select([
            normalize_ts_ms(ts_col).alias("ts_ms"),
            pl.col(rate_col).cast(pl.Float64).alias("funding_rate")
                ]).with_columns(
            pl.from_epoch(
                # calc_time is a settlement time, not a candle open time.
                # Archives can record it a few milliseconds after the hour.
                ((pl.col("ts_ms") + 1_800_000) // 3_600_000) * 3_600_000,
                time_unit="ms"
            ).alias("timestamp"),
            pl.lit(symbol).alias("symbol")
        ).drop("ts_ms")
    )

def align_funding(panel: pl.DataFrame, df_fund: pl.DataFrame) -> pl.DataFrame:
    keys = ["timestamp", "symbol"]
    df_fund = df_fund.unique(keys).sort(["symbol", "timestamp"])
    panel = panel.sort(["symbol", "timestamp"])
    missing_symbols = set(panel["symbol"].unique()) - set(df_fund["symbol"].unique())
    if missing_symbols:
        raise RuntimeError(f"Missing funding data for {sorted(missing_symbols)}")

    panel = panel.join(df_fund, on=keys, how="left").sort(["symbol", "timestamp"])
    unmatched = (
        panel.group_by("symbol")
        .agg(pl.col("funding_rate").count().alias("matches"))
        .filter(pl.col("matches") == 0)
    )
    if unmatched.height:
        raise RuntimeError(f"No funding timestamps matched: {unmatched['symbol'].to_list()}")

    # Use the original funding timeline so the first panel hours can inherit
    # a settlement before the panel starts. Only look backwards within a symbol.
    return (
        panel.join_asof(
            df_fund.rename({"funding_rate": "funding_rate_ffill"}),
            on="timestamp",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(
            pl.col("funding_rate").fill_null(0.0).alias("funding_payment_rate")
        )
    )

async def download_all():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(16)
    tasks = []

    async with httpx.AsyncClient() as client:
        for sym in SYMBOLS:
            for y in YEARS:
                for m in MONTHS:
                    ym = f"{y}-{m:02d}"
                    u_perp = f"{BASE_URL}/futures/um/monthly/klines/{sym}/1h/{sym}-1h-{ym}.zip"
                    u_spot = f"{BASE_URL}/spot/monthly/klines/{sym}/1h/{sym}-1h-{ym}.zip"
                    u_fund = f"{BASE_URL}/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip"
                    tasks.append((sym, "perp", fetch_zip_csv(client, u_perp, sem)))
                    tasks.append((sym, "spot", fetch_zip_csv(client, u_spot, sem)))
                    tasks.append((sym, "fund", fetch_zip_csv(client, u_fund, sem)))
        coros = [t[2] for t in tasks]
        results = await tqdm.gather(*coros, desc="Downloading Binance Archives")

    perp_dfs, spot_dfs, fund_dfs = [], [], []
    for (sym, dtype, _), raw_bytes in zip(tasks, results):
        if raw_bytes is None:
            if dtype == "fund":
                raise RuntimeError(f"Funding archive missing for {sym}; refusing to fill download gaps with zeros")
            continue
        if dtype == "perp":
            perp_dfs.append(parse_klines(raw_bytes, sym, "perp"))
        elif dtype == "spot":
            spot_dfs.append(parse_klines(raw_bytes, sym, "spot"))
        elif dtype == "fund":
            fund_dfs.append(parse_funding(raw_bytes, sym))
    for name, frames in {
        "perp": perp_dfs,
        "spot": spot_dfs,
        "fund": fund_dfs,
    }.items():
        if not frames:
            raise RuntimeError(
                f"{name} doesn't have data。"
            )

    df_perp = pl.concat(perp_dfs).unique(["timestamp", "symbol"]).sort(["timestamp", "symbol"])
    df_spot = pl.concat(spot_dfs).unique(["timestamp", "symbol"]).sort(["timestamp", "symbol"])
    df_fund = pl.concat(fund_dfs).unique(["timestamp", "symbol"]).sort(["timestamp", "symbol"])

    panel = align_funding(
        df_perp
        .join(df_spot, on=["timestamp", "symbol"], how="inner"),
        df_fund,
    )
    print(panel.group_by("symbol").agg(
        pl.len().alias("hours"),
        pl.col("funding_rate").count().alias("funding_events"),
    ).sort("symbol"))

    out_path = Path("data/processed/panel_1h_raw.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(out_path)
    print(f"Saved aligned panel: {panel.shape} to {out_path}")

if __name__ == "__main__":
    asyncio.run(download_all())
