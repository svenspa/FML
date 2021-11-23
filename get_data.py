from typing import List
import pandas as pd
import yfinance as yf

def get_option_data(
    df: pd.DataFrame,
    option_ids: List[str],
    yf_ticker: str = "^OEX",
    drop_not_traded: bool = True,
    save_dfs: bool = False,
) -> dict:
    d = {}
    ticker = yf.Ticker(yf_ticker)
    for option_id in option_ids:

        option_df = df[df.loc[:, "optionid"] == option_id]

        if drop_not_traded:
            option_df.dropna(inplace=True, subset=["last_date"])

        option_df.loc[:, "date"] = pd.to_datetime(
            option_df.loc[:, "date"], format="%Y%m%d"
        )

        underlying_df = ticker.history(
            start=option_df.loc[:, "date"].iloc[0] + pd.Timedelta('1 days'),
            end=option_df.loc[:, "date"].iloc[-1] + pd.Timedelta('1 days'),
        )

        assert option_df.shape[0] == underlying_df.shape[0]
        option_df = pd.merge(option_df,underlying_df[['Close']],left_on='date', right_index=True)

        d[option_id] = option_df

        if save_dfs:
            path = "data/" + str(option_id) + ".csv"
            option_df.to_csv(path)

    return d
