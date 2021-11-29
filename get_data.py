from typing import List
import pandas as pd
import numpy as np
import yfinance as yf
import warnings


def get_option_data(
    df: pd.DataFrame,
    option_ids: List[int],
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
            start=option_df.loc[:, "date"].iloc[0] + pd.Timedelta("1 days"),
            end=option_df.loc[:, "date"].iloc[-1] + pd.Timedelta("1 days"),
        )

        if option_df.shape[0] != underlying_df.shape[0]:
            warnings.warn(
                f"option_df and underlying_df do not match. \n The shapes are {option_df.shape=} and {underlying_df.shape=}"
            )
            continue

        option_df = pd.merge(
            option_df, underlying_df[["Close"]], left_on="date", right_index=True
        )
        option_df = option_df.rename(columns={"Close": "S"})

        d[option_id] = option_df

        if save_dfs:
            path = "data/" + str(option_id) + ".csv"
            option_df.to_csv(path)

    return d


def remove_until_first_nonzero(data, columnName, minV):
    data.loc[:, "indicator"] = 0
    data.loc[data[columnName] > minV, "indicator"] = 1
    data["indicator"] = (
        data.sort_values(by="date").groupby("optionid").indicator.cumsum()
    )
    data = data[data["indicator"] > 0]
    return data.drop(columns="indicator")


def min_n_days(data, nMIN, nMAX):
    keep = (data.groupby("optionid").volume.count() >= nMIN) & (data.groupby("optionid").volume.count() <= nMAX)
    return data[data.optionid.isin(keep[keep == True].index)]


def get_within_date(data, start, end):
    data["exdate"] = pd.to_datetime(data["date"], format="%Y%m%d")
    optionids = np.unique(
        data[
            (data["exdate"] >= pd.to_datetime("2018-01-01"))
            & (data["exdate"] <= pd.to_datetime("2019-01-01"))
        ]["optionid"].values
    )
    return data[data["optionid"].isin(optionids)]

def get_best_options(data, start, end, nMIN, nMAX, minV):
    data = get_within_date(data, start, end)
    data = remove_until_first_nonzero(data, "volume", minV)
    data = data[data["cp_flag"] == "C"]
    data = min_n_days(data, nMIN, nMAX)

    return (
        data,
        (
            data[data["volume"] == 0].groupby("optionid").volume.count().sort_index()
            / data.groupby("optionid").volume.count().sort_index()
        ).sort_values(),
    )
