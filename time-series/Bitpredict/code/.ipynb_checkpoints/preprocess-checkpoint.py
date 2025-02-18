# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../LTCUSDT.csv", parse_dates=["Date"])
ltc_df = pd.DataFrame(df[["Date", "Close"]])
# %%
plt.plot(ltc_df["Date"], ltc_df["Close"])
plt.ylabel("LTC_PRICE")
plt.title("LTC Price")
