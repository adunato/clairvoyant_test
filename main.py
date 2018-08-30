from clairvoyant.engine import Backtest
import pandas as pd
import bulbea as bb
import os
import talib
os.environ["BULBEA_QUANDL_API_KEY"] = 'VGjjpcct_DscJ4Fa8DCP'


features  = ["SMA", "RSI"]   # Financial indicators of choice
trainStart = 0               # Start of training period
trainEnd   = 1000            # End of training period
testStart  = 1001             # Start of testing period
testEnd    = 5000            # End of testing period
buyThreshold  = 0.1         # Confidence threshold for predicting buy (default = 0.65)
sellThreshold = 0.65         # Confidence threshold for predicting sell (default = 0.65)
continuedTraining = False    # Continue training during testing period? (default = false)

# Initialize backtester
backtest = Backtest(features, trainStart, trainEnd, testStart, testEnd, buyThreshold, sellThreshold, continuedTraining)

# gets share data
provider = 'alphavantage'
share = bb.Share(source='SSE', ticker='MSFT', provider=provider)
df = pd.DataFrame(share.data)
df = df.transpose()
df.columns = ['high','low','open','dropme1','close','dropme2', 'dropme3', 'dropme4']
df = df.drop(['dropme1','dropme2','dropme3', 'dropme4'], axis=1)
df['date'] = pd.to_datetime(df.index, infer_datetime_format=True)
df['high'] = pd.to_numeric(df['high'])
df['low'] = pd.to_numeric(df['low'])
df['open'] = pd.to_numeric(df['open'])
df['close'] = pd.to_numeric(df['close'])
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df['SMA'] = talib.SMA(df["close"])
df['RSI'] = talib.RSI(df["close"])
# df['MACD'] = talib.MACD(df["close"])[0]
df = df.dropna()

# A little bit of pre-processing
# data = pd.read_csv("SBUX.csv", date_parser=['date'])
data = df.round(3)

# Start backtesting and optionally modify SVC parameters
# Available paramaters can be found at: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
backtest.start(data, kernel='rbf', C=1, gamma=10)
backtest.conditions()
backtest.statistics()
backtest.visualize('MSFT')