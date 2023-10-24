import numpy as np
import pandas as pd


def ema(close, period):
    alpha = 2 / (period + 1)
    ema_calc = np.zeros_like(close)
    ema_calc[0] = float(close[0])
    for i in range(1, close.shape[0]):
        ema_calc[i] = alpha * float(close[i]) + (1 - alpha) * float(ema_calc[i-1])
    return ema_calc.astype(float)


def sma(klines, period):
    sma_calc = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    return sma_calc['close'].rolling(window=period).mean()


def rsi(klines, period):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data = data.astype(float)
    delta = data['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    gain = up.rolling(window=period).mean()
    loss = abs(down.rolling(window=period).mean())
    rs = gain / loss
    rsi_calc = 100.0 - (100.0 / (1.0 + rs))
    return rsi_calc


def macd(klines, fast_period=12, slow_period=26, signal_period=9):
    klines = np.array(klines)
    close = klines[:, 4]
    ema_fast = ema(close, fast_period)
    ema_slow = ema(close, slow_period)
    macd_calc = ema_fast - ema_slow
    signal = ema(macd_calc, signal_period)
    hist = macd_calc - signal
    return macd_calc, signal, hist


def bollinger_bands(klines, period):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    sma_calc = sma(klines, period)
    std = data['close'].rolling(window=period).std()
    upper_band = sma_calc + 2 * std
    lower_band = sma_calc - 2 * std
    return lower_band, upper_band


def stochastic_oscillator(klines, k, d):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    low_min = data['low'].rolling(window=k).min()
    high_max = data['high'].rolling(window=k).max()
    k_percent = 100 * (data['close'] - low_min) / (high_max - low_min)
    d_percent = k_percent.rolling(window=d).mean()
    return k_percent, d_percent


def adx(klines, period):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    data['high-low'] = abs(data['high'] - data['low'])
    data['high-pc'] = abs(data['high'] - data['close'].shift(1))
    data['low-pc'] = abs(data['low'] - data['close'].shift(1))
    tr = data[['high-low', 'high-pc', 'low-pc']].max(axis=1)
    data['tr'] = tr
    data['+DM'] = (data['high'] - data['high'].shift(1)).apply(lambda x: x if x > 0 else 0)
    data['-DM'] = (data['low'].shift(1) - data['low']).apply(lambda x: x if x > 0 else 0)
    atr = tr.rolling(window=period).mean()
    data['atr'] = atr
    di_plus = 100 * (data['+DM'] / atr).rolling(window=period).mean()
    di_minus = 100 * (data['-DM'] / atr).rolling(window=period).mean()
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_calc = dx.rolling(window=period).mean()
    return adx_calc


def obv(klines):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    obv_calc = []
    last_obv = 0
    for i in range(1, len(data)):
        if data['close'][i] > data['close'][i - 1]:
            current_obv = last_obv + data['volume'][i]
        elif data['close'][i] < data['close'][i - 1]:
            current_obv = last_obv - data['volume'][i]
        else:
            current_obv = last_obv
        obv_calc.append(current_obv)
        last_obv = current_obv
    data['obv'] = [0] + obv_calc
    return data['obv']


def cci(klines, period):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    sma_calc = typical_price.rolling(window=period).mean()
    mean_deviation = (typical_price - sma_calc).abs().rolling(window=period).mean()
    cci_calc = (typical_price - sma_calc) / (0.015 * mean_deviation)
    return cci_calc


def mfi(klines, period):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    positive_mf = money_flow[data['close'] > data['close'].shift(1)].rolling(window=period).sum()
    negative_mf = money_flow[data['close'] < data['close'].shift(1)].rolling(window=period).sum()
    money_flow_ratio = positive_mf / negative_mf
    mfi_calc = 100 - 100 / (1 + money_flow_ratio)
    return mfi_calc


def ichimoku_cloud(klines, conversion_period, base_period, span_period, displacement):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    data['conversion_line'] = (data['high'].rolling(window=conversion_period).max() + data['low'].rolling(
        window=conversion_period).min()) / 2
    data['base_line'] = (data['high'].rolling(window=base_period).max() + data['low'].rolling(
        window=base_period).min()) / 2
    data['span_a'] = (data['conversion_line'] + data['base_line']) / 2
    data['span_b'] = (data['high'].rolling(window=span_period).max() + data['low'].rolling(
        window=span_period).min()) / 2
    data['span_a'] = data['span_a'].shift(displacement)
    data['span_b'] = data['span_b'].shift(displacement)
    return data


def fibonacci_retracement(klines):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    max_price = max(data['high'])
    min_price = min(data['low'])
    diff = max_price - min_price
    levels = {
        "23.6%": max_price - (diff * 0.236),
        "38.2%": max_price - (diff * 0.382),
        "50%": max_price - (diff * 0.5),
        "61.8%": max_price - (diff * 0.618),
        "100%": max_price
    }
    return levels


def parabolic_sar(klines, acceleration_factor_initial=0.02, acceleration_factor_step=0.02, max_acceleration_factor=0.2):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    high = data['high']
    low = data['low']
    psar = [None] * len(high)
    trend = [None] * len(high)
    acceleration_factor = acceleration_factor_initial
    extreme_point = high[0]
    psar[0] = low[0] - acceleration_factor * (high[0] - low[0])
    trend[0] = 1 if high[1] > high[0] else -1
    for i in range(1, len(high)):
        if trend[i - 1] == 1:
            if low[i] < psar[i - 1]:
                trend[i] = -1
                psar[i] = extreme_point
                acceleration_factor = acceleration_factor_initial
                extreme_point = high[i]
            else:
                trend[i] = 1
                psar[i] = psar[i - 1] + acceleration_factor * (high[i - 1] - psar[i - 1])
                if high[i] > extreme_point:
                    extreme_point = high[i]
                    acceleration_factor = min(acceleration_factor + acceleration_factor_step, max_acceleration_factor)
        else:
            if high[i] > psar[i - 1]:
                trend[i] = 1
                psar[i] = extreme_point
                acceleration_factor = acceleration_factor_initial
                extreme_point = low[i]
            else:
                trend[i] = -1
                psar[i] = psar[i - 1] - acceleration_factor * (psar[i - 1] - low[i - 1])
                if low[i] < extreme_point:
                    extreme_point = low[i]
                    acceleration_factor = min(acceleration_factor + acceleration_factor_step, max_acceleration_factor)
    return psar, trend


def atr(klines, period):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    high = data['high']
    low = data['low']
    close = data['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_calc = true_range.rolling(window=period).mean()
    return atr_calc


def chaikin_money_flow(klines, period):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    mf_volume = mfm * data['volume']
    cmf = mf_volume.rolling(period).sum() / data['volume'].rolling(period).sum()
    return cmf


def keltner_channel(klines, period, factor=2.0):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    ema_calc = data['close'].ewm(span=period, min_periods=period).mean()
    atr_calc = pd.DataFrame(abs(data['high'] - data['low']))
    atr_calc.columns = ['TR']
    atr_calc['ATR'] = atr_calc['TR'].rolling(period).mean()
    upper_band = ema_calc + (atr_calc['ATR'] * factor)
    lower_band = ema_calc - (atr_calc['ATR'] * factor)
    return ema_calc, upper_band, lower_band


def donchain_channel(klines, period):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    high = data['high']
    low = data['low']
    high_max = high.rolling(period, min_periods=1).max()
    low_min = low.rolling(period, min_periods=1).min()
    upper_band = high_max.rolling(period, min_periods=1).max()
    lower_band = low_min.rolling(period, min_periods=1).min()
    return upper_band, lower_band


def williams_r(klines, period):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    high = data['high']
    low = data['low']
    close = data['close']
    hh = np.maximum.accumulate(high)
    ll = np.minimum.accumulate(low)
    wr = -100 * (hh - close) / (hh - ll)
    wr = wr[-period:]
    return wr


def vwap(klines):
    data = pd.DataFrame(klines, columns=['open_time', "open", "high", "low", "close", "volume", "close_time"])
    data.astype(float)
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    total_volume = data['volume'].sum()
    sum_price_volume = (typical_price * data['volume']).sum()
    vwap_calc = sum_price_volume / total_volume
    return vwap_calc


def stddev(klines, period):
    data = np.array(klines)
    mean = np.mean(data)
    deviations = data - mean
    square_deviations = deviations ** 2
    moving_average_of_square_deviations = np.convolve(square_deviations, np.ones(period) / period, mode='valid')
    std = np.sqrt(moving_average_of_square_deviations)
    return std
