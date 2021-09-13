#! /usr/bin/env python3

import time
import numpy as np

""" Base file used for my crypto  trading Algorithms, Indicators and formatting. """

"""
[######################################################################################################################]
[#############################################] TAS & INCICATORS SECTION [##############################################]
[#############################################]v^v^v^v^v^v^vv^v^v^v^v^v^v[##############################################]

### Indicators List ###
- BB
- RSI
- StochRSI
- Stochastic Oscillator
- SMA
- EMA
- SS
- MACD
- TR
- ATR
- DM
- ADX_DI
- Ichimoku
"""


## This function is used to calculate and return the Bollinger Band indicator.
def get_BOLL(prices, time_values=None, ma_type=21, stDev=2, map_time=False):
    """
    This function uses 2 parameters to calculate the BB-
    
    [PARAMETERS]
        prices  : A list of prices.
        ma_type : BB ma type.
    
    [CALCULATION]
        ---
    
    [RETURN]
        [{
        'M':float,
        'T':float,
        'B':float
        }, ... ]
    """
    span = len(prices)-ma_type
    stdev = np.array([np.std(prices[i:(ma_type+i)+1]) for i in range(span)])
    sma = get_SMA(prices, ma_type)

    BTop = np.array([sma[i] + (stdev[i] * stDev) for i in range(span)])
    BBot = np.array([sma[i] - (stdev[i] * stDev) for i in range(span)])

    boll = [{
        "T":BTop[i], 
        "M":sma[i], 
        "B":BBot[i]} for i in range(span)]

    if map_time:
       boll = [ [ time_values[i], boll[i] ] for i in range(len(boll)) ]

    return boll


## This function is used to calculate and return the RSI indicator.
def get_RSI(prices, time_values=None, rsiType=14, map_time=False):
    """ 
    This function uses 2 parameters to calculate the RSI-
    
    [PARAMETERS]
        prices  : The prices to be used.
        rsiType : The interval type.
    
    [CALCULATION]
        ---
    
    [RETURN]
        [
        float,
        float,
        ... ]
    """
    prices  = np.flipud(np.array(prices))
    deltas  = np.diff(prices)
    rsi     = np.zeros_like(prices)
    seed    = deltas[:rsiType+1]
    up      = seed[seed>=0].sum()/rsiType
    down    = abs(seed[seed<0].sum()/rsiType)
    rs      = up/down
    rsi[-1] = 100 - 100 /(1+rs)

    for i in range(rsiType, len(prices)):
        cDeltas = deltas[i-1]

        if cDeltas > 0:
            upVal = cDeltas
            downVal = 0
        else:
            upVal = 0
            downVal = abs(cDeltas)

        up = (up*(rsiType-1)+upVal)/rsiType
        down = (down*(rsiType-1)+downVal)/rsiType

        rs = up/down
        rsi[i] = 100 - 100 /(1+rs)

    fRSI = np.flipud(np.array(rsi[rsiType:]))

    fRSI.round(2)

    if map_time:
       fRSI = [ [ time_values[i], fRSI[i] ] for i in range(len(fRSI)) ]

    return fRSI


def get_stochastics(priceClose, priceHigh, priceLow, period=14):

    span = len(priceClose)-period
    stochastic = np.array([[priceHigh[i:period+i].max()-priceLow[i:period+i].min(), priceClose[i]-priceLow[i:period+i].min()] for i in range(span)])

    return stochastic


## This function is used to calculate and return the stochastics RSI indicator.
def get_stochRSI(prices, time_values=None, rsiPrim=14, rsiSecon=14, K=3, D=3, map_time=False):
    """
    This function uses 3 parameters to calculate the  Stochastics RSI-
    
    [PARAMETERS]
        prices  : A list of prices.
        rsiType : The interval type.
        ind_span: The span of the indicator.
    
    [CALCULATION]
        ---
    
    [RETURN]
        [{
        "%K":float,
        "%D":float
        }, ... ]
    """
    span = len(prices)-rsiPrim-rsiSecon-K
    RSI = get_RSI(prices, rsiType=rsiPrim)

    stoch_rsi = get_S_O(RSI, RSI, RSI, time_values=time_values, period=rsiSecon, K=K, D=D, map_time=map_time)
    
    return stoch_rsi


## This function is used to calculate and return the Stochastic Oscillator indicator.
def get_S_O(priceClose, priceHigh, priceLow, time_values=None, period=14, K=3, D=3, map_time=False):
    """
    This function uses 5 parameters to calculate the  Stochastic Oscillator-
    
    [PARAMETERS]
        candles : A list of prices.
        K_period : The interval for the inital K calculation.
        K_smooth: The smooting interval or the K period.
        D_smooth: The smooting interval or the K period.
        ind_span: The span of the indicator.
    
    [CALCULATION]
        ---
    
    [RETURN]
        [{
        "%K":float,
        "%D":float
        }, ... ]
    """
    priceClose = np.array(priceClose)
    priceHigh = np.array(priceHigh)
    priceLow = np.array(priceLow)

    span = len(priceClose)-period
    stochastic = get_stochastics(priceClose, priceHigh, priceLow, period)

    HL_CL = np.array([((stochastic[:, 1][i] / stochastic[:, 0][i]) * 100) for i in range(span)])

    sto_K = get_SMA(HL_CL, K)
    sto_D = get_SMA(sto_K, D)

    stoc_osc = [{
        "%K":sto_K[i], 
        "%D":sto_D[i]} for i in range(len(sto_D))]

    if map_time:
       stoc_osc = [ [ time_values[i], stoc_osc[i] ] for i in range(len(stoc_osc)) ]

    return stoc_osc


## This function is used to calculate and return SMA.
def get_SMA(prices, maPeriod, time_values=None, prec=8, map_time=False, result_format='normal'):
    """
    This function uses 3 parameters to calculate the Simple Moving Average-
    
    [PARAMETERS]
        prices  : A list of prices.
        ma_type : The interval type.
        ind_span: The span of the indicator.
    
    [CALCULATION]
        SMA = average of prices within a given period
    
    [RETURN]
        [
        float,
        float,
        ... ]
    """
    span = len(prices) - maPeriod + 1
    ma_list = np.array([np.mean(prices[i:(maPeriod+i)]) for i in range(span)])

    return_vals = ma_list.round(prec)

    if result_format == 'normal':
        return_vals = [ val for val in return_vals ]

    if map_time:
       return_vals = [ [ time_values[i], return_vals[i] ] for i in range(len(return_vals)) ]

    return return_vals


## This function is used to calculate and return EMA.
def get_EMA(prices, maPeriod, time_values=None, prec=8, map_time=False, result_format='normal'):
    """
    This function uses 3 parameters to calculate the Exponential Moving Average-
    
    [PARAMETERS]
        prices  : A list of prices.
        ma_type : The interval type.
        ind_span: The span of the indicator.
    
    [CALCULATION]
        weight = 2 / (maPerido + 1)
        EMA = ((close - prevEMA) * weight + prevEMA)
    
    [RETURN]
        [
        float,
        float,
        ... ]
    """
    span = len(prices) - maPeriod
    EMA = np.zeros_like(prices[:span])
    weight = (2 / (maPeriod +1))
    SMA = get_SMA(prices[span:], maPeriod, result_format='numpy')
    seed = SMA + weight * (prices[span-1] - SMA)
    EMA[0] = seed

    for i in range(1, span):
        EMA[i] = (EMA[i-1] + weight * (prices[span-i-1] - EMA[i-1]))

    return_vals = np.flipud(EMA.round(prec))

    if result_format == 'normal':
        return_vals = [ val for val in return_vals ]

    if map_time:
       return_vals = [ [ time_values[i], return_vals[i] ] for i in range(len(return_vals)) ]

    return return_vals


## This function is used to calculate and return Rolling Moving Average.
def get_RMA(prices, maPeriod, time_values=None, prec=8, map_time=False, result_format='normal'):
    """
    This function uses 3 parameters to calculate the Rolling Moving Average-
    
    [PARAMETERS]
        prices  : A list of prices.
        SS_type : The interval type.
        ind_span: The span of the indicator.
    
    [CALCULATION]
        RMA = ((prevRMA * (period - 1)) + currPrice) / period
    
    [RETURN]
        [
        float,
        float,
        ... ]
    """
    span = len(prices) - maPeriod
    SS = np.zeros_like(prices[:span])
    SMA = get_SMA(prices[span:], maPeriod)
    seed = ((SMA * (maPeriod-1)) + prices[span-1]) / maPeriod
    SS[0] = seed

    for i in range(1, span):
        SS[i] = ((SS[i-1] * (maPeriod-1)) + prices[span-i-1]) / maPeriod

    return_vals = np.flipud(SS.round(prec))

    if result_format == 'normal':
        return_vals = [ val for val in return_vals ]

    if map_time:
       return_vals = [ [ time_values[i], return_vals[i] ] for i in range(len(return_vals)) ]

    return return_vals


## This function is used to calculate and return the the MACD indicator.
def get_MACD(prices, time_values=None, Efast=12, Eslow=26, signal=9, map_time=False):
    """
    This function uses 5 parameters to calculate the Moving Average Convergence/Divergence-
    
    [PARAMETERS]
        prices  : A list of prices.
        Efast   : Fast line type.
        Eslow   : Slow line type.
        signal  : Signal line type.
    
    [CALCULATION]
        MACDLine = fastEMA - slowEMA
        SignalLine = EMA of MACDLine
        Histogram = MACDLine - SignalLine
    
    [RETURN]
        [{
        'fast':float,
        'slow':float,
        'his':float
        }, ... ]
    """
    fastEMA = get_EMA(prices, Efast)
    slowEMA = get_EMA(prices, Eslow)

    macdLine = np.subtract(fastEMA[:len(slowEMA)], slowEMA)
    signalLine = get_SMA(macdLine, signal)
    histogram = np.subtract(macdLine[:len(signalLine)], signalLine)

    macd = [({
        "macd":float("{0}".format(macdLine[i])), 
        "signal":float("{0}".format(signalLine[i])), 
        "hist":float("{0}".format(histogram[i]))}) for i in range(len(signalLine))]

    if map_time:
       macd = [ [ time_values[i], macd[i] ] for i in range(len(macd)) ]

    return(macd)


def get_DEMA(prices, maPeriod, prec=8):
    EMA1 = get_EMA(prices, maPeriod)
    EMA2 = get_EMA(EMA1, maPeriod)
    DEMA = np.subtract ((2 * EMA1[:len(EMA2)]), EMA2)

    return DEMA.round(prec)


## This function is used to calculate and return the the MACD indicator.
def get_zeroLagMACD(prices, time_values=None, Efast=12, Eslow=26, signal=9, map_time=False):
    """
    This function uses 5 parameters to calculate the Moving Average Convergence/Divergence-

    Solution with thanks to @dsiens
    
    [PARAMETERS]
        prices  : A list of prices.
        Efast   : Fast line type.
        Eslow   : Slow line type.
        signal  : Signal line type.
    
    [CALCULATION]
        MACDLine = (2 * EMA(price, FAST) - EMA(EMA(price, FAST), FAST)) - (2 * EMA(price, SLOW) - EMA(EMA(price, SLOW), SLOW))
        SignalLine = 2 * EMA(MACD, SIG) - EMA(EMA(MACD, SIG), SIG))
        Histogram = MACDLine - SignalLine

    [RETURN]
        [{
        'fast':float,
        'slow':float,
        'his':float
        }, ... ]
    """
    z1 = get_DEMA(prices, Efast)
    z2 = get_DEMA(prices, Eslow)
    lineMACD = np.subtract (z1[:len(z2)], z2)
    lineSIGNAL = get_DEMA (lineMACD, signal)
    histogram = np.subtract(lineMACD[:len(lineSIGNAL)], lineSIGNAL)

    z_lag_macd = [({
        "macd":float("{0}".format(lineMACD[i])), 
        "signal":float("{0}".format(lineSIGNAL[i])), 
        "hist":float("{0}".format(histogram[i]))}) for i in range(len(lineSIGNAL))]

    if map_time:
       z_lag_macd = [ [ time_values[i], z_lag_macd[i] ] for i in range(len(z_lag_macd)) ]

    return(z_lag_macd)


## This function is used to calculate and return the True Range.
def get_trueRange(candles):
    """
    This function uses 2 parameters to calculate the True Range-

    [PARAMETERS]
        candles : Dict of candles.
        ind_span: The span of the indicator.
    
    [CALCULATION]
        ---
    
    [RETURN]
        [
        float,
        float,
        ... ]
    """
    highPrices = [candle[2] for candle in candles]
    lowPrices = [candle[3] for candle in candles]
    closePrices = [candle[4] for candle in candles]
    span = len(closePrices) - 1
    trueRange = []

    for i in range(span):
        ## Get the true range CURRENT HIGH minus CURRENT LOW, CURRENT HIGH mins LAST CLOSE, CURRENT LOW minus LAST CLOSE.
        HL = highPrices[i] - lowPrices[i]
        H_PC = abs(highPrices[i] - closePrices[i+1])
        L_PC = abs(lowPrices[i] - closePrices[i+1])

        ## True range is the max of all 3
        trueRange.append(max([HL, H_PC, L_PC]))

    return trueRange


## This function is used to calculate and return the Average True Range.
def get_ATR(candles, atrPeriod=14):
    """
    This function uses 3 parameters to calculate the Average True Range-
    
    [PARAMETERS]
        candles : Dict of candles.
        atr_type: The ATR type.
        ind_span: The span of the indicator.
    
    [CALCULATION]
        ---
    
    [RETURN]
        [
        float,
        float,
        ... ]
    """
    trueRange = get_trueRange(candles)
    ATR = get_RMA(trueRange, atrPeriod)

    return ATR


## This is used to calculate the Direcional Movement.
def get_DM(candles):
    """
    This function uses 2 parameters to calculate the positive and negative Directional Movement-
    
    [PARAMETERS]
        candles : Dict of candles.
        ind_span: The span of the indicator.
    
    [CALCULATION]
        ---
    
    [RETURN]
        [[
        float,
        float
        ], ... ]
    """
    highPrices = [candle[2] for candle in candles]
    lowPrices = [candle[3] for candle in candles]
    span = len(highPrices) - 1
    PDM = np.zeros_like(highPrices[:span])
    NDM = np.zeros_like(highPrices[:span])

    for i in range(span):
        ## UP MOVE: current high minus last high, DOWN MOVE: last low minus current low.
        upMove = highPrices[i] - highPrices[i+1]
        downMove = lowPrices[i+1] - lowPrices[i]

        ## If MOVE is greater than other MOVE and greater than 0.
        PDM[i] = upMove if 0 < upMove > downMove else 0
        NDM[i] = downMove if 0 < downMove > upMove else 0

    return [PDM, NDM]


## This function is used to calculate and return the ADX indicator.
def get_ADX_DI(rCandles, adxLen=14, dataType="numpy", map_time=False):
    """
    This function uses 3 parameters to calculate the ADX-

    [PARAMETERS]
        candles : Dict of candles.
        adx_type: The ADX type.
        adx_smooth: The smooting interval or the adx.
        ind_span: The span of the indicator.
    
    [CALCULATION]
        ---
    
    [RETURN]
        [{
        'ADX':float,
        '+DI':float,
        '-DI':float
        }, ... ]
    """

    if dataType == "normal":
        candles = np.array([[0, 
            rCandles["open"][i],
            rCandles["high"][i],
            rCandles["low"][i],
            rCandles["close"][i]]for i in range(len(rCandles["open"]))]).astype(np.float)

    elif dataType == "numpy":
        candles = rCandles

    baseIndLen = adxLen

    DM = get_DM(candles)
    ATR = get_ATR(candles, baseIndLen)
    PDM = get_RMA(DM[0], baseIndLen)
    NDM = get_RMA(DM[1], baseIndLen)

    newRange = len(ATR) if len(ATR) < len(PDM) else len(PDM)
    PDI = np.array([((PDM[i] / ATR[i]) * 100) for i in range(newRange)])
    NDI = np.array([((NDM[i] / ATR[i]) * 100) for i in range(newRange)])

    DI = np.array([(abs(PDI[i] - NDI[i]) / (PDI[i] + NDI[i]) * 100) for i in range(len(NDI))])

    ADX = get_SMA(DI, adxLen)

    ADX_DI = [({
        "ADX":float("{0:3f}".format(ADX[i])), 
        "+DI":float("{0:.3f}".format(PDI[i])), 
        "-DI":float("{0:.3f}".format(NDI[i]))}) for i in range(len(ADX))]

    if map_time:
       ADX_DI = [ [ candles[i][0], ADX_DI[i] ] for i in range(len(ADX_DI)) ]

    return(ADX_DI)


## This function is used to calculate and return the ichimoku indicator.
def get_Ichimoku(candles, tS_type=9, kS_type=26, sSB_type=52, dataType="numpy", map_time=False):
    """
    This function uses 5 parameters to calculate the Ichimoku Cloud-

    [PARAMETERS]
        candles : Dict of candles.
        ind_span: The span of the indicator.
    
    [CALCULATION]
        Tenkan-sen      = (9-day high + 9-day low) / 2
        Kijun-sen       = (26-day high + 26-day low) / 2
        Senkou Span A   = (Tenkan-sen + Kijun-sen) / 2 (This is usually plotted 26 intervals ahead)
        Senkou Span B   = (52-day high + 52-day low) / 2 (This is usually plotted 26 intervals ahead)
        Chikou Span     = current close (This is usually plotted 26 intervals behind)
    
    [RETURN]
        [{
        'Temka':float,
        'Kijun':float,
        'Senkou A':float,
        'Senkou B':float,
        'Chikou':float
        }, ... ]
    """
    if dataType == "normal":
        highPrices = np.array(candles["high"]).astype(np.float)
        lowPrices = np.array(candles["low"]).astype(np.float)
        closePrices = np.array(candles["close"]).astype(np.float)

    elif dataType == "numpy":
        highPrices = [candle[2].astype(np.float) for candle in candles]
        lowPrices = [candle[3].astype(np.float) for candle in candles]
        closePrices = [candle[4].astype(np.float) for candle in candles]

    span = len(lowPrices)

    tS = [ ((max(highPrices[i:tS_type+i]) + min(lowPrices[i:tS_type+i])) / 2) for i in range(span) ]
    kS = [ ((max(highPrices[i:kS_type+i]) + min(lowPrices[i:kS_type+i])) / 2) for i in range(span) ]

    sSA = [ ((kS[i] + tS[i]) / 2) for i in range(span) ]

    sSB = [ ((max(highPrices[i:sSB_type+i]) + min(lowPrices[i:sSB_type+i])) / 2) for i in range(span) ]

    ichimoku = [ ({
        "Tenkan":float("{0}".format(tS[i])),
        "Kijun":float("{0}".format(kS[i])),
        "Senkou A":float("{0}".format(sSA[i])),
        "Senkou B":float("{0}".format(sSB[i])),
        "Chikou":float("{0}".format(closePrices[i]))}) for i in range(span) ]

    if map_time:
       ichimoku = [ [ candles[i][0], ichimoku[i] ] for i in range(len(ichimoku)) ]

    return(ichimoku)


def get_CCI(rCandles, source='all', period=14, constant=0.015, dataType="numpy", map_time=False):
    """
    Commodity channel index

    source is refering too where the typical price will come from. (high, low, close, all)

    CCI = (Typical Price  -  20-period SMA of TP) / (.015 x Mean Deviation)

    Typical Price (TP) = Close

    Constant = .015

    """
    if dataType == "normal":
        candles = np.array([[0, 
                rCandles["open"][i],
                rCandles["high"][i],
                rCandles["low"][i],
                rCandles["close"][i]] for i in range(len(rCandles["open"]))]).astype(np.float)

    elif dataType == "numpy":
        candles = rCandles

    typical_price = get_typical_price(candles, source)

    MAD = get_Mean_ABS_Deviation(typical_price, period)
    smTP = get_SMA(typical_price, period)

    CCI = [ ((typical_price[i] - smTP[i]) / (constant * MAD[i])).round(2) for i in range(len(MAD)) ]

    if map_time:
       CCI = [ [ candles[i][0], CCI[i] ] for i in range(len(CCI)) ]

    return(CCI)


def get_Mean_ABS_Deviation(prices, period):
    """
    There are four steps to calculating the Mean Deviation: 
    First, subtract the most recent 20-period average of the typical price from each period's typical price. 
    Second, take the absolute values of these numbers. 
    Third, sum the absolute values. 
    Fourth, divide by the total number of periods (20).
    """
    partOneTwo = []
    partThreeFour = []

    sma = get_SMA(prices, period)

    for i,ma in enumerate(sma):
        partTwo = [abs(price - ma) for price in prices[i:i+period]]

        partThreeFour.append(np.mean(partTwo))

    return(partThreeFour)



def get_Force_Index(cPrices, volume, maPeriod=9):
    """
    Force Index(1) = {Close (current period)  -  Close (prior period)} x Volume
    Force Index(13) = 13-period EMA of Force Index(1)
    """
    span = len(cPrices) - 1

    baseValues = [(cPrices[i] - cPrices[i+1])*volume[i] for i in range(span)]

    forceIndex = get_EMA(baseValues, maPeriod)

    return(forceIndex)


def get_typical_price(candles, source='all'):

    if source == 'high':
        typical_price = np.array([candle[2] for candle in candles])
    elif source == 'low':
        typical_price = np.array([candle[3] for candle in candles])
    elif source == 'close':
        typical_price = np.array([candle[4] for candle in candles])
    elif source == 'volume':
        typical_price = np.array([candle[5] for candle in candles])
    elif source == 'all':
        typical_price = np.array([((candle[4]+candle[3]+candle[2])/3) for candle in candles])
    else:
        raise ValueError('Issue calculating typical price (Invalid source)')

    return(typical_price)



def get_flow_NEG_POS(typical_price, price_list, period):
    """
    If current price is > close day then its condidered POSTIVE else NEGATIVE

    POSITIVE FLOW = sum of all positives over the period
    NEGATIVE FLOW = sum of all negative over the period
    """

    flow = []

    for index in range(len(price_list)-period):

        negative_flow   = 0
        positive_flow   = 0
        currflowrange   = price_list[index:index+period]
        currTypical     = typical_price[index:index+period+1]

        for i in range(period):

            if currTypical[i] > currTypical[i+1]:
                positive_flow += currflowrange[i]
            elif currTypical[i] < currTypical[i+1]:
                negative_flow += currflowrange[i]

        if negative_flow != 0 and positive_flow != 0:
            flow.append(positive_flow/negative_flow)

    return(flow)



def get_MFI(rCandles, period=14, dataType="numpy", map_time=False):
    '''
    Money flow index

    Typical price

    money flow ratio

    '''

    if dataType == "normal":
        candles = np.array([[0, 
            rCandles["open"][i],
            rCandles["high"][i],
            rCandles["low"][i],
            rCandles["close"][i],
            rCandles["volume"][i]] for i in range(len(rCandles["open"]))]).astype(np.float)

    elif dataType == "numpy":
        candles = rCandles

    typical_price = get_typical_price(candles)

    raw_money_flow = []
    for index, t_price in enumerate(typical_price):
        raw_money_flow.append(t_price*candles[index][5])

    money_flow_ration = get_flow_NEG_POS(typical_price, raw_money_flow, period)

    MFI = [ float('{0:.6f}'.format(100 - 100/(1+item))) for item in money_flow_ration ]

    if map_time:
       MFI = [ [ candles[i][0], MFI[i] ] for i in range(len(MFI)) ]

    return(MFI)


def get_heiken_ashi_candles(rCandles, dataType="numpy"):
    '''
    Open = (open of previous bar + close of previous bar)/2
    Close = (open + high + low + close)/4
    High = the maximum value from the high, open, or close of the current period
    Low = the minimum value from the low, open, or close of the current period
    '''

    if dataType == "normal":
        candles = np.array([[0, 
            rCandles["open"][i],
            rCandles["high"][i],
            rCandles["low"][i],
            rCandles["close"][i],
            rCandles["volume"][i]] for i in range(len(rCandles["open"]))]).astype(np.float)

    elif dataType == "numpy":
        candles = rCandles

    heiken_ashi_candles = []
    candle_len = len(candles)-1

    for i in range(candle_len):
        open_price = (candles[i+1][1]+candles[i+1][4])/2
        close_price = (candles[i][1]+candles[i][2]+candles[i][3]+candles[i][4])/2
        high_price = max([candles[i][1],candles[i][2],candles[i][4]])
        low_price = min([candles[i][1],candles[i][3],candles[i][4]])
        heiken_ashi_candles.append([ int(candles[i][0]), open_price, close_price, high_price, low_price ])

    return(heiken_ashi_candles)


'''
x : Price list
y : Last comparible value.
z : Historic comparible value.

# find_high_high
( x : price list, y : last high, z : historic high value )
Return highest value seen vs both recent and historically or None.

# find_high
( x : price list, y : last high )
Return the highest value seen or None.

# find_low_high
( x : price list, y : last high, z : historic high value )
Return highest value seen recently but lower historically or None.

# find_low_low
( x : price list, y : last low, z : historic low value )
Return the lowest value seen vs both recent and historically or None.

# find_low
( x : price list, y : last low )
Return the lowest value seen or None.

# find_high_low
( x : price list, y : last low, z : historic low value )
Return lowest value seen recently but higher historically or None.
'''
## High setups
find_high_high  = lambda x, y, z: x.max() if z < x.max() > y else None
find_high       = lambda x, y: x.max() if x.max() > y else None
find_low_high   = lambda x, y, z: x.max() if z > x.max() > y else None
## Low setup
find_low_low    = lambda x, y, z: x.min() if z > x.min() < y else None
find_low        = lambda x, y: x.min() if x.min() < y else None
find_high_low   = lambda x, y, z: x.min() if z < x.min() < y else None


def get_tops_bottoms(candles, segment_span, price_point, is_reverse=True, map_time=False):
    data_points = []
    last_timestamp = 0

    if not(candles[0][0] < candles[-1][0]):
        candles = candles[::-1]

    c_move = "up"
    last_val = find_high(np.asarray(candles[0:segment_span])[:,1], 0)
    set_start = 1

    while True:
        if price_point == 0:
            val_index = 3 if c_move == 'down' else 2
        elif price_point == 1:
            val_index = 4
        elif price_point == 2:
            val_index = 1

        # Get the range of candles.
        set_offset = (set_start+segment_span)
        if set_offset > len(candles):
            set_end = len(candles)
        else:
            set_end = set_offset

        base_candle_list = np.asarray(candles[set_start:set_end])
        base_values_list = base_candle_list[:,val_index]

        # This section is used to split of if a higher high or lower low is seen.
        new_end = None
        if len(data_points) > 0:
            if c_move == 'up':
                find_new_end = find_low(base_values_list, data_points[-1][1])
            else:
                find_new_end = find_high(base_values_list, data_points[-1][1])
                
            if find_new_end:
                new_end = np.where(base_values_list == find_new_end)[0][0]

        if new_end:
            current_values = base_values_list[:new_end]
        else:
            current_values = base_values_list

        # Check for high/low values.
        if c_move == 'up':
            find_result = find_high(current_values, last_val)
        else:
            find_result = find_low(current_values, last_val)

        if find_result:
            # Used to find timestamp of new value.
            time_index = np.where(current_values == find_result)[0][0]
            last_timestamp = base_candle_list[time_index][0]
            last_val = find_result
            set_start += time_index+1

        else:
            # Record the value to be used later.
            data_points.append([ int(last_timestamp), last_val ])
            c_move = "down" if c_move == "up" else "up"
            first_search = True

        if set_end == len(candles):
            data_points.append([ int(last_timestamp), last_val ])
            break

    if data_points[0][0] != int(candles[-1][0]):
        data_points.append([int(candles[-1][0]), candles[-1][4]])

    if is_reverse:
        data_points = data_points[::-1]

    data_points = data_points if map_time else [point[1] for point in data_points]
    
    return(data_points)


def get_CPS(rCandles, dataType="numpy", map_time=False):
    '''
    
    '''
    if dataType == "normal":
        candles = np.array([[0, 
            rCandles["open"][i],
            rCandles["high"][i],
            rCandles["low"][i],
            rCandles["close"][i],
            rCandles["volume"][i]] for i in range(len(rCandles["open"]))]).astype(np.float)

    elif dataType == "numpy":
        candles = rCandles

    candle_price_strength = [ (candle[5]/abs(candle[1]-candle[4])) for candle in candles ]

    if map_time:
       candle_price_strength = [ [ candles[i][0], candle_price_strength[i] ] for i in range(len(candle_price_strength)) ]

    return(candle_price_strength)