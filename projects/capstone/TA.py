'''This module calculates the technical indicators that will serve
as state variables in the RL model. The function names correspond with the
abbreviations used in the industry'''

import talib


def reverse_array(x_var):
    ''' reverses the order of an array '''
    x_var = x_var[::-1]
    return x_var

def cmo(close, period, order):
    '''calculates the Change Momentum Oscillator
    order parameter can have 2 values (ASC for ascending and DESC for descending)
    and indicates if price data is in ascending or descending order with respect
     to dates'''
    if order == 'DESC':
        close = reverse_array(close)
    cmo_ind = talib.CMO(close, period)
    #if we selected the option DESC we need to reverse the results again
    if order == 'DESC':
        cmo_ind = reverse_array(cmo_ind)
    return cmo_ind


def adxr(high, low, close, period, order):
    '''calculates the Average Directional Movement Index
    order parameter can have 2 values (ASC for ascending and DESC for
    descending) and indicates if price data is in ascending or descending
    order with respect to dates'''
    if order == 'DESC':
        high = reverse_array(high)
        low = reverse_array(low)
        close = reverse_array(close)
    adxr_ind = talib.ADXR(high, low, close, period)
    #if we selected the option DESC we need to reverse the results again
    if order == 'DESC':
        adxr_ind = reverse_array(adxr_ind)
    return adxr_ind

def macd(close, fastperiod, slowperiod, columntoshow, order):
    '''Calculates the Moving Average Convergence/Divergence indicator
    order parameter can have 2 values (ASC for ascending and DESC for
    descending) and indicates if price data is in ascending or descending
    order with respect to dates
    #columntoshow=0 returns MACD
    #columntoshow=1 returns trigger
    #columntoshow=2 returns the oscillator'''

    if order == 'DESC':
        close = reverse_array(close)
    macd_ind = talib.MACD(close, fastperiod, slowperiod)
    data2show = macd_ind[columntoshow]
    if order == 'DESC':
        #if we selected the option DESC we need to reverse the results again
        data2show = reverse_array(data2show)
    return data2show

def mom(close, period, order):
    '''Calculkates the Momentum Indicator
    order parameter can have 2 values (ASC for ascending and DESC for
    descending) and indicates if price data is in ascending or descending
    order with respect to dates'''
    if order == 'DESC':
        close = reverse_array(close)
    mom_ind = talib.MOM(close, period)
    #if we selected the option DESC we need to reverse the results again
    if order == 'DESC':
        mom_ind = reverse_array(mom_ind)
    return mom_ind

def rsi(close, period, order):
    '''Calculkates the Relative Strength Indicator
    order parameter can have 2 values (ASC for ascending and DESC for
    descending) and indicates if price data is in ascending or descending
    order with respect to dates'''
    if order == 'DESC':
        close = reverse_array(close)
    rsi_ind = talib.RSI(close, period)
    #if we selected the option DESC we need to reverse the results again
    if order == 'DESC':
        rsi_ind = reverse_array(rsi_ind)
    return rsi_ind
