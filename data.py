 
import requests
import pandas as pd
import datetime
import time
import streamlit as st
import yfinance as yf

def get_nse_equity():
    nse_equity_df = pd.read_csv("data/output.csv")
    nse_equity_df = nse_equity_df.iloc[:,:]
    return nse_equity_df

def get_symbol_lst():
    nse_equity_df = pd.read_csv("data/output.csv")
#    symbol_lst = nse_equity_df["NSE SYMBOL"].tolist()
    symbol_yh =  nse_equity_df["YAHOO SYMBOL"].tolist()  
    return symbol_yh

def get_company_info(stock):
    company_dict = yf.Ticker(stock).info
    return company_dict

def get_stock_history(stock):
    start_date = yf.Ticker(stock).history(period="max").reset_index().iloc[0][0].strftime("%Y-%m-%d")
    end_date= datetime.date.today()
    stock_history = yf.Ticker(stock).history(start=start_date, end=end_date)
    return stock_history

def stock_ohlc(stock, time_range):
    ohlc_info = yf.Ticker(stock).history(period=time_range)
    return ohlc_info.sort_values(by="Date", ascending=False).iloc[:,:-2]
