#!/usr/bin/env python3

import argparse
import sys
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# main function
def main(args):
    # read portfolio
    df_portfolio = pd.read_csv(args.portfolio)

    # get dates
    start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')

    if args.end_date:
        end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.datetime.today() + datetime.timedelta(days=1)

    # get symbols and weights
    symbols = df_portfolio['symbol'].values
    weights = df_portfolio['weight'].values

    # normalize weights
    weights = weights / weights.sum()

    # get financial data
    if args.data:
        df_data = pd.read_csv(args.data)
    else:
        # download data
        print('Downloading data...')
        df_data = yf.download(symbols.tolist(), start_date, end_date)

    # calculate returns
    df_returns = df_data['Adj Close'][symbols].pct_change()

    # calculate portfolio returns based on weights
    df_portfolio_returns = (df_returns * weights).sum(axis=1)

    # calculate portfolio value
    df_portfolio_value = (1 + df_portfolio_returns).cumprod()

    # calculate portfolio statistics, such as sharpe ratio, max drawdown, etc.
    sharpe_ratio = df_portfolio_returns.mean() / df_portfolio_returns.std() * np.sqrt(252)
    max_drawdown = (df_portfolio_value / df_portfolio_value.cummax() - 1).min()

    # print results
    print('Portfolio value: {:.2f}%'.format(df_portfolio_value.iloc[-1] * 100))
    print('Annualized Sharpe ratio: {:.2f}'.format(sharpe_ratio))
    print('Max drawdown: {:.2f}%'.format(max_drawdown * 100))

    # plot
    plt.figure(figsize=(10, 6))

    plt.plot(df_portfolio_value.index, df_portfolio_value)

    plt.title('Portfolio value')
    plt.xlabel('Date')
    plt.ylabel('Value')

    plt.xlim(start_date, end_date)

    # Y-axis tick display as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}%'.format(x * 100)))

    plt.grid(True)
    plt.tight_layout()

    # save plot
    plt.savefig(f'portfolio_value_{args.portfolio.split(".")[0]}.png', dpi=300)
    plt.savefig(f'portfolio_value_{args.portfolio.split(".")[0]}.pdf')

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--portfolio', type=str, help='Portfolio file')
    parser.add_argument('--data', type=str, nargs='?', help='Financial data file. If not provided, download data')
    parser.add_argument('--start_date', type=str, help='Start date')
    parser.add_argument('--end_date', type=str, help='End date. Default is today')

    args = parser.parse_args()

    main(args)
