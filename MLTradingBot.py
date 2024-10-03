import numpy as np
import yfinance as yf
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime 
from alpaca_trade_api import REST 
from pandas import Timedelta 
from finbert_utils import estimate_sentiment

apiKey = "xxx"
apiSecret = "xxx"
baseURL = "https://paper-api.alpaca.markets/v2"


#Set up alpaca credentials to be used
AlapacaCreds={
    "API_KEY":apiKey,
    "API_SECRET":apiSecret,
    "PAPER": True
}

#The class for the SPY bot trader
class SPYBotTrader(Strategy):

    #Create an instance of this class
    def initialize(self, symbol:str="SPY",cash_at_risk:float=0.50):
        self.ticker = symbol
        self.risk = cash_at_risk
        self.lastTrade = None
        self.sleeptime = "24H"
        self.api = REST(base_url=baseURL, key_id=apiKey, secret_key=apiSecret)

    #Calculate the current volatility of the stock
    def Calc_Volatility(self):
        ticker = self.ticker
        today = self.get_datetime()
        backEightDays = today - Timedelta(days=5)

        # Fetch historical prices
        data = yf.download(ticker, start=backEightDays.date(), end=today.date())

        # Calculate daily returns from data
        data['returns'] = data['Close'].pct_change()

        # Get the last five trading days' returns
        weekReturns = data['returns'].tail(5).dropna()

        # Check if weekReturns has enough valid entries
        if len(weekReturns) < 2:
            return 0.01  # Return a small value if not enough valid returns or zero variance
        # Calculate past week volatility
        volatility = weekReturns.std()

        return max(volatility, 0.0001)  # So that it cannot be too low

    
    #Calculate mulitplier for 
    def Calc_Multiplier(self):
        volatility = self.Calc_Volatility()

        #Log volatility for multiplication (max is 4 min is 0)
        loggedVolatility = -(np.log10(volatility))

        #Multiplier on a scale of 0.50 to 1.50
        multiplier = 1.5-loggedVolatility*(1/4)
        return multiplier
    
    #Create position to buy stocks
    def Create_Position(self):
        cash = self.get_cash()
        price = self.get_last_price(self.ticker)
        risk = self.risk
        
        percentageCash = risk*self.Calc_Multiplier()

        quantityToPurchase = round(cash*percentageCash/price,0)
        return cash, price, quantityToPurchase
    
    #Formate dates to pass through sentimate machine learning algorithm
    def Format_Dates(self):
        today = self.get_datetime()
        threeDaysAgo = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), threeDaysAgo.strftime('%Y-%m-%d')

    #Calculate sentimates using machine learning
    def Calc_Sentimates(self): 
        today, threeDaysAgo = self.Format_Dates()
        news = self.api.get_news(symbol=self.ticker, 
                                 start=threeDaysAgo, 
                                 end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment 

    #Iterate through trading algorithm based on sentimates and calculated position
    def on_trading_iteration(self):
        cash, price, quantity = self.Create_Position()
        probability, sentiment = self.Calc_Sentimates()

        if cash > price: 
            if sentiment == "positive" and probability > .999: 
                if self.lastTrade == "sell": 
                    self.sell_all() 
                order = self.create_order(
                    self.ticker, 
                    quantity, 
                    "buy", 
                    type="bracket", 
                    take_profit_price=price*1.20, 
                    stop_loss_price=price*.95
                )
                self.submit_order(order) 
                self.lastTrade = "buy"
            elif sentiment == "negative" and probability > .999: 
                if self.lastTrade == "buy": 
                    self.sell_all() 
                order = self.create_order(
                    self.ticker, 
                    quantity, 
                    "sell", 
                    type="bracket", 
                    take_profit_price=price*.8, 
                    stop_loss_price=price*1.05
                )
                self.submit_order(order) 
                self.lastTrade = "sell"

#Defining variables (setting the backtesting time to 8 years of data)
startDate = datetime(2016,1,1)
endDate = datetime(2024,5,31) 
broker = Alpaca(AlapacaCreds) 

strategy = SPYBotTrader(name='SPYBotTrader', broker=broker, 
                    parameters={"symbol":"SPY", 
                                "cash_at_risk":0.50})

#Backtesting the strategy across 8 years
strategy.backtest(
    YahooDataBacktesting, 
    startDate, 
    endDate, 
    parameters={"symbol":"SPY", "cash_at_risk":0.50}
)

# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
