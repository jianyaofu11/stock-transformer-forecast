import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockDataProcessor:
    """Handles data fetching and feature engineering for multiple stocks"""
    
    def __init__(self, symbols, period='2y'):
        self.symbols = symbols
        self.period = period
        self.scalers = {}
        
    def fetch_data(self):
        """Fetch stock data for all symbols"""
        print("Fetching stock data...")
        stock_data = {}
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=self.period)
                stock_data[symbol] = data
                print(f"✓ Fetched {symbol}: {len(data)} days")
            except Exception as e:
                print(f"✗ Failed to fetch {symbol}: {e}")
                
        return stock_data
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Price change percentage
        df['Price_Change_Pct'] = df['Close'].pct_change()
        
        # Volume moving average
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        
        return df
    
    def prepare_data(self, stock_data, sequence_length=60, forecast_horizon=5):
        """Prepare data for transformer model"""
        processed_data = []
        
        for symbol, data in stock_data.items():
            # Add technical indicators
            data = self.add_technical_indicators(data)
            
            # Select features
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'MA_5', 'MA_20', 'MA_50', 'RSI', 
                       'BB_Upper', 'BB_Lower', 'Price_Change_Pct', 'Volume_MA']
            
            # Drop rows with NaN values
            data = data[features].dropna()
            
            # Scale the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            self.scalers[symbol] = scaler
            
            # Create sequences
            sequences = []
            targets = []
            
            for i in range(sequence_length, len(scaled_data) - forecast_horizon + 1):
                # Input sequence
                seq = scaled_data[i-sequence_length:i]
                # Target (next forecast_horizon close prices)
                target = scaled_data[i:i+forecast_horizon, 3]  # Close price index
                
                sequences.append(seq)
                targets.append(target)
            
            processed_data.append({
                'symbol': symbol,
                'sequences': np.array(sequences),
                'targets': np.array(targets),
                'scaler': scaler,
                'feature_names': features
            })
        
        return processed_data
