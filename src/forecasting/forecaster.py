import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.data.processor import StockDataProcessor
from src.data.dataset import MultiStockDataset
from src.model.transformer import TimeSeriesTransformer

class StockForecaster:
    """Main forecasting class that orchestrates the entire pipeline"""
    
    def __init__(self, symbols, sequence_length=60, forecast_horizon=5):
        self.symbols = symbols
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.processor = StockDataProcessor(symbols)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def prepare_data(self):
        """Fetch and prepare data for training"""
        # Fetch stock data
        stock_data = self.processor.fetch_data()
        
        # Process data
        processed_data = self.processor.prepare_data(
            stock_data, self.sequence_length, self.forecast_horizon
        )
        
        # Create dataset
        dataset = MultiStockDataset(processed_data)
        
        # Split into train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Store processed data for later use
        self.processed_data = processed_data
        
        print(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
    def build_model(self):
        """Build the transformer model"""
        # Get input dimension from processed data
        input_dim = len(self.processed_data[0]['feature_names'])
        
        self.model = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            forecast_horizon=self.forecast_horizon,
            dropout=0.1
        ).to(self.device)
        
        print(f"Model built with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
    def train(self, epochs=50, lr=0.001):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=True
        )
        
        train_losses = []
        val_losses = []
        
        print("Starting training...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_idx, (sequences, targets, _) in enumerate(self.train_loader):
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, targets, _ in self.val_loader:
                    sequences, targets = sequences.to(self.device), targets.to(self.device)
                    outputs = self.model(sequences)
                    val_loss += criterion(outputs, targets).item()
            
            train_loss /= len(self.train_loader)
            val_loss /= len(self.val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        self.train_losses = train_losses
        self.val_losses = val_losses
        print("Training completed!")
        
    def predict(self, symbol, last_sequence=None):
        """Make predictions for a specific symbol"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        self.model.eval()
        
        # Get the symbol's data
        symbol_data = next(data for data in self.processed_data if data['symbol'] == symbol)
        
        if last_sequence is None:
            # Use the last sequence from training data
            last_sequence = symbol_data['sequences'][-1]
        
        # Make prediction
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
            prediction = self.model(sequence_tensor).cpu().numpy().flatten()
        
        # Inverse transform the prediction (only for close price)
        scaler = symbol_data['scaler']
        
        # Create a dummy array with the same shape as the original features
        dummy_features = np.zeros((len(prediction), len(symbol_data['feature_names'])))
        dummy_features[:, 3] = prediction  # Close price is at index 3
        
        # Inverse transform
        inverse_transformed = scaler.inverse_transform(dummy_features)
        predicted_prices = inverse_transformed[:, 3]
        
        return predicted_prices
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses[-20:], label='Training Loss (Last 20)')
        plt.plot(self.val_losses[-20:], label='Validation Loss (Last 20)')
        plt.title('Recent Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, symbol, days_to_show=30):
        """Plot predictions vs actual prices for a symbol"""
        # Get recent actual data
        ticker = yf.Ticker(symbol)
        recent_data = ticker.history(period='3mo')['Close'].values
        
        # Make prediction
        predictions = self.predict(symbol)
        
        # Plot
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(range(-days_to_show, 0), recent_data[-days_to_show:], 
                label=f'{symbol} Historical', linewidth=2, color='blue')
        
        # Plot predictions
        plt.plot(range(0, len(predictions)), predictions, 
                label=f'{symbol} Forecast', linewidth=2, color='red', marker='o')
        
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, label='Forecast Start')
        plt.xlabel('Days')
        plt.ylabel('Price ($)')
        plt.title(f'{symbol} Stock Price Forecast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return predictions