from src.forecasting.forecaster import StockForecaster

def main():
    """Main function to demonstrate the forecasting system"""
    
    # Define stocks to forecast
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    print("🚀 Multi-Stock Time Series Transformer Forecaster")
    print("=" * 50)
    
    # Initialize forecaster
    forecaster = StockForecaster(
        symbols=symbols, 
        sequence_length=60, 
        forecast_horizon=5
    )
    
    try:
        # Prepare data
        print("\n📊 Step 1: Preparing data...")
        forecaster.prepare_data()
        
        # Build model
        print("\n🏗️ Step 2: Building transformer model...")
        forecaster.build_model()
        
        # Train model
        print("\n🎯 Step 3: Training model...")
        forecaster.train(epochs=30, lr=0.001)
        
        # Plot training history
        print("\n📈 Step 4: Plotting training history...")
        forecaster.plot_training_history()
        
        # Make predictions for each symbol
        print("\n🔮 Step 5: Making predictions...")
        for symbol in symbols:
            print(f"\nForecasting {symbol}...")
            predictions = forecaster.plot_predictions(symbol)
            print(f"{symbol} 5-day forecast: {predictions}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have internet connection for fetching stock data.")

if __name__ == "__main__":
    main()
