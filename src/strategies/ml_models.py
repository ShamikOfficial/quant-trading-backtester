# ML-based trading strategies using XGBoost

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from src.strategies.base_strategy import BaseStrategy

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    print("Warning: scikit-learn not installed. Some metrics functions may not work.")
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("Warning: xgboost not installed. Please install it using: pip install xgboost")


class XGBoostModel:
    # XGBoost model for price prediction and trading signals
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 5,
                 learning_rate: float = 0.1,
                 objective: str = 'reg:squarederror',
                 random_state: int = 42):
        # Initialize XGBoost model
        if xgb is None:
            raise ImportError("xgboost is required. Install it using: pip install xgboost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.random_state = random_state
        self.model = None
        self.fitted = False
        self.feature_names = None
        
    def prepare_features(self, data: pd.DataFrame, 
                        lookback_window: int = 10) -> pd.DataFrame:
        # Prepare technical features from price data
        df = data.copy()
        
        # Ensure we have required columns
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        # Price-based features (only if not already present)
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        if 'log_returns' not in df.columns:
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages (use existing if available, otherwise calculate)
        for window in [5, 10, 20]:
            if f'sma_{window}' not in df.columns:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
        
        # Use existing SMA if available
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['sma_20_50_ratio'] = df['sma_20'] / df['sma_50']
        
        # Use existing EMA if available
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            df['ema_12_26_diff'] = df['ema_12'] - df['ema_26']
            df['ema_12_26_ratio'] = df['ema_12'] / df['ema_26']
        
        # Use existing RSI if available
        if 'rsi_14' in df.columns:
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        
        # Volatility
        if 'volatility' not in df.columns:
            df['volatility'] = df['returns'].rolling(window=lookback_window).std()
        
        # Price momentum
        if 'momentum' not in df.columns:
            df['momentum'] = df['close'] / df['close'].shift(lookback_window) - 1
        
        # High-Low spread
        if 'high' in df.columns and 'low' in df.columns:
            if 'hl_ratio' not in df.columns:
                df['hl_ratio'] = df['high'] / df['low']
            if 'hl_spread' not in df.columns:
                df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Volume features
        if 'volume' in df.columns:
            if 'volume_ma' not in df.columns:
                df['volume_ma'] = df['volume'].rolling(window=lookback_window).mean()
            if 'volume_ratio' not in df.columns:
                df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # VWAP features if available
        if 'vwap' in df.columns:
            df['price_vwap_ratio'] = df['close'] / df['vwap']
            df['vwap_spread'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            if f'close_lag_{lag}' not in df.columns:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
            if f'returns_lag_{lag}' not in df.columns:
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Drop rows with NaN values (but keep non-feature columns)
        feature_cols = [col for col in df.columns 
                        if col not in ['ticker', 'datetime_et', 'datetime_utc', 'date_et']]
        df = df.dropna(subset=feature_cols)
        
        return df
    
    def prepare_target(self, data: pd.DataFrame, 
                      target_type: str = 'returns',
                      forward_periods: int = 1) -> pd.Series:
        # Prepare target variable for prediction
        if target_type == 'returns':
            # Predict future returns
            target = data['close'].pct_change(periods=forward_periods).shift(-forward_periods)
        elif target_type == 'direction':
            # Predict price direction (1 for up, 0 for down)
            future_price = data['close'].shift(-forward_periods)
            target = (future_price > data['close']).astype(int)
        elif target_type == 'price':
            # Predict future price
            target = data['close'].shift(-forward_periods)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        return target
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Train the XGBoost model
        if xgb is None:
            raise ImportError("xgboost is required. Install it using: pip install xgboost")
        
        # Align X and y, remove NaN values
        aligned_data = pd.concat([X, y], axis=1).dropna()
        X_clean = aligned_data[X.columns]
        y_clean = aligned_data[y.name]
        
        # Store feature names
        self.feature_names = list(X_clean.columns)
        
        # Create and train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=self.objective,
            random_state=self.random_state
        )
        
        self.model.fit(X_clean, y_clean)
        self.fitted = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Generate predictions from trained model
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.feature_names:
            # Ensure feature order matches training
            X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def forecast(self, data: pd.DataFrame, 
                lookback_window: int = 10,
                target_type: str = 'returns') -> Tuple[np.ndarray, pd.DataFrame]:
        # Prepare features, fit model, and generate predictions
        # Prepare features
        features_df = self.prepare_features(data, lookback_window)
        
        # Prepare target
        target = self.prepare_target(data, target_type)
        
        # Ensure target is a Series and rename to avoid conflicts
        if not isinstance(target, pd.Series):
            raise ValueError(f"prepare_target returned {type(target)}, expected Series")
        target_name = 'target_returns'
        target = target.rename(target_name)
        
        # Align and split data
        aligned_data = pd.concat([features_df, target], axis=1).dropna()
        
        # Select feature columns (exclude target and original price columns)
        feature_cols = [col for col in aligned_data.columns 
                       if col not in ['close', 'open', 'high', 'low', 'volume', target_name]]
        
        X = aligned_data[feature_cols]
        y = aligned_data[target_name]
        
        # Split into train and predict sets (use last row for prediction)
        if len(X) > 1:
            X_train = X.iloc[:-1]
            y_train = y.iloc[:-1]
            X_predict = X.iloc[-1:].values.reshape(1, -1)
            
            # Fit model
            self.fit(X_train, y_train)
            
            # Make prediction
            prediction = self.predict(pd.DataFrame(X_predict, columns=self.feature_names))
            
            return prediction, features_df
        else:
            raise ValueError("Insufficient data for forecasting")
    
    def save_model(self, filepath: str) -> None:
        # Save trained model to file
        if not self.fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': self.objective,
            'random_state': self.random_state
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        # Load trained model from file
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.n_estimators = model_data['n_estimators']
        self.max_depth = model_data['max_depth']
        self.learning_rate = model_data['learning_rate']
        self.objective = model_data['objective']
        self.random_state = model_data['random_state']
        self.fitted = True


class MLStrategy(BaseStrategy):
    # Trading strategy using XGBoost ML model
    
    def __init__(self, name: str = "MLStrategy",
                 parameters: Optional[Dict[str, Any]] = None):
        # Initialize ML strategy
        super().__init__(name, parameters)
        
        # Model parameters
        self.n_estimators = self.parameters.get('n_estimators', 100)
        self.max_depth = self.parameters.get('max_depth', 5)
        self.learning_rate = self.parameters.get('learning_rate', 0.1)
        self.lookback_window = self.parameters.get('lookback_window', 10)
        self.min_confidence = self.parameters.get('min_confidence', 0.6)
        
        # Initialize model
        self.model = XGBoostModel(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate
        )
        self.model_trained = False
        self.training_data = None
    
    def generate_signal(self, data: pd.DataFrame, 
                       current_position: Dict[str, Any]) -> Dict[str, Any]:
        # Generate trading signals using XGBoost predictions
        if 'close' not in data.columns:
            return {'action': 'HOLD', 'confidence': 0.0}
        
        # Need sufficient data for prediction
        if len(data) < self.lookback_window + 10:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        try:
            # Prepare features
            features_df = self.model.prepare_features(data, self.lookback_window)
            
            if len(features_df) == 0:
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'No features available'}
            
            # Get latest features for prediction
            feature_cols = [col for col in features_df.columns 
                           if col not in ['close', 'open', 'high', 'low', 'volume']]
            
            if not feature_cols:
                return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'No valid features'}
            
            X_latest = features_df[feature_cols].iloc[-1:]
            
            # Train model if not trained (skip if model is already loaded and trained)
            # Check if model has been fitted (for loaded models)
            model_is_fitted = hasattr(self.model, 'fitted') and self.model.fitted
            if (not self.model_trained or self.training_data is None) and not model_is_fitted:
                self.training_data = data.copy()
                target = self.model.prepare_target(data, target_type='returns')
                
                # Ensure target is a Series and rename to avoid conflicts
                if not isinstance(target, pd.Series):
                    raise ValueError(f"prepare_target returned {type(target)}, expected Series")
                target_name = 'target_returns'
                target = target.rename(target_name)
                
                aligned_data = pd.concat([features_df, target], axis=1).dropna()
                
                if len(aligned_data) > 20:  # Need minimum data for training
                    X_train = aligned_data[feature_cols].iloc[:-1]
                    y_train = aligned_data[target_name].iloc[:-1]
                    self.model.fit(X_train, y_train)
                    self.model_trained = True
            
            # Make prediction
            if self.model_trained:
                # Ensure feature columns match model's expected features
                if hasattr(self.model, 'feature_names') and self.model.feature_names:
                    # Check for missing features
                    missing_features = set(self.model.feature_names) - set(X_latest.columns)
                    extra_features = set(X_latest.columns) - set(self.model.feature_names)
                    
                    if missing_features:
                        # Add missing features with 0 values (or forward fill from available data)
                        for feat in missing_features:
                            X_latest[feat] = 0.0
                    
                    # Remove extra features that model doesn't expect
                    if extra_features:
                        X_latest = X_latest.drop(columns=list(extra_features))
                    
                    # Reorder to match model's feature order
                    X_latest = X_latest[self.model.feature_names]
                
                try:
                    pred_array = self.model.predict(X_latest)
                    if len(pred_array) > 0:
                        prediction = float(pred_array[0])
                    else:
                        prediction = 0.0
                    # Handle NaN or invalid predictions
                    if pd.isna(prediction) or not isinstance(prediction, (int, float)):
                        prediction = 0.0
                except Exception as e:
                    # If prediction fails, return HOLD with error reason
                    import traceback
                    error_msg = f'Prediction error: {str(e)}'
                    # Uncomment for debugging: print(f"Prediction error: {error_msg}\n{traceback.format_exc()}")
                    return {'action': 'HOLD', 'confidence': 0.0, 'reason': error_msg, 'predicted_return': 0.0}
                
                current_price = data['close'].iloc[-1]
                
                # Convert prediction to signal
                # Positive prediction = buy, negative = sell
                # Predictions are returns (typically 0.0001 to 0.01 range for minute-level data)
                # Use very low threshold since returns are small
                prediction_threshold = 0.00005  # Very low threshold (0.005% return)
                
                # Calculate confidence: use absolute prediction value
                # Scale predictions (which are typically 0.0001-0.01) to 0-1 confidence range
                # A prediction of 0.001 (0.1% return) should give decent confidence
                abs_pred = abs(prediction)
                # Use sigmoid-like scaling: confidence increases with prediction magnitude
                # Scale so that 0.001 = ~0.3 confidence, 0.01 = ~0.9 confidence
                confidence = min(abs_pred * 300, 1.0)  # 0.001 * 300 = 0.3, 0.01 * 300 = 3.0 -> capped at 1.0
                
                if prediction > prediction_threshold:  # Positive expected return threshold
                    if confidence >= self.min_confidence:
                        signal = {
                            'action': 'BUY',
                            'confidence': confidence,
                            'predicted_return': prediction,
                            'reason': 'XGBoost positive prediction'
                        }
                    else:
                        signal = {'action': 'HOLD', 'confidence': confidence, 'predicted_return': prediction}
                
                elif prediction < -prediction_threshold:  # Negative expected return threshold
                    if confidence >= self.min_confidence:
                        signal = {
                            'action': 'SELL',
                            'confidence': confidence,
                            'predicted_return': prediction,
                            'reason': 'XGBoost negative prediction'
                        }
                    else:
                        signal = {'action': 'HOLD', 'confidence': confidence, 'predicted_return': prediction}
                else:
                    # Prediction is too small, but still include it for debugging
                    signal = {
                        'action': 'HOLD', 
                        'confidence': confidence,  # Use calculated confidence even if below threshold
                        'predicted_return': prediction,
                        'reason': f'Prediction {prediction:.6f} below threshold {prediction_threshold}'
                    }
            else:
                signal = {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Model not trained'}
            
            self.signals_history.append(signal)
            return signal
            
        except Exception as e:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def create_order(self, signal: Dict[str, Any], 
                    portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        # Create order from ML signal
        from datetime import datetime
        
        if signal['action'] == 'HOLD':
            return None
        
        current_price = portfolio_state.get('current_price', 0.0)
        symbol = portfolio_state.get('symbol', 'UNKNOWN')
        cash = portfolio_state.get('cash', 0.0)
        current_position = portfolio_state.get('positions', {}).get(symbol, {})
        current_quantity = current_position.get('quantity', 0)
        
        confidence = signal.get('confidence', 0.5)
        predicted_return = signal.get('predicted_return', 0.0)
        
        if signal['action'] == 'BUY':
            if current_price > 0:
                # Use confidence and predicted return to determine position size
                position_size_multiplier = confidence * (1 + abs(predicted_return))
                cash_to_use = cash * min(position_size_multiplier * 0.15, 0.3)  # Max 30% per trade
                quantity = int(cash_to_use / current_price)
            else:
                quantity = 0
            
            if quantity > 0:
                order = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'predicted_return': predicted_return
                }
                self.orders_history.append(order)
                return order
        
        elif signal['action'] == 'SELL':
            if current_quantity > 0:
                # Sell based on confidence and predicted return
                sell_percentage = confidence * (1 + abs(predicted_return))
                quantity_to_sell = int(current_quantity * min(sell_percentage, 1.0))
                
                if quantity_to_sell > 0:
                    order = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity_to_sell,
                        'price': current_price,
                        'timestamp': datetime.now(),
                        'predicted_return': predicted_return
                    }
                    self.orders_history.append(order)
                    return order
        
        return None
    
    def retrain_model(self, new_data: pd.DataFrame) -> None:
        # Retrain model with new data
        if self.training_data is not None:
            # Combine old and new data
            combined_data = pd.concat([self.training_data, new_data]).drop_duplicates()
        else:
            combined_data = new_data.copy()
        
        self.training_data = combined_data
        
        # Prepare features and target
        features_df = self.model.prepare_features(combined_data, self.lookback_window)
        target = self.model.prepare_target(combined_data, target_type='returns')
        
        # Ensure target is a Series and rename to avoid conflicts
        if not isinstance(target, pd.Series):
            raise ValueError(f"prepare_target returned {type(target)}, expected Series")
        target_name = 'target_returns'
        target = target.rename(target_name)
        
        # Align data
        aligned_data = pd.concat([features_df, target], axis=1).dropna()
        
        if len(aligned_data) > 20:
            feature_cols = [col for col in aligned_data.columns 
                           if col not in ['close', 'open', 'high', 'low', 'volume', target_name]]
            
            X_train = aligned_data[feature_cols].iloc[:-1]
            y_train = aligned_data[target_name].iloc[:-1]
            
            self.model.fit(X_train, y_train)
            self.model_trained = True


def find_processed_file(pattern_or_path: str = "data/processed/processed_*.csv") -> str:
    # Find most recent processed data file matching pattern
    from glob import glob
    
    # If it's an exact path and exists, return it
    path = Path(pattern_or_path)
    if path.exists() and path.is_file():
        return str(path)
    
    # Try glob pattern
    matches = glob(pattern_or_path)
    if not matches:
        raise FileNotFoundError(f"No files found matching pattern: {pattern_or_path}")
    
    # Return most recent file
    return max(matches, key=lambda p: Path(p).stat().st_mtime)


def load_processed_data(filepath: str, ticker: Optional[str] = None) -> pd.DataFrame:
    # Load processed data file, optionally filter by ticker
    # Handle glob patterns
    if '*' in filepath or '?' in filepath:
        filepath = find_processed_file(filepath)
    
    df = pd.read_csv(filepath)
    
    # Convert datetime columns if present (data is already in Eastern time)
    # Parse with utc=True to handle mixed timezones, then convert to naive datetime
    if 'datetime_et' in df.columns:
        try:
            # Parse with UTC to handle mixed timezones, then convert to naive (removes timezone info)
            df['datetime_et'] = pd.to_datetime(df['datetime_et'], utc=True, errors='coerce')
            # Convert to naive datetime (removes timezone info, keeps the time values)
            if df['datetime_et'].dt.tz is not None:
                df['datetime_et'] = df['datetime_et'].dt.tz_convert(None)
        except Exception:
            # Fallback: try without UTC conversion
            try:
                df['datetime_et'] = pd.to_datetime(df['datetime_et'], errors='coerce')
                if df['datetime_et'].dt.tz is not None:
                    df['datetime_et'] = df['datetime_et'].dt.tz_convert(None)
            except Exception:
                # Last resort: parse as string and remove timezone info manually
                df['datetime_et'] = pd.to_datetime(df['datetime_et'].astype(str).str.replace(r'[+-]\d{2}:\d{2}$', '', regex=True), errors='coerce')
    
    if 'datetime_utc' in df.columns:
        try:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True, errors='coerce')
            if df['datetime_utc'].dt.tz is not None:
                df['datetime_utc'] = df['datetime_utc'].dt.tz_convert(None)
        except Exception:
            try:
                df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], errors='coerce')
                if df['datetime_utc'].dt.tz is not None:
                    df['datetime_utc'] = df['datetime_utc'].dt.tz_convert(None)
            except Exception:
                df['datetime_utc'] = pd.to_datetime(df['datetime_utc'].astype(str).str.replace(r'[+-]\d{2}:\d{2}$', '', regex=True), errors='coerce')
    
    # Filter by ticker if specified
    if ticker:
        df = df[df['ticker'] == ticker].copy()
    
    # Sort by datetime
    if 'datetime_et' in df.columns:
        df = df.sort_values('datetime_et').reset_index(drop=True)
    
    return df


def train_model_on_processed_data(processed_file: str,
                                  ticker: Optional[str] = None,
                                  lookback_window: int = 10,
                                  train_test_split: float = 0.8,
                                  model_params: Optional[Dict[str, Any]] = None,
                                  save_model_path: Optional[str] = None) -> Tuple[XGBoostModel, Dict[str, Any]]:
    # Train XGBoost model on processed data
    # Load data
    data = load_processed_data(processed_file, ticker=ticker)
    
    if len(data) < 100:
        raise ValueError(f"Insufficient data: {len(data)} rows. Need at least 100 rows.")
    
    # Initialize model
    if model_params is None:
        model_params = {}
    
    model = XGBoostModel(**model_params)
    
    # Prepare features
    features_df = model.prepare_features(data, lookback_window)
    
    # Prepare target
    target = model.prepare_target(data, target_type='returns')
    
    # Ensure target is a Series with a unique name
    if not isinstance(target, pd.Series):
        raise ValueError(f"prepare_target returned {type(target)}, expected Series")
    
    # Rename target to avoid conflicts with existing columns
    target_name = 'target_returns'
    target = target.rename(target_name)
    
    # Align data by index
    aligned_data = pd.concat([features_df, target], axis=1).dropna()
    
    if len(aligned_data) < 50:
        raise ValueError(f"Insufficient aligned data: {len(aligned_data)} rows after feature engineering.")
    
    # Select feature columns (exclude target and metadata columns)
    feature_cols = [col for col in aligned_data.columns 
                   if col not in ['close', 'open', 'high', 'low', 'volume', 'vwap', 'num_trades',
                                 'ticker', 'datetime_et', 'datetime_utc', 'date_et', target_name]]
    
    if not feature_cols:
        raise ValueError("No valid feature columns found")
    
    X = aligned_data[feature_cols]
    y = aligned_data[target_name]
    
    # Split train/test
    split_idx = int(len(X) * train_test_split)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # Train model
    print(f"Training model on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    if mean_absolute_error is None:
        # Fallback if sklearn not available
        train_mae = np.mean(np.abs(y_train - train_pred))
        test_mae = np.mean(np.abs(y_test - test_pred))
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test - test_pred) ** 2))
        train_r2 = 1 - np.sum((y_train - train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
        test_r2 = 1 - np.sum((y_test - test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    else:
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
    
    metrics = {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': len(feature_cols)
    }
    
    print(f"\nModel Performance:")
    print(f"  Train MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}")
    print(f"  Test MAE:  {test_mae:.6f}, RMSE: {test_rmse:.6f}, R²: {test_r2:.4f}")
    
    # Save model if path provided
    if save_model_path:
        model.save_model(save_model_path)
        print(f"\nModel saved to: {save_model_path}")
    
    return model, metrics


def predict_from_processed_data(model: XGBoostModel,
                                processed_file: str,
                                ticker: Optional[str] = None,
                                lookback_window: int = 10,
                                n_predictions: int = 1) -> pd.DataFrame:
    # Generate predictions from processed data using trained model
    # Load data
    data = load_processed_data(processed_file, ticker=ticker)
    
    if len(data) < lookback_window + 10:
        raise ValueError(f"Insufficient data: {len(data)} rows")
    
    # Prepare features
    features_df = model.prepare_features(data, lookback_window)
    
    if len(features_df) == 0:
        raise ValueError("No features available after preparation")
    
    # Get feature columns
    feature_cols = [col for col in features_df.columns 
                   if col not in ['close', 'open', 'high', 'low', 'volume', 'vwap', 'num_trades',
                                 'ticker', 'datetime_et', 'datetime_utc', 'date_et']]
    
    # Use last n_predictions rows for prediction
    X_predict = features_df[feature_cols].iloc[-n_predictions:]
    
    # Generate predictions
    predictions = model.predict(X_predict)
    
    # Create results DataFrame
    results_data = {
        'current_price': data['close'].iloc[-n_predictions:].values,
        'predicted_return': predictions,
        'predicted_price': data['close'].iloc[-n_predictions:].values * (1 + predictions)
    }
    
    if 'datetime_et' in data.columns:
        results_data['datetime_et'] = data['datetime_et'].iloc[-n_predictions:].values
    if 'ticker' in data.columns:
        results_data['ticker'] = data['ticker'].iloc[-n_predictions:].values
    elif ticker:
        results_data['ticker'] = [ticker] * n_predictions
    
    results = pd.DataFrame(results_data)
    
    return results


def train_models_for_all_tickers(processed_file: str,
                                 model_dir: str = "models",
                                 lookback_window: int = 10,
                                 model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    # Train separate models for each ticker
    # Load data to get tickers
    data = load_processed_data(processed_file)
    tickers = data['ticker'].unique()
    
    # Create model directory
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Training model for {ticker}")
        print(f"{'='*60}")
        
        try:
            model_path = str(Path(model_dir) / f"model_{ticker}.pkl")
            model, metrics = train_model_on_processed_data(
                processed_file=processed_file,
                ticker=ticker,
                lookback_window=lookback_window,
                model_params=model_params,
                save_model_path=model_path
            )
            results[ticker] = metrics
        except Exception as e:
            print(f"Error training model for {ticker}: {e}")
            results[ticker] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Main script to train ML models
    import argparse
    
    parser = argparse.ArgumentParser(description='Train XGBoost models on processed data')
    parser.add_argument('--processed-file', type=str, required=True,
                       help='Path to processed CSV file')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Specific ticker to train on (if None, trains on all)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--lookback-window', type=int, default=10,
                       help='Lookback window for features')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of XGBoost estimators')
    parser.add_argument('--max-depth', type=int, default=5,
                       help='Max tree depth')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate
    }
    
    if args.ticker:
        # Train single model
        print(f"Training model for {args.ticker}...")
        model_path = str(Path(args.model_dir) / f"model_{args.ticker}.pkl")
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        
        model, metrics = train_model_on_processed_data(
            processed_file=args.processed_file,
            ticker=args.ticker,
            lookback_window=args.lookback_window,
            model_params=model_params,
            save_model_path=model_path
        )
        
        print(f"\n✓ Model training completed for {args.ticker}")
    else:
        # Train models for all tickers
        print("Training models for all tickers...")
        results = train_models_for_all_tickers(
            processed_file=args.processed_file,
            model_dir=args.model_dir,
            lookback_window=args.lookback_window,
            model_params=model_params
        )
        
        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        for ticker, metrics in results.items():
            if 'error' not in metrics:
                print(f"{ticker}: Test R² = {metrics.get('test_r2', 0):.4f}, Test RMSE = {metrics.get('test_rmse', 0):.6f}")
            else:
                print(f"{ticker}: ERROR - {metrics['error']}")
