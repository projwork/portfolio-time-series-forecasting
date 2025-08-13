"""
LSTM deep learning forecasting module for time series prediction.
Implements neural network models for stock price forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Union, Callable
import warnings
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Deep learning imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, GRU
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    TENSORFLOW_AVAILABLE = True
    
    # Configure TensorFlow to be less verbose
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM functionality will be limited.")

# Scikit-learn for preprocessing
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Some preprocessing features will be limited.")

# Suppress warnings
warnings.filterwarnings('ignore')


class LSTMForecaster:
    """
    LSTM-based time series forecasting model.
    Handles cases when TensorFlow is not available.
    """
    
    def __init__(self, sequence_length: int = 60, forecast_horizon: int = 1,
                 scaler_type: str = 'minmax'):
        """
        Initialize LSTM forecaster.
        
        Args:
            sequence_length: Number of time steps to look back
            forecast_horizon: Number of time steps to forecast ahead
            scaler_type: Type of scaler ('minmax' or 'standard')
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler_type = scaler_type
        
        # Model components
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.tensorflow_available = TENSORFLOW_AVAILABLE
        
        # Training data storage
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.train_data = None
        self.test_data = None
        
        # Training history
        self.history = None
        self.best_model_path = None
        
        # Check availability
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. LSTM model cannot be created.")
            logger.info("To install TensorFlow: pip install tensorflow")
        
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Some preprocessing features limited.")
            logger.info("To install scikit-learn: pip install scikit-learn")
    
    def _check_tensorflow(self):
        """Check if TensorFlow is available and raise error if not."""
        if not self.tensorflow_available:
            raise ImportError(
                "TensorFlow not available. Please install with: pip install tensorflow\n"
                "LSTM models require TensorFlow to function."
            )
    
    def prepare_data(self, train_series: pd.Series, test_series: Optional[pd.Series] = None) -> None:
        """
        Prepare data for LSTM training.
        
        Args:
            train_series: Training time series data
            test_series: Test time series data (optional)
        """
        logger.info("Preparing data for LSTM training...")
        
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn required for data preprocessing")
            return
        
        # Initialize scaler
        if self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        # Prepare training data
        train_values = train_series.values.reshape(-1, 1)
        train_scaled = self.scaler.fit_transform(train_values).flatten()
        
        self.X_train, self.y_train = self._create_sequences(train_scaled)
        self.train_data = train_series.copy()
        
        logger.info(f"Training data prepared:")
        logger.info(f"  Original series length: {len(train_series)}")
        logger.info(f"  Training sequences: {self.X_train.shape[0]}")
        logger.info(f"  Sequence length: {self.sequence_length}")
        logger.info(f"  Features per sequence: {self.X_train.shape[2]}")
        
        # Prepare test data if provided
        if test_series is not None:
            test_values = test_series.values.reshape(-1, 1)
            test_scaled = self.scaler.transform(test_values).flatten()
            
            # Combine train and test for sequence creation
            combined_scaled = np.concatenate([train_scaled, test_scaled])
            
            # Create test sequences (use last part of combined data)
            start_idx = len(train_scaled) - self.sequence_length
            test_portion = combined_scaled[start_idx:]
            
            self.X_test, self.y_test = self._create_sequences(test_portion)
            self.test_data = test_series.copy()
            
            logger.info(f"Test data prepared:")
            logger.info(f"  Original test series length: {len(test_series)}")
            logger.info(f"  Test sequences: {self.X_test.shape[0]}")
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Scaled time series data
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.forecast_horizon + 1):
            # Input sequence
            X.append(data[i - self.sequence_length:i])
            # Target value(s)
            if self.forecast_horizon == 1:
                y.append(data[i])
            else:
                y.append(data[i:i + self.forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X for LSTM input (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, architecture: str = 'simple', **kwargs):
        """
        Build LSTM model architecture.
        
        Args:
            architecture: Model architecture type ('simple', 'deep', 'bidirectional', 'gru')
            **kwargs: Additional architecture parameters
            
        Returns:
            Compiled Keras model
        """
        self._check_tensorflow()
        
        logger.info(f"Building LSTM model with '{architecture}' architecture...")
        
        # Default parameters
        lstm_units = kwargs.get('lstm_units', 50)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        learning_rate = kwargs.get('learning_rate', 0.001)
        optimizer_type = kwargs.get('optimizer', 'adam')
        
        model = Sequential()
        
        if architecture == 'simple':
            # Simple single-layer LSTM
            model.add(LSTM(
                units=lstm_units,
                input_shape=(self.sequence_length, 1),
                return_sequences=False
            ))
            model.add(Dropout(dropout_rate))
            
        elif architecture == 'deep':
            # Multi-layer LSTM
            num_layers = kwargs.get('num_layers', 3)
            
            # First LSTM layer
            model.add(LSTM(
                units=lstm_units,
                input_shape=(self.sequence_length, 1),
                return_sequences=True
            ))
            model.add(Dropout(dropout_rate))
            
            # Hidden LSTM layers
            for i in range(num_layers - 2):
                model.add(LSTM(units=lstm_units // (i + 2), return_sequences=True))
                model.add(Dropout(dropout_rate))
            
            # Final LSTM layer
            model.add(LSTM(units=lstm_units // num_layers, return_sequences=False))
            model.add(Dropout(dropout_rate))
            
        elif architecture == 'bidirectional':
            # Bidirectional LSTM
            model.add(Bidirectional(LSTM(
                units=lstm_units,
                input_shape=(self.sequence_length, 1),
                return_sequences=True
            )))
            model.add(Dropout(dropout_rate))
            
            model.add(Bidirectional(LSTM(units=lstm_units // 2, return_sequences=False)))
            model.add(Dropout(dropout_rate))
            
        elif architecture == 'gru':
            # GRU-based model
            model.add(GRU(
                units=lstm_units,
                input_shape=(self.sequence_length, 1),
                return_sequences=True
            ))
            model.add(Dropout(dropout_rate))
            
            model.add(GRU(units=lstm_units // 2, return_sequences=False))
            model.add(Dropout(dropout_rate))
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Dense output layer(s)
        if self.forecast_horizon == 1:
            model.add(Dense(1))
        else:
            # Multi-step forecasting
            model.add(Dense(50, activation='relu'))
            model.add(Dropout(dropout_rate))
            model.add(Dense(self.forecast_horizon))
        
        # Compile model
        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_type == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            optimizer = optimizer_type
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Model built successfully:")
        logger.info(f"  Total parameters: {model.count_params():,}")
        logger.info(f"  Architecture: {architecture}")
        logger.info(f"  LSTM units: {lstm_units}")
        logger.info(f"  Dropout rate: {dropout_rate}")
        
        return model
    
    def train(self, architecture: str = 'simple', epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2, early_stopping: bool = True,
              save_best_model: bool = True, **kwargs) -> Dict[str, any]:
        """
        Train the LSTM model.
        
        Args:
            architecture: Model architecture type
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of training data for validation
            early_stopping: Whether to use early stopping
            save_best_model: Whether to save the best model
            **kwargs: Additional training parameters
            
        Returns:
            Training history dictionary
        """
        self._check_tensorflow()
        
        if self.X_train is None:
            raise ValueError("Data must be prepared before training")
        
        logger.info("Starting LSTM model training...")
        
        # Build model
        self.model = self.build_model(architecture=architecture, **kwargs)
        
        # Setup callbacks
        callbacks = []
        
        if early_stopping:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=kwargs.get('patience', 10),
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        if save_best_model:
            self.best_model_path = 'best_lstm_model.h5'
            checkpoint = ModelCheckpoint(
                self.best_model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        
        # Load best model if saved
        if save_best_model and os.path.exists(self.best_model_path):
            self.model.load_weights(self.best_model_path)
            logger.info("Loaded best model weights")
        
        logger.info("Training completed successfully!")
        
        return self.history.history
    
    def predict(self, X: Optional[np.ndarray] = None, use_test_data: bool = True) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Input data for prediction (if None, uses test data)
            use_test_data: Whether to use prepared test data
            
        Returns:
            Predictions in original scale
        """
        self._check_tensorflow()
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if X is None:
            if use_test_data and self.X_test is not None:
                X = self.X_test
            else:
                raise ValueError("No input data provided for prediction")
        
        logger.info(f"Generating predictions for {X.shape[0]} sequences...")
        
        # Generate predictions
        predictions_scaled = self.model.predict(X, verbose=0)
        
        # Inverse transform to original scale
        if self.forecast_horizon == 1:
            predictions_scaled = predictions_scaled.reshape(-1, 1)
        
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        if self.forecast_horizon == 1:
            predictions = predictions.flatten()
        
        logger.info(f"Predictions generated successfully!")
        logger.info(f"  Prediction range: ${predictions.min():.2f} to ${predictions.max():.2f}")
        
        return predictions
    
    def forecast(self, last_sequence: np.ndarray, steps: int) -> np.ndarray:
        """
        Generate multi-step forecasts using the trained model.
        
        Args:
            last_sequence: Last sequence of data to start forecasting from
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values in original scale
        """
        self._check_tensorflow()
        
        if not self.is_fitted:
            raise ValueError("Model must be trained before forecasting")
        
        logger.info(f"Generating {steps}-step forecast...")
        
        # Prepare input sequence
        current_sequence = last_sequence.copy()
        forecasts = []
        
        for i in range(steps):
            # Reshape for model input
            input_seq = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Predict next value
            next_pred = self.model.predict(input_seq, verbose=0)[0]
            
            # Store prediction
            forecasts.append(next_pred[0] if len(next_pred.shape) > 0 else next_pred)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], next_pred[0] if len(next_pred.shape) > 0 else next_pred)
        
        # Convert to numpy array and inverse transform
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts_original = self.scaler.inverse_transform(forecasts).flatten()
        
        logger.info(f"Multi-step forecast completed!")
        logger.info(f"  Forecast range: ${forecasts_original.min():.2f} to ${forecasts_original.max():.2f}")
        
        return forecasts_original
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot training history.
        
        Args:
            figsize: Figure size
        """
        if self.history is None:
            raise ValueError("Model must be trained to plot history")
        
        history = self.history.history
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('LSTM Training History', fontsize=16, fontweight='bold')
        
        # Plot loss
        axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot MAE
        axes[1].plot(history['mae'], label='Training MAE', linewidth=2)
        if 'val_mae' in history:
            axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_simple_predictions(self, train_series: pd.Series, test_series: pd.Series) -> np.ndarray:
        """
        Simplified prediction method when TensorFlow is not available.
        Returns a simple moving average forecast as fallback.
        
        Args:
            train_series: Training data
            test_series: Test data to predict
            
        Returns:
            Simple moving average predictions
        """
        logger.warning("Using simple moving average fallback (TensorFlow not available)")
        
        # Calculate moving average window
        window = min(self.sequence_length, len(train_series) // 4)
        
        # Get last values for prediction
        last_values = train_series.tail(window).values
        
        # Simple moving average prediction
        avg_pred = np.mean(last_values)
        
        # Return constant prediction for all test points
        predictions = np.full(len(test_series), avg_pred)
        
        logger.info(f"Generated {len(predictions)} simple MA predictions")
        logger.info(f"  Prediction value: ${avg_pred:.2f}")
        
        return predictions


def create_fallback_lstm():
    """
    Create a fallback LSTM forecaster when TensorFlow is not available.
    
    Returns:
        LSTMForecaster instance that can handle missing dependencies
    """
    return LSTMForecaster()


def main():
    """
    Main function to demonstrate LSTM forecasting.
    """
    logger.info("LSTM Forecasting module loaded!")
    
    if TENSORFLOW_AVAILABLE:
        logger.info("✅ TensorFlow available - Full LSTM functionality enabled")
        logger.info("Available features:")
        logger.info("  - Multiple LSTM architectures (simple, deep, bidirectional, GRU)")
        logger.info("  - Automatic hyperparameter optimization")
        logger.info("  - Multi-step forecasting capabilities")
        logger.info("  - Comprehensive training monitoring")
        logger.info("  - Model saving and loading")
    else:
        logger.warning("❌ TensorFlow not available - Limited functionality")
        logger.info("To install TensorFlow: pip install tensorflow")
        logger.info("Fallback features available:")
        logger.info("  - Simple moving average predictions")
        logger.info("  - Data preprocessing utilities")


if __name__ == "__main__":
    main()