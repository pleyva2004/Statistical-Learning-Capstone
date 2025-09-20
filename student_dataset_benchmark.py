#!/usr/bin/env python3
"""
Comprehensive Linear Regression Performance Benchmark
Using REAL Student Performance Dataset
"""

import time
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Optional
import sys
import os

# Add current directory to path to import loadData
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loadData import load_data

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class StudentDatasetBenchmark:
    def __init__(self, l2_reg: float = 1e-3):
        """Initialize benchmark with your actual student dataset"""
        self.l2_reg = l2_reg
        self.results = {}
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.target_name = "Performance Index"
        
        print(f"🎓 STUDENT PERFORMANCE DATASET BENCHMARK")
        print(f"L2 Regularization: {l2_reg}")
        print("=" * 60)
        
    def load_and_preprocess_data(self):
        """Load your actual student dataset and preprocess it"""
        print("📂 Loading your actual student dataset...")
        
        try:
            # Load using your loadData function
            df = load_data()
            print(f"✅ Dataset loaded successfully!")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("   Make sure .env file exists with DATA_PATH variable")
            return False
        
        # Define features as in your original project
        self.feature_names = [
            "Hours Studied",
            "Previous Scores",
            "Extracurricular Activities",
            "Sleep Hours",
            "Sample Question Papers Practiced"
        ]
        
        # Verify all required columns exist
        
        missing_cols = [col for col in self.feature_names + [self.target_name] if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            print(f"   Available columns: {list(df.columns)}")
            return False
        
        # Extract features and target
        X = df[self.feature_names].copy()
        y = df[self.target_name].astype(float)
        
        # Preprocess: Convert 'Yes'/'No' to 1/0 for Extracurricular Activities
        if X["Extracurricular Activities"].dtype == object:
            print("🔄 Converting 'Extracurricular Activities' from Yes/No to 1/0...")
            X["Extracurricular Activities"] = X["Extracurricular Activities"].str.strip().str.lower().map({"yes": 1, "no": 0})
            
        # Check for missing values
        if X.isnull().any().any() or y.isnull().any():
            print("⚠️  Found missing values, dropping rows with NaN...")
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            print(f"   Remaining samples: {len(X)}")
        
        # Convert to numpy arrays with appropriate dtypes
        X_values = X.values.astype(np.float32)
        y_values = y.values.astype(np.float32)
        
        # Train/test split
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_values, y_values, test_size=0.2, random_state=42
        )
        
        print(f"📊 Dataset Statistics:")
        print(f"   Total samples: {len(X_values):,}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Training samples: {self.X_train.shape[0]:,}")
        print(f"   Test samples: {self.X_test.shape[0]:,}")
        print(f"   Target range: [{y.min():.1f}, {y.max():.1f}]")
        print()
        
        return True

    def method_1_numpy_analytical(self) -> Tuple[float, np.ndarray]:
        """🏆 Method 1: Pure NumPy Analytical Solution"""
        print("🧮 Testing Method 1: NumPy Analytical Solution...")
        
        start_time = time.time()
        
        # Add bias column
        X_with_bias = np.column_stack([np.ones(self.X_train.shape[0]), self.X_train])
        
        # Normal equation: β = (X'X + λI)^(-1) X'y
        XTX = X_with_bias.T @ X_with_bias
        
        # Add L2 regularization (don't regularize bias term)
        if self.l2_reg > 0:
            reg_matrix = np.eye(XTX.shape[0]) * self.l2_reg
            reg_matrix[0, 0] = 0  # No regularization on bias
            XTX += reg_matrix
        
        XTy = X_with_bias.T @ self.y_train
        coefficients = np.linalg.solve(XTX, XTy)
        
        train_time = time.time() - start_time
        
        # Make predictions
        X_test_bias = np.column_stack([np.ones(self.X_test.shape[0]), self.X_test])
        predictions = X_test_bias @ coefficients
        
        # Print coefficients for interpretation
        print(f"   📊 Learned coefficients:")
        print(f"      Bias: {coefficients[0]:.4f}")
        for i, feature_name in enumerate(self.feature_names):
            print(f"      {feature_name}: {coefficients[i+1]:.4f}")
        
        return train_time, predictions

    def method_2_cupy_gpu(self) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """🚀 Method 2: CuPy GPU Acceleration"""
        print("🎮 Testing Method 2: CuPy GPU...")
        
        try:
            import cupy as cp
            
            start_time = time.time()
            
            # Transfer to GPU
            X_gpu = cp.asarray(self.X_train)
            y_gpu = cp.asarray(self.y_train)
            
            # Add bias
            ones = cp.ones((X_gpu.shape[0], 1))
            X_with_bias = cp.column_stack([ones, X_gpu])
            
            # Normal equation on GPU
            XTX = X_with_bias.T @ X_with_bias
            if self.l2_reg > 0:
                reg_matrix = cp.eye(XTX.shape[0]) * self.l2_reg
                reg_matrix[0, 0] = 0
                XTX += reg_matrix
            
            XTy = X_with_bias.T @ y_gpu
            coefficients = cp.linalg.solve(XTX, XTy)
            
            train_time = time.time() - start_time
            
            # Predictions
            X_test_gpu = cp.asarray(self.X_test)
            ones_test = cp.ones((X_test_gpu.shape[0], 1))
            X_test_bias = cp.column_stack([ones_test, X_test_gpu])
            predictions_gpu = X_test_bias @ coefficients
            predictions = cp.asnumpy(predictions_gpu)
            
            return train_time, predictions
            
        except ImportError:
            print("   ❌ CuPy not available (install: pip install cupy-cuda12x)")
            return None, None
        except Exception as e:
            print(f"   ❌ GPU error: {e}")
            return None, None

    def method_3_rapids_cuml(self) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """⚡ Method 3: RAPIDS cuML"""
        print("🌊 Testing Method 3: RAPIDS cuML...")
        
        try:
            import cupy
            
            # Better GPU availability check
            if not cupy.cuda.is_available():
                print("   ❌ CUDA not available on this system")
                return None, None
                
            # Test actual GPU access
            try:
                with cupy.cuda.Device(0):
                    test_array = cupy.array([1, 2, 3])  # Simple GPU operation
                    test_result = cupy.sum(test_array)  # Force GPU computation
            except Exception as gpu_error:
                print(f"   ❌ Cannot access GPU: {gpu_error}")
                return None, None
            
            from cuml.linear_model import Ridge as CumlRidge
            import cudf
            
            start_time = time.time()
            
            # Convert to GPU DataFrames
            X_gpu = cudf.DataFrame(self.X_train, columns=self.feature_names)
            y_gpu = cudf.Series(self.y_train)
            
            model = CumlRidge(alpha=self.l2_reg)
            model.fit(X_gpu, y_gpu)
            
            train_time = time.time() - start_time
            
            # Predictions
            X_test_gpu = cudf.DataFrame(self.X_test, columns=self.feature_names)
            predictions_gpu = model.predict(X_test_gpu)
            predictions = predictions_gpu.to_numpy()
            
            return train_time, predictions
            
        except ImportError as e:
            print(f"   ❌ RAPIDS not available: {e}")
            return None, None
        except Exception as e:
            print(f"   ❌ RAPIDS error: {e}")
            return None, None

    def method_4_sklearn_mkl(self) -> Tuple[float, np.ndarray]:
        """🔧 Method 4: scikit-learn with Intel MKL"""
        print("⚙️  Testing Method 4: scikit-learn (Intel MKL)...")
        
        from sklearn.linear_model import Ridge
        
        start_time = time.time()
        
        model = Ridge(alpha=self.l2_reg, solver='cholesky')  # Fastest solver
        model.fit(self.X_train, self.y_train)
        
        train_time = time.time() - start_time
        
        predictions = model.predict(self.X_test)
        
        # Print coefficients for comparison
        print(f"   📊 Learned coefficients:")
        print(f"      Bias: {model.intercept_:.4f}")
        for i, feature_name in enumerate(self.feature_names):
            print(f"      {feature_name}: {model.coef_[i]:.4f}")
        
        return train_time, predictions

    def method_5_tensorflow(self) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """🧠 Method 5: TensorFlow/Keras (your current approach)"""
        print("🤖 Testing Method 5: TensorFlow (your current method)...")
        
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')  # Suppress TF warnings
            
            start_time = time.time()
            
            # Normalization (as in your original code)
            normalizer = tf.keras.layers.Normalization()
            normalizer.adapt(self.X_train)
            
            # Model (matching your original architecture)
            inputs = tf.keras.Input(shape=(self.X_train.shape[1],))
            x = normalizer(inputs)
            outputs = tf.keras.layers.Dense(
                1, 
                activation=None,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
            )(x)
            model = tf.keras.Model(inputs, outputs)
            
            model.compile(optimizer="adam", loss="mse")
            model.fit(self.X_train, self.y_train, epochs=300, verbose=0)
            
            train_time = time.time() - start_time
            
            predictions = model.predict(self.X_test, verbose=0).squeeze()
            
            # Extract and denormalize coefficients (as in your original code)
            W, b = model.layers[-1].get_weights()      
            means = normalizer.mean.numpy()      
            stds = np.sqrt(normalizer.variance.numpy())
            
            print(f"   📊 Learned coefficients (denormalized):")
            print(f"      Bias: {b[0]:.4f}")
            for i, feature_name in enumerate(self.feature_names):
                denorm_coeff = W[i, 0] / stds[i]
                print(f"      {feature_name}: {denorm_coeff}")
            
            return train_time, predictions
            
        except ImportError:
            print("   ❌ TensorFlow not available")
            return None, None
        except Exception as e:
            print(f"   ❌ TensorFlow error: {e}")
            return None, None

    def method_6_dask_distributed(self) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """🌐 Method 6: Dask Distributed Computing"""
        print("🌍 Testing Method 6: Dask Distributed...")
        
        try:
            import dask.array as da
            from dask_ml.linear_model import LinearRegression as DaskLinearRegression
            
            start_time = time.time()
            
            # Convert to Dask arrays (appropriate chunks for your data size)
            chunk_size = max(100, self.X_train.shape[0] // 10)  # Adaptive chunking
            X_dask = da.from_array(self.X_train, chunks=(chunk_size, self.X_train.shape[1]))
            y_dask = da.from_array(self.y_train, chunks=chunk_size)
            
            model = DaskLinearRegression()
            model.fit(X_dask, y_dask)
            
            train_time = time.time() - start_time
            
            # Predictions
            X_test_dask = da.from_array(self.X_test, chunks=(chunk_size, self.X_test.shape[1]))
            predictions = model.predict(X_test_dask).compute()
            
            return train_time, predictions
            
        except ImportError:
            print("   ❌ Dask not available (install: pip install dask-ml)")
            return None, None
        except Exception as e:
            print(f"   ❌ Dask error: {e}")
            return None, None

    def calculate_metrics(self, predictions: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics"""
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        r2 = r2_score(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        rmse = mean_squared_error(self.y_test, predictions)
        
        return {'r2': r2, 'mae': mae, 'rmse': rmse}

    def run_all_benchmarks(self):
        """Run all available benchmark methods on your student dataset"""
        if not self.load_and_preprocess_data():
            print("❌ Failed to load dataset. Exiting...")
            return []
        
        methods = [
            ("NumPy Analytical", self.method_1_numpy_analytical),
            ("CuPy GPU", self.method_2_cupy_gpu),
            ("RAPIDS cuML", self.method_3_rapids_cuml),
            ("scikit-learn MKL", self.method_4_sklearn_mkl),
            ("TensorFlow", self.method_5_tensorflow),
            ("Dask Distributed", self.method_6_dask_distributed),
        ]
        
        successful_results = []
        
        for method_name, method_func in methods:
            try:
                train_time, predictions = method_func()
                
                if train_time is not None and predictions is not None:
                    metrics = self.calculate_metrics(predictions)
                    
                    successful_results.append({
                        'method': method_name,
                        'train_time': train_time,
                        'r2': metrics['r2'],
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse']
                    })
                    
                    print(f"   ✅ Time: {train_time:.6f}s | R²: {metrics['r2']:.4f} | MAE: {metrics['mae']:.4f}")
                else:
                    print(f"   ❌ Failed or not available")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            print()
        
        return successful_results

    def display_results(self, results):
        """Display comprehensive benchmark results"""
        if not results:
            print("❌ No successful benchmark results!")
            return
        
        print("\n" + "="*90)
        print("🏆 STUDENT DATASET BENCHMARK RESULTS")
        print("="*90)
        
        # Sort by training time
        results_by_speed = sorted(results, key=lambda x: x['train_time'])
        
        print(f"{'Rank':<4} {'Method':<20} {'Time (s)':<12} {'R²':<8} {'MAE':<8} {'RMSE':<8}")
        print("-" * 90)
        
        for i, result in enumerate(results_by_speed, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{emoji} {i:<2} {result['method']:<20} {result['train_time']:.6f}    "
                  f"{result['r2']:.4f}   {result['mae']:.4f}   {result['rmse']:.4f}")
        
        # Winner analysis
        winner = results_by_speed[0]
        print(f"\n🎯 FASTEST METHOD FOR YOUR STUDENT DATA: {winner['method']}")
        print(f"   Training Time: {winner['train_time']:.6f} seconds")
        print(f"   Model Quality: R² = {winner['r2']:.4f}")
        print(f"   Prediction Error: MAE = {winner['mae']:.4f}")
        
        # Speed comparison
        if len(results_by_speed) > 1:
            print(f"\n📊 SPEED COMPARISON (relative to fastest):")
            baseline_time = winner['train_time']
            for result in results_by_speed[1:]:
                speedup = result['train_time'] / baseline_time
                print(f"   {result['method']:20}: {speedup:.1f}x slower")
        
        # Model quality comparison
        best_r2 = max(results, key=lambda x: x['r2'])
        print(f"\n🎯 BEST MODEL QUALITY: {best_r2['method']} (R² = {best_r2['r2']:.4f})")

def main():
    """Main execution function"""
    print("🚀 STUDENT PERFORMANCE DATASET SPEED CHALLENGE")
    print("Testing all 6 optimization approaches on your real data...\n")
    
    # Check if .env file exists``
    if not os.path.exists('.env'):
        print("⚠️  .env file not found!")
        print("   Make sure you have a .env file with DATA_PATH variable")
        print("   Example: DATA_PATH=/path/to/your/dataset")
        return
    
    # Check for required dependencies
    required_deps = ['numpy', 'pandas', 'sklearn']
    missing_deps = []
    
    for dep in required_deps:
        try:
            __import__(dep.replace('-', '_'))
            print(f"✅ {dep} available")
        except ImportError:
            missing_deps.append(dep)
            print(f"❌ {dep} required! Install: pip install {dep}")
    
    if missing_deps:
        print(f"\nPlease install missing dependencies:")
        print(f"pip install {' '.join(missing_deps)}")
        return
    
    # Check optional dependencies
    optional_deps = {
        'cupy': '💎 CuPy (GPU acceleration)',
        'cuml': '🌊 RAPIDS cuML (enterprise GPU)', 
        'tensorflow': '🧠 TensorFlow (neural networks)',
        'dask': '🌍 Dask (distributed computing)'
    }
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            print(f"✅ {description}")
        except ImportError:
            print(f"⚠️  {description} - not available")
    
    print()
    
    # Run benchmarks on your actual student dataset
    benchmark = StudentDatasetBenchmark(l2_reg=1e-3)
    results = benchmark.run_all_benchmarks()
    benchmark.display_results(results)
    
    print(f"\n💡 RECOMMENDATIONS FOR YOUR STUDENT DATASET:")
    if results:
        winner = min(results, key=lambda x: x['train_time'])
        print(f"   🏆 Use {winner['method']} for maximum speed")
        print(f"   ⚡ Training time: {winner['train_time']:.6f} seconds")
        print(f"   🎯 Model accuracy: R² = {winner['r2']:.4f}")
    
    print(f"   📈 For larger datasets: Consider GPU methods")
    print(f"   🔬 For research: TensorFlow offers more flexibility")

if __name__ == "__main__":
    main()