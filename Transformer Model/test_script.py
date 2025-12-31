"""
Local testing script for wildfire analysis pipeline
Tests each component and the full pipeline
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test 1: Check if all required packages are installed"""
    logger.info("="*60)
    logger.info("TEST 1: Checking Package Imports")
    logger.info("="*60)
    
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical operations',
        'torch': 'PyTorch for transformer model',
        'xarray': 'NetCDF data handling',
        'geopandas': 'Geospatial operations',
        'rasterio': 'Raster data handling',
        'richdem': 'DEM processing',
        'sklearn': 'Machine learning utilities',
        'folium': 'Interactive mapping'
    }
    
    failed = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {package:15} - {description}")
        except ImportError:
            logger.error(f"✗ {package:15} - {description} (NOT INSTALLED)")
            failed.append(package)
    
    if failed:
        logger.error(f"\nMissing packages: {', '.join(failed)}")
        logger.info("\nInstall missing packages with:")
        logger.info(f"pip install {' '.join(failed)}")
        return False
    
    logger.info("\n✓ All required packages installed")
    return True


def test_data_files():
    """Test 2: Check if required data files exist"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Checking Data Files")
    logger.info("="*60)
    
    required_files = {
        'custom_date_fires.geojson': 'VIIRS fire data',
        'era5_wind_la.nc': 'ERA5 wind data',
        'USGS3DEP_30m_33.5_34.5_-119.0_-118.0.tif': 'DEM elevation data'
    }
    
    missing = []
    for filepath, description in required_files.items():
        if Path(filepath).exists():
            size_mb = Path(filepath).stat().st_size / (1024**2)
            logger.info(f"✓ {filepath:45} - {description} ({size_mb:.1f} MB)")
        else:
            logger.error(f"✗ {filepath:45} - {description} (NOT FOUND)")
            missing.append(filepath)
    
    if missing:
        logger.error(f"\nMissing data files: {', '.join(missing)}")
        logger.info("\nPlease ensure all data files are in the current directory")
        return False
    
    logger.info("\n✓ All required data files present")
    return True


def test_data_extraction():
    """Test 3: Test data extraction module"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Testing Data Extraction")
    logger.info("="*60)
    
    try:
        from data_extraction import WildfireDataExtractor
        
        # Initialize extractor
        extractor = WildfireDataExtractor(
            viirs_path='custom_date_fires.geojson',
            era5_path='era5_wind_la.nc',
            dem_path='USGS3DEP_30m_33.5_34.5_-119.0_-118.0.tif'
        )
        
        # Load data
        logger.info("Loading data sources...")
        extractor.load_all_data()
        logger.info(f"✓ Loaded {len(extractor.gdf)} VIIRS points")
        
        # Clip to bounds
        extractor.clip_to_dem_bounds()
        logger.info(f"✓ Clipped to {len(extractor.gdf)} points within DEM bounds")
        
        # Extract features (just first 10 for testing)
        logger.info("Extracting features (sample)...")
        extractor.gdf = extractor.gdf.head(10)
        features_df = extractor.extract_features()
        
        logger.info(f"✓ Extracted {len(features_df)} features")
        logger.info(f"  Feature columns: {list(features_df.columns)}")
        logger.info(f"  Sample feature values:")
        logger.info(features_df.head(3).to_string())
        
        logger.info("\n✓ Data extraction test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data extraction test failed: {str(e)}", exc_info=True)
        return False


def test_transformer_model():
    """Test 4: Test transformer model initialization and forward pass"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Testing Transformer Model")
    logger.info("="*60)
    
    try:
        import torch
        from transformer_model import WildfireTransformer
        
        # Create dummy data
        batch_size = 4
        input_dim = 5  # frp, u10, v10, elevation, slope
        dummy_input = torch.randn(batch_size, input_dim)
        
        # Initialize model
        logger.info("Initializing transformer model...")
        model = WildfireTransformer(
            input_dim=input_dim,
            d_model=64,  # Smaller for testing
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            num_classes=3
        )
        
        # Test forward pass
        logger.info("Testing forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info(f"✓ Input shape: {dummy_input.shape}")
        logger.info(f"✓ Output shape: {output.shape}")
        logger.info(f"✓ Expected output shape: ({batch_size}, 3)")
        
        # Check output shape
        assert output.shape == (batch_size, 3), "Output shape mismatch"
        
        # Test predictions
        _, predictions = output.max(1)
        logger.info(f"✓ Sample predictions: {predictions.numpy()}")
        
        logger.info("\n✓ Transformer model test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Transformer model test failed: {str(e)}", exc_info=True)
        return False


def test_training_pipeline():
    """Test 5: Test training pipeline with small dataset"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Testing Training Pipeline")
    logger.info("="*60)
    
    try:
        import torch
        from transformer_model import WildfireTransformer, WildfireModelTrainer
        
        # Create synthetic dataset
        logger.info("Creating synthetic dataset...")
        n_samples = 100
        synthetic_data = pd.DataFrame({
            'frp': np.random.uniform(0.5, 100, n_samples),
            'u10': np.random.uniform(-10, 10, n_samples),
            'v10': np.random.uniform(-10, 10, n_samples),
            'elevation': np.random.uniform(0, 1500, n_samples),
            'slope': np.random.uniform(0, 80, n_samples),
            'confidence_num': np.random.choice([0, 1, 2], n_samples)
        })
        
        # Initialize model
        model = WildfireTransformer(
            input_dim=5,
            d_model=32,  # Small for fast testing
            nhead=2,
            num_layers=1,
            dim_feedforward=64,
            num_classes=3
        )
        
        # Initialize trainer
        trainer = WildfireModelTrainer(model)
        
        # Prepare data
        logger.info("Preparing data loaders...")
        feature_cols = ['frp', 'u10', 'v10', 'elevation', 'slope']
        train_loader, val_loader = trainer.prepare_data(
            synthetic_data,
            feature_cols,
            batch_size=16,
            test_size=0.2
        )
        
        # Train for a few epochs
        logger.info("Training model (5 epochs)...")
        trainer.train(
            train_loader,
            val_loader,
            num_epochs=5,
            learning_rate=0.001
        )
        
        # Test prediction
        logger.info("Testing predictions...")
        test_features = synthetic_data[feature_cols].values[:5]
        predictions = trainer.predict(test_features)
        logger.info(f"✓ Sample predictions: {predictions}")
        
        logger.info("\n✓ Training pipeline test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Training pipeline test failed: {str(e)}", exc_info=True)
        return False


def test_full_pipeline():
    """Test 6: Test full pipeline with real data"""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Testing Full Pipeline")
    logger.info("="*60)
    
    try:
        from main_pipeline import WildfirePipeline
        
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = WildfirePipeline()
        
        # Step 1: Extract data
        logger.info("\nStep 1: Extracting data...")
        pipeline.step1_extract_data(force_reextract=False)
        logger.info(f"✓ Extracted {len(pipeline.features_df)} features")
        
        # Step 2: Train model (using Random Forest for speed)
        logger.info("\nStep 2: Training model (Random Forest for speed)...")
        pipeline.step2_train_model(model_type='random_forest')
        logger.info("✓ Model training complete")
        
        # Check if predictions file exists
        predictions_file = Path('outputs/wildfire_predictions.csv')
        if predictions_file.exists():
            predictions = pd.read_csv(predictions_file)
            logger.info(f"✓ Predictions generated: {len(predictions)} rows")
        
        logger.info("\n✓ Full pipeline test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Full pipeline test failed: {str(e)}", exc_info=True)
        return False


def run_all_tests():
    """Run all tests in sequence"""
    logger.info("="*60)
    logger.info("WILDFIRE ANALYSIS PIPELINE - TEST SUITE")
    logger.info("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Files", test_data_files),
        ("Data Extraction", test_data_extraction),
        ("Transformer Model", test_transformer_model),
        ("Training Pipeline", test_training_pipeline),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {str(e)}")
            results[test_name] = False
        
        # Add separator
        logger.info("\n")
    
    # Final summary
    logger.info("="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:25} {status}")
    
    total = len(results)
    passed = sum(results.values())
    logger.info("="*60)
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Your pipeline is ready to use.")
    else:
        logger.warning(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before proceeding.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    
    # Run tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
