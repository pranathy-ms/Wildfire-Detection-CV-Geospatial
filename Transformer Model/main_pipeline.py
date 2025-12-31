"""
Main pipeline for wildfire analysis
Combines data extraction, model training, and prediction
"""

import logging
import argparse
from pathlib import Path
import pandas as pd
import torch

# Import custom modules (assuming they're in the same directory)
from data_extraction import WildfireDataExtractor
from config_module import config
from transformer_model import WildfireTransformer, WildfireModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WildfirePipeline:
    """End-to-end wildfire analysis pipeline"""
    
    def __init__(self, config_obj=None):
        """
        Initialize pipeline with configuration
        
        Args:
            config_obj: Configuration object (uses global config if None)
        """
        self.config = config_obj or config
        self.extractor = None
        self.trainer = None
        self.features_df = None
        
    def step1_extract_data(self, force_reextract: bool = False):
        """
        Step 1: Extract features from all data sources
        
        Args:
            force_reextract: If True, always re-extract even if cache exists
        """
        logger.info("="*60)
        logger.info("STEP 1: Data Extraction")
        logger.info("="*60)
        
        features_path = Path(self.config.data.output_dir) / 'wildfire_features.csv'
        
        # Check if cached features exist
        if features_path.exists() and not force_reextract:
            logger.info(f"Loading cached features from {features_path}")
            self.features_df = pd.read_csv(features_path)
            logger.info(f"Loaded {len(self.features_df)} features")
            return
        
        # Initialize extractor
        self.extractor = WildfireDataExtractor(
            viirs_path=self.config.data.viirs_geojson,
            era5_path=self.config.data.era5_netcdf,
            dem_path=self.config.data.dem_tif
        )
        
        # Load all data
        self.extractor.load_all_data()
        
        # Clip to DEM bounds
        self.extractor.clip_to_dem_bounds()
        
        # Extract features
        self.features_df = self.extractor.extract_features()
        
        # Save features
        self.features_df.to_csv(features_path, index=False)
        logger.info(f"Saved features to {features_path}")
        
        # Print summary statistics
        stats = self.extractor.get_summary_stats(self.features_df)
        logger.info("\n=== Data Summary ===")
        logger.info(f"Total features: {stats['total_features']}")
        logger.info(f"Confidence distribution: {stats['confidence_distribution']}")
        logger.info(f"FRP range: {stats['frp_stats']['min']:.2f} - {stats['frp_stats']['max']:.2f} MW")
        logger.info(f"Elevation range: {stats['elevation_stats']['min']:.1f} - {stats['elevation_stats']['max']:.1f} m")
        logger.info(f"Slope range: {stats['slope_stats']['min']:.1f} - {stats['slope_stats']['max']:.1f}°")
        
    def step2_train_model(self, model_type: str = 'transformer'):
        """
        Step 2: Train prediction model
        
        Args:
            model_type: 'transformer' or 'random_forest'
        """
        logger.info("="*60)
        logger.info(f"STEP 2: Training {model_type.upper()} Model")
        logger.info("="*60)
        
        if self.features_df is None:
            raise ValueError("Must run step1_extract_data first")
        
        if model_type == 'transformer':
            self._train_transformer()
        elif model_type == 'random_forest':
            self._train_random_forest()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_transformer(self):
        """Train transformer model"""
        # Initialize model
        model = WildfireTransformer(
            input_dim=len(self.config.model.feature_columns),
            d_model=self.config.model.d_model,
            nhead=self.config.model.nhead,
            num_layers=self.config.model.num_layers,
            dim_feedforward=self.config.model.dim_feedforward,
            dropout=self.config.model.dropout,
            num_classes=3
        )
        
        # Initialize trainer
        self.trainer = WildfireModelTrainer(model)
        
        # Prepare data
        train_loader, val_loader = self.trainer.prepare_data(
            self.features_df,
            self.config.model.feature_columns,
            self.config.model.target_column,
            test_size=self.config.model.test_size,
            batch_size=32
        )
        
        # Train
        self.trainer.train(
            train_loader,
            val_loader,
            num_epochs=50,
            learning_rate=0.001,
            weight_decay=1e-4
        )
        
        # Save model
        model_path = Path(self.config.data.output_dir) / 'transformer_model.pth'
        self.trainer.save_checkpoint(str(model_path))
        
        # Log final results
        final_val_acc = self.trainer.history['val_acc'][-1]
        logger.info(f"\nFinal Validation Accuracy: {final_val_acc:.2f}%")
        
    def _train_random_forest(self):
        """Train random forest model (baseline)"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        import joblib
        
        # Prepare data
        X = self.features_df[self.config.model.feature_columns].values
        y = self.features_df[self.config.model.target_column].values
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
            stratify=y
        )
        
        # Train model
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=self.config.model.n_estimators,
            max_depth=self.config.model.max_depth,
            min_samples_split=self.config.model.min_samples_split,
            class_weight='balanced',
            random_state=self.config.model.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        val_acc = accuracy_score(y_val, model.predict(X_val))
        logger.info(f"\nValidation Accuracy: {val_acc*100:.2f}%")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.config.model.feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nFeature Importances:")
        logger.info(importance_df.to_string(index=False))
        
        # Save model
        model_path = Path(self.config.data.output_dir) / 'random_forest_model.pkl'
        joblib.dump(model, model_path)
        logger.info(f"\nModel saved to {model_path}")
        
    def step3_generate_predictions(self, model_path: str = None):
        """
        Step 3: Generate predictions on full dataset
        
        Args:
            model_path: Path to saved model (uses latest if None)
        """
        logger.info("="*60)
        logger.info("STEP 3: Generating Predictions")
        logger.info("="*60)
        
        if self.features_df is None:
            raise ValueError("Must run step1_extract_data first")
        
        # Load model if not already loaded
        if model_path is None:
            model_path = Path(self.config.data.output_dir) / 'transformer_model.pth'
        
        if self.trainer is None:
            model = WildfireTransformer(
                input_dim=len(self.config.model.feature_columns),
                d_model=self.config.model.d_model,
                nhead=self.config.model.nhead,
                num_layers=self.config.model.num_layers,
                num_classes=3
            )
            self.trainer = WildfireModelTrainer(model)
            self.trainer.load_checkpoint(str(model_path))
        
        # Generate predictions
        X = self.features_df[self.config.model.feature_columns].values
        predictions = self.trainer.predict(X)
        
        self.features_df['predicted_confidence'] = predictions
        self.features_df['predicted_confidence_label'] = self.features_df['predicted_confidence'].map({
            0: 'l', 1: 'n', 2: 'h'
        })
        
        # Save predictions
        predictions_path = Path(self.config.data.output_dir) / 'wildfire_predictions.csv'
        self.features_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")
        
        # Summary
        comparison = pd.crosstab(
            self.features_df['confidence'],
            self.features_df['predicted_confidence_label'],
            rownames=['Actual'],
            colnames=['Predicted']
        )
        logger.info("\nConfusion Matrix:")
        logger.info(comparison)
        
    def run_full_pipeline(self, force_reextract: bool = False, model_type: str = 'transformer'):
        """
        Run complete pipeline
        
        Args:
            force_reextract: If True, re-extract data even if cached
            model_type: 'transformer' or 'random_forest'
        """
        logger.info("Starting Full Wildfire Analysis Pipeline")
        logger.info("="*60)
        
        try:
            # Step 1: Extract data
            self.step1_extract_data(force_reextract=force_reextract)
            
            # Step 2: Train model
            self.step2_train_model(model_type=model_type)
            
            # Step 3: Generate predictions
            self.step3_generate_predictions()
            
            logger.info("="*60)
            logger.info("Pipeline completed successfully!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Wildfire Analysis Pipeline')
    parser.add_argument(
        '--step',
        choices=['extract', 'train', 'predict', 'full'],
        default='full',
        help='Pipeline step to run'
    )
    parser.add_argument(
        '--model',
        choices=['transformer', 'random_forest'],
        default='transformer',
        help='Model type to use'
    )
    parser.add_argument(
        '--force-reextract',
        action='store_true',
        help='Force re-extraction of features'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = WildfirePipeline()
    
    # Run requested step
    if args.step == 'full':
        pipeline.run_full_pipeline(
            force_reextract=args.force_reextract,
            model_type=args.model
        )
    elif args.step == 'extract':
        pipeline.step1_extract_data(force_reextract=args.force_reextract)
    elif args.step == 'train':
        pipeline.step1_extract_data()  # Load cached data
        pipeline.step2_train_model(model_type=args.model)
    elif args.step == 'predict':
        pipeline.step1_extract_data()  # Load cached data
        pipeline.step3_generate_predictions()


if __name__ == "__main__":
    main()
