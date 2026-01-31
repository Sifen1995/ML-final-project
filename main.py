"""
CLI tool for Student Grade Prediction pipeline.
"""
import click
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader, load_train_test_split
from src.feature_engineer import StudentFeatureEngineer, prepare_features
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer, train_all_models
from src.model_evaluator import ModelEvaluator
from src.predictor import load_predictor
from src.config import data_config, model_config
from src.logger import setup_logger

logger = setup_logger(__name__, "main.log")


@click.group()
def cli():
    """Student Grade Prediction CLI - Predict academic performance using ML."""
    pass


@cli.command()
@click.option('--dataset', type=click.Choice(['math', 'portuguese', 'both']), 
              default='math', help='Dataset to preprocess')
def preprocess(dataset):
    """Preprocess student data and save processed features."""
    click.echo(f"Preprocessing {dataset} dataset...")
    
    loader = DataLoader()
    engineer = StudentFeatureEngineer()
    
    datasets = ['math', 'portuguese'] if dataset == 'both' else [dataset]
    
    for ds in datasets:
        try:
            click.echo(f"\nProcessing {ds} dataset...")
            
            # Load data
            if ds == 'math':
                df = loader.load_math_data()
            else:
                df = loader.load_portuguese_data()
            
            click.echo(f"  Loaded {len(df)} records")
            
            # Feature engineering
            df_engineered = engineer.engineer_features(df, fit=True)
            click.echo(f"  Engineered {len(df_engineered.columns)} features")
            
            # Prepare final features
            df_processed = prepare_features(df, ds)
            click.echo(f"  Final features: {len(df_processed.columns)}")
            
            # Save processed data
            loader.save_processed_data(df_processed, ds)
            click.echo(f"  ✓ Saved processed data for {ds}")
            
        except FileNotFoundError as e:
            click.echo(f"  ✗ Error: {e}", err=True)
            continue
        except Exception as e:
            click.echo(f"  ✗ Error processing {ds}: {e}", err=True)
            continue
    
    click.echo("\n✓ Preprocessing complete!")


@cli.command()
@click.option('--dataset', type=click.Choice(['math', 'portuguese', 'both']),
              default='math', help='Dataset to train on')
@click.option('--model', type=click.Choice(['linear_regression', 'random_forest', 'xgboost', 'all']),
              default='all', help='Model type to train')
def train(dataset, model):
    """Train grade prediction models."""
    click.echo(f"Training {model} model(s) on {dataset} dataset...")
    
    datasets = ['math', 'portuguese'] if dataset == 'both' else [dataset]
    models = model_config.model_types if model == 'all' else [model]
    
    for ds in datasets:
        try:
            click.echo(f"\nTraining on {ds} dataset...")
            
            # Load train/test split
            X_train, X_test, y_train, y_test = load_train_test_split(ds)
            click.echo(f"  Training samples: {len(X_train)}")
            click.echo(f"  Test samples: {len(X_test)}")
            
            # Preprocess
            preprocessor = DataPreprocessor()
            X_train_proc, y_train = preprocessor.fit_transform(X_train, y_train)
            
            # Save scaler for later use
            import joblib
            scaler_path = Path("models") / f"scaler_{ds}.joblib"
            joblib.dump(preprocessor.scaler, scaler_path)
            # Also save as default scaler
            joblib.dump(preprocessor.scaler, Path("models/scaler.joblib"))
            
            # Train models
            for model_type in models:
                click.echo(f"\n  Training {model_type}...")
                
                trainer = ModelTrainer(model_type=model_type)
                metrics = trainer.train(X_train_proc, y_train, validate=True)
                
                click.echo(f"    Train R²: {metrics['train_r2']:.4f}")
                if 'cv_mean_r2' in metrics:
                    click.echo(f"    CV R²: {metrics['cv_mean_r2']:.4f} (+/- {metrics['cv_std_r2']:.4f})")
                    click.echo(f"    CV RMSE: {metrics['cv_mean_rmse']:.4f}")
                
                # Save model
                trainer.save_model(dataset_name=ds)
                click.echo(f"    ✓ Model saved")
                
        except FileNotFoundError as e:
            click.echo(f"  ✗ Error: {e}. Run 'preprocess' first.", err=True)
            continue
        except Exception as e:
            click.echo(f"  ✗ Error training on {ds}: {e}", err=True)
            continue
    
    click.echo("\n✓ Training complete!")


@cli.command()
@click.option('--dataset', type=click.Choice(['math', 'portuguese']),
              default='math', help='Dataset to evaluate')
@click.option('--model', type=click.Choice(['linear_regression', 'random_forest', 'xgboost', 'all']),
              default='all', help='Model type to evaluate')
def evaluate(dataset, model):
    """Evaluate trained models on test data."""
    click.echo(f"Evaluating {model} model(s) on {dataset} dataset...")
    
    models = model_config.model_types if model == 'all' else [model]
    
    try:
        # Load test data
        X_train, X_test, y_train, y_test = load_train_test_split(dataset)
        
        # Preprocess
        preprocessor = DataPreprocessor()
        X_train_proc, _ = preprocessor.fit_transform(X_train, y_train)
        X_test_proc = preprocessor.transform(X_test)
        
        evaluator = ModelEvaluator()
        
        results = []
        for model_type in models:
            try:
                click.echo(f"\nEvaluating {model_type}...")
                
                # Load model
                predictor = load_predictor(model_type, dataset)
                
                # Make predictions
                y_pred = predictor.predict(X_test_proc)
                
                # Evaluate
                metrics = evaluator.evaluate(y_test, y_pred)
                
                results.append({
                    'Model': model_type,
                    'R²': metrics['r2'],
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'Within 2pt': f"{metrics['accuracy_within_2pt']:.1f}%"
                })
                
                click.echo(f"  R²: {metrics['r2']:.4f}")
                click.echo(f"  RMSE: {metrics['rmse']:.4f}")
                click.echo(f"  MAE: {metrics['mae']:.4f}")
                click.echo(f"  Within 2 points: {metrics['accuracy_within_2pt']:.1f}%")
                
            except FileNotFoundError:
                click.echo(f"  ✗ Model {model_type} not found. Train it first.", err=True)
                continue
        
        # Print summary
        if results:
            click.echo("\n" + "="*60)
            click.echo("EVALUATION SUMMARY")
            click.echo("="*60)
            for r in results:
                click.echo(f"{r['Model']:20s} R²={r['R²']:.4f}  RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}")
            
    except FileNotFoundError as e:
        click.echo(f"✗ Error: {e}. Run 'preprocess' first.", err=True)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)


@cli.command()
@click.option('--dataset', type=click.Choice(['math', 'portuguese']),
              default='math', help='Dataset type')
@click.option('--model', type=click.Choice(['linear_regression', 'random_forest', 'xgboost']),
              default='random_forest', help='Model to use')
@click.option('--input', 'input_file', type=click.Path(exists=True),
              help='JSON file with student data')
def predict(dataset, model, input_file):
    """Make grade predictions for new students."""
    import json
    import pandas as pd
    
    click.echo(f"Making predictions using {model} model...")
    
    try:
        predictor = load_predictor(model, dataset)
        
        if input_file:
            # Load input data
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
                results = predictor.predict_batch(df, include_details=True)
                click.echo("\nPredictions:")
                click.echo(results.to_string())
            else:
                result = predictor.predict_single(data)
                click.echo("\nPrediction:")
                click.echo(f"  Predicted Grade: {result['predicted_grade']}")
                click.echo(f"  Performance Level: {result['performance_level']}")
                click.echo(f"  Pass Status: {result['pass_status']}")
                click.echo(f"  Confidence Range: {result['confidence_range']}")
        else:
            click.echo("No input file provided. Use --input to specify student data.")
            
    except FileNotFoundError as e:
        click.echo(f"✗ Error: {e}", err=True)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)


@cli.command()
def info():
    """Show project information and configuration."""
    click.echo("\n" + "="*60)
    click.echo("STUDENT GRADE PREDICTION SYSTEM")
    click.echo("="*60)
    click.echo("\nDataset Paths:")
    click.echo(f"  Math: {data_config.student_math_path}")
    click.echo(f"  Portuguese: {data_config.student_por_path}")
    click.echo(f"\nProcessed Data:")
    click.echo(f"  Math: {data_config.math_processed_path}")
    click.echo(f"  Portuguese: {data_config.por_processed_path}")
    click.echo(f"\nModel Directory: {model_config.model_save_dir}")
    click.echo(f"\nAvailable Models: {', '.join(model_config.model_types)}")
    click.echo(f"\nTarget Column: {data_config.target_column}")
    click.echo(f"Passing Grade: {data_config.passing_grade}")
    click.echo(f"Test Split: {data_config.test_size}")
    click.echo("="*60 + "\n")


if __name__ == '__main__':
    cli()
