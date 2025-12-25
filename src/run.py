"""
Command-line interface for volatility estimator stack.

Usage:
    python src/run.py --symbol SPY --estimator close_to_close --window 60
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.comparison import (
    run_all_estimators,
    calculate_correlation_matrix,
    calculate_mse_matrix,
    generate_comparison_statistics
)
from src.data_loader import get_market_data, check_data_quality
from src.event_analysis import analyze_all_events
from src.estimators import get_estimator, list_estimators
from src.predictions import (
    build_pattern_database,
    find_similar_events,
    predict_volatility_path,
    backtest_predictions
)
from src.reporting import (
    export_to_excel,
    plot_volatility_comparison,
    plot_predictions,
    generate_summary_report
)
from src.utils import load_events, setup_logging, parse_date

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary with configuration
    """
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return {}

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config or {}




def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Volatility Estimator Stack - Calculate volatility estimates from market data'
    )

    # Required arguments
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Asset symbol (e.g., SPY, QQQ, AAPL)'
    )

    # Optional arguments
    parser.add_argument(
        '--estimator',
        type=str,
        default=None,
        choices=list_estimators(),
        help='Volatility estimator type (default: from config, or yang_zhang)'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run all estimators and compare results'
    )

    parser.add_argument(
        '--window',
        type=int,
        default=None,
        help='Rolling window size in trading days (default: from config)'
    )

    parser.add_argument(
        '--lambda',
        type=float,
        default=None,
        dest='lambda_param',
        help='EWMA lambda parameter (default: 0.94, only used with EWMA estimator)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results (optional)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--events',
        action='store_true',
        help='Run event analysis'
    )

    parser.add_argument(
        '--event_window',
        type=int,
        default=None,
        help='Days before/after event for analysis (default: from config)'
    )

    parser.add_argument(
        '--predict',
        action='store_true',
        help='Generate volatility predictions for upcoming events'
    )

    parser.add_argument(
        '--excel',
        action='store_true',
        help='Generate Excel report (auto-enabled if --output_dir specified)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up logging
    log_config = config.get('logging', {})
    log_level = 'DEBUG' if args.verbose else log_config.get('level', 'INFO')
    logger = setup_logging(
        log_file=log_config.get('file'),
        log_level=log_level,
        console=log_config.get('console', True)
    )

    logger.info(f"Starting volatility estimation for {args.symbol}")

    # Get parameters (CLI overrides config)
    data_config = config.get('data', {})
    vol_config = config.get('volatility', {})

    start_date = data_config.get('start_date', '2004-01-01')
    end_date = data_config.get('end_date', '2024-12-31')
    window = args.window or vol_config.get('default_window', 60)
    annualization_factor = vol_config.get('annualization_factor', 252)
    lambda_param = args.lambda_param or vol_config.get('ewma_lambda', 0.94)
    cache_dir = data_config.get('cache_dir', './data/cache')
    cache_format = data_config.get('cache_format', 'parquet')

    # Event analysis parameters
    events_config = config.get('events', {})
    event_window = args.event_window or events_config.get('pre_window', 5)
    events_csv_path = events_config.get('csv_path', './data/events/economic_calendar.csv')

    # Prediction parameters
    predictions_config = config.get('predictions', {})
    historical_lookback = predictions_config.get('historical_lookback', 252)
    confidence_threshold = predictions_config.get('confidence_threshold', 0.8)

    # Determine estimator mode
    if args.compare:
        estimator_name = None
        logger.info("Comparison mode: running all estimators")
    else:
        estimator_name = args.estimator or vol_config.get('default_estimator', 'yang_zhang')
        logger.info(f"Single estimator mode: {estimator_name}")

    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Window size: {window} days")

    # Load market data
    try:
        logger.info(f"Loading market data for {args.symbol}...")
        df = get_market_data(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            use_cache=True,
            cache_dir=cache_dir,
            cache_format=cache_format
        )
        logger.info(f"Data loaded successfully: {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        sys.exit(1)

    # Check data quality
    quality_report = check_data_quality(df)
    logger.info(f"Data quality: {quality_report['total_rows']} rows, "
                f"missing values: {sum(v['count'] for v in quality_report['missing_values'].values())}")

    # Calculate volatility
    try:
        if args.compare:
            # Comparison mode: run all estimators
            logger.info("Running all estimators for comparison...")
            results = run_all_estimators(
                df, window=window, annualization_factor=annualization_factor, lambda_param=lambda_param
            )

            # Remove rows with all NaN
            results = results.dropna(how='all')

            logger.info(f"Comparison complete: {len(results)} estimates per estimator")

            # Calculate comparison statistics
            correlation_matrix = calculate_correlation_matrix(results)
            mse_matrix = calculate_mse_matrix(results)
            stats = generate_comparison_statistics(results)

            # Print results
            print("\n" + "="*70)
            print(f"Volatility Estimator Comparison: {args.symbol}")
            print("="*70)
            print(f"Window: {window} days")
            print(f"Date Range: {results['date'].min()} to {results['date'].max()}")
            print(f"Total Estimates: {len(results)}")

            print("\n" + "-"*70)
            print("Summary Statistics (Annualized %):")
            print("-"*70)
            print(f"{'Estimator':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            print("-"*70)
            for est_name, est_stats in stats.items():
                print(f"{est_name:<20} {est_stats['mean']:>10.2f} {est_stats['std']:>10.2f} "
                      f"{est_stats['min']:>10.2f} {est_stats['max']:>10.2f}")

            print("\n" + "-"*70)
            print("Correlation Matrix:")
            print("-"*70)
            print(correlation_matrix.round(4))

            print("\n" + "-"*70)
            print("Mean Squared Error Matrix:")
            print("-"*70)
            print(mse_matrix.round(4))

            print("="*70)

            # Save to output directory if specified
            if args.output_dir:
                output_path = Path(args.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / f"{args.symbol}_comparison_volatility.csv"
                results.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")

        else:
            # Single estimator mode
            logger.info(f"Calculating volatility using {estimator_name}...")

            # Get estimator
            try:
                if estimator_name == 'ewma':
                    estimator = get_estimator(
                        estimator_name, window, annualization_factor, lambda_param=lambda_param
                    )
                else:
                    estimator = get_estimator(estimator_name, window, annualization_factor)
            except Exception as e:
                logger.error(f"Failed to create estimator: {str(e)}")
                sys.exit(1)

            volatility = estimator.compute(df, annualize=True)

            # Create results DataFrame
            results = pd.DataFrame({
                'date': df['date'],
                'volatility': volatility
            })

            # Remove NaN values (insufficient data periods)
            results = results.dropna()

            logger.info(f"Volatility calculation complete: {len(results)} estimates")

            # Print summary statistics
            print("\n" + "="*60)
            print(f"Volatility Estimation Results: {args.symbol}")
            print("="*60)
            print(f"Estimator: {estimator_name}")
            if estimator_name == 'ewma':
                print(f"Lambda: {lambda_param}")
            print(f"Window: {window} days")
            print(f"Date Range: {results['date'].min()} to {results['date'].max()}")
            print(f"Total Estimates: {len(results)}")
            print("\nSummary Statistics (Annualized %):")
            print(f"  Mean:   {results['volatility'].mean():.2f}%")
            print(f"  Std:    {results['volatility'].std():.2f}%")
            print(f"  Min:    {results['volatility'].min():.2f}%")
            print(f"  Max:    {results['volatility'].max():.2f}%")
            print("="*60)

            # Event analysis if requested
            event_results = None
            if args.events:
                try:
                    logger.info("Loading events and running event analysis...")
                    events_df = load_events(events_csv_path)
                    logger.info(f"Loaded {len(events_df)} events")

                    # Run event analysis
                    event_results = analyze_all_events(
                        volatility_series=results['volatility'],
                        volatility_dates=results['date'],
                        events_df=events_df,
                        pre_window=event_window,
                        post_window=event_window
                    )

                    # Filter out events with insufficient data
                    event_results = event_results.dropna(subset=['pre_vol', 'post_vol'])

                    if len(event_results) > 0:
                        print("\n" + "-"*60)
                        print("Event Impact Analysis:")
                        print("-"*60)
                        print(f"Total Events Analyzed: {len(event_results)}")
                        significant_count = event_results['significant'].sum()
                        print(f"Significant Events: {significant_count} ({100*significant_count/len(event_results):.1f}%)")
                        avg_change = event_results['volatility_change_pct'].mean()
                        print(f"Average Volatility Change: {avg_change:.2f}%")
                        print("\nTop 5 Most Impactful Events:")
                        top_events = event_results.nlargest(5, 'volatility_change_pct')[
                            ['event_date', 'event_type', 'volatility_change_pct', 'significant']
                        ]
                        for _, row in top_events.iterrows():
                            sig_marker = "*" if row['significant'] else ""
                            print(f"  {row['event_date']} ({row['event_type']}): {row['volatility_change_pct']:.2f}%{sig_marker}")
                        print("-"*60)

                except Exception as e:
                    logger.warning(f"Event analysis failed: {str(e)}")

            # Save to output directory if specified
            if args.output_dir:
                output_path = Path(args.output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / f"{args.symbol}_{estimator_name}_volatility.csv"
                results.to_csv(output_file, index=False)
                logger.info(f"Results saved to {output_file}")

                if event_results is not None and len(event_results) > 0:
                    event_file = output_path / f"{args.symbol}_event_analysis.csv"
                    event_results.to_csv(event_file, index=False)
                    logger.info(f"Event analysis saved to {event_file}")

            # Event analysis for comparison mode
            if args.compare and args.events:
                try:
                    logger.info("Running event analysis for comparison mode...")
                    events_df = load_events(events_csv_path)
                    # Use first estimator column for event analysis in comparison mode
                    first_estimator = [col for col in results.columns if col != 'date'][0]
                    event_results = analyze_all_events(
                        volatility_series=results[first_estimator],
                        volatility_dates=results['date'],
                        events_df=events_df,
                        pre_window=event_window,
                        post_window=event_window
                    )
                    event_results = event_results.dropna(subset=['pre_vol', 'post_vol'])
                    if len(event_results) > 0:
                        logger.info(f"Event analysis complete: {len(event_results)} events analyzed")
                except Exception as e:
                    logger.warning(f"Event analysis failed in comparison mode: {str(e)}")
                    event_results = None

            # Predictions if requested
            predictions_df = None
            backtest_metrics = None
            if args.predict and args.events:
                try:
                    logger.info("Generating volatility predictions...")
                    events_df = load_events(events_csv_path)

                    # Build pattern database
                    pattern_db = build_pattern_database(
                        events_df, results['volatility'], results['date'],
                        pre_window=event_window, post_window=event_window
                    )

                    # Find upcoming events (future dates)
                    current_date = results['date'].max()
                    upcoming_events = events_df[pd.to_datetime(events_df['date']) > pd.to_datetime(current_date)]

                    if len(upcoming_events) > 0:
                        # Generate predictions for first upcoming event
                        upcoming_event = upcoming_events.iloc[0]
                        similar = find_similar_events(
                            upcoming_event, events_df, results['volatility'], results['date']
                        )

                        if len(similar) > 0:
                            predictions_df = predict_volatility_path(
                                parse_date(upcoming_event['date']),
                                pattern_db, results['volatility'], results['date'],
                                similar, lookback_window=10
                            )

                            logger.info(f"Generated prediction for {upcoming_event['date']}")

                    # Run backtesting
                    backtest_metrics = backtest_predictions(
                        events_df, results['volatility'], results['date'],
                        pattern_db, event_window, event_window
                    )

                    if backtest_metrics['total_predictions'] > 0:
                        logger.info(f"Backtesting complete: {backtest_metrics['total_predictions']} predictions")
                        print("\n" + "-"*60)
                        print("Prediction Accuracy (Backtesting):")
                        print("-"*60)
                        print(f"Mean Absolute Error: {backtest_metrics['prediction_mae']:.2f}%")
                        print(f"Root Mean Squared Error: {backtest_metrics['prediction_rmse']:.2f}%")
                        print(f"Correlation: {backtest_metrics['prediction_correlation']:.3f}")
                        print("-"*60)

                except Exception as e:
                    logger.warning(f"Prediction generation failed: {str(e)}")

            # Generate Excel report if requested
            if (args.excel or args.output_dir) and args.output_dir:
                try:
                    logger.info("Generating Excel report...")
                    excel_path = Path(args.output_dir) / f"{args.symbol}_volatility_report.xlsx"
                    export_to_excel(
                        volatility_data=results if not args.compare else None,
                        events_data=event_results,
                        predictions=predictions_df,
                        backtest_results=backtest_metrics['backtest_results'] if backtest_metrics else None,
                        output_path=str(excel_path)
                    )
                    logger.info(f"Excel report saved to {excel_path}")
                except Exception as e:
                    logger.warning(f"Excel export failed: {str(e)}")

            # Generate charts if output directory specified
            if args.output_dir:
                try:
                    logger.info("Generating charts...")
                    chart_dir = Path(args.output_dir) / 'charts'
                    chart_dir.mkdir(parents=True, exist_ok=True)

                    # Volatility comparison chart
                    if args.compare:
                        plot_volatility_comparison(
                            volatility_data if 'date' in volatility_data.columns else results,
                            events_df if args.events else None,
                            str(chart_dir / f"{args.symbol}_volatility_comparison.png")
                        )

                    # Prediction chart
                    if predictions_df is not None:
                        plot_predictions(
                            predicted=predictions_df,
                            events=upcoming_events if args.predict else None,
                            output_path=str(chart_dir / f"{args.symbol}_predictions.png")
                        )

                    logger.info(f"Charts saved to {chart_dir}")
                except Exception as e:
                    logger.warning(f"Chart generation failed: {str(e)}")

            # Generate summary report
            if args.output_dir:
                try:
                    summary_dir = Path(args.output_dir) / 'summaries'
                    summary_dir.mkdir(parents=True, exist_ok=True)
                    generate_summary_report(
                        volatility_df=results,
                        events_df=event_results,
                        predictions=predictions_df,
                        backtest_metrics=backtest_metrics,
                        output_path=str(summary_dir / f"{args.symbol}_summary.txt")
                    )
                    logger.info(f"Summary report saved")
                except Exception as e:
                    logger.warning(f"Summary report generation failed: {str(e)}")

    except Exception as e:
        logger.error(f"Failed to calculate volatility: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()

