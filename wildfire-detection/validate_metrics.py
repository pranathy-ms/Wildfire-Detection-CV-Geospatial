"""
Validate predictions against actual fire spread
"""

import numpy as np
import pandas as pd
import sqlite3
import config
from prediction import FirePredictor
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import json

def validate_prediction(predict_date, actual_date):
    """
    Predict on predict_date, validate against actual_date
    
    predict_date: Date to make predictions for (e.g., Jan 8)
    actual_date: Date to check actual spread (e.g., Jan 9)
    """
    print("=" * 60)
    print(f"VALIDATION: {predict_date} → {actual_date}")
    print("=" * 60)
    
    # Make predictions
    predictor = FirePredictor()
    predictions = predictor.predict(predict_date)
    
    if not predictions:
        print("No predictions generated")
        return None
    
    # Load actual fires for next day
    conn = sqlite3.connect(config.DB_PATH)
    query = """
        SELECT lat, lon
        FROM fires
        WHERE date = ?
    """
    actual_fires = pd.read_sql_query(query, conn, params=(actual_date,))
    conn.close()
    
    print(f"\n Ground Truth:")
    print(f"  Fires on {actual_date}: {len(actual_fires)}")
    
    # Convert actual fires to grid cells
    grid_step = config.GRID_SIZE_KM * 0.01
    actual_fire_cells = set()
    
    for _, fire in actual_fires.iterrows():
        lat_idx = int((fire['lat'] - config.LAT_MIN) / grid_step)
        lon_idx = int((fire['lon'] - config.LON_MIN) / grid_step)
        
        # Convert back to lat/lon for matching
        cell_lat = config.LAT_MIN + lat_idx * grid_step
        cell_lon = config.LON_MIN + lon_idx * grid_step
        actual_fire_cells.add((round(cell_lat, 2), round(cell_lon, 2)))
    
    print(f"  Grid cells burning: {len(actual_fire_cells)}")
    
    # Match predictions to actual
    y_true = []
    y_pred = []
    y_prob = []
    
    for pred in predictions:
        pred_lat = round(pred['lat'], 2)
        pred_lon = round(pred['lon'], 2)
        
        # Did this cell actually burn?
        actually_burned = (pred_lat, pred_lon) in actual_fire_cells
        
        y_true.append(1 if actually_burned else 0)
        y_pred.append(1 if pred['spread_probability'] > 0.5 else 0)
        y_prob.append(pred['spread_probability'])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Calculate metrics
    print(f"\n Validation Metrics:")
    print(f"  Cells predicted: {len(predictions)}")
    print(f"  Actually burned: {y_true.sum()}")
    print(f"  Predicted to burn (>50%): {y_pred.sum()}")
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    print(f"\n  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        print(f"  AUC-ROC:   {auc:.3f}")
    
    # Confusion matrix
    #cm = confusion_matrix(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n Confusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]:4d} (correctly predicted no spread)")
    print(f"  False Positives: {cm[0,1]:4d} (false alarms)")
    print(f"  False Negatives: {cm[1,0]:4d} (missed fires) ⚠️")
    print(f"  True Positives:  {cm[1,1]:4d} (caught fires)")
    
    # Risk level breakdown
    print(f"\n Prediction Accuracy by Risk Level:")
    
    for threshold, label in [(0.7, "HIGH"), (0.5, "MEDIUM"), (0.3, "LOW")]:
        high_risk_mask = y_prob > threshold
        if high_risk_mask.sum() > 0:
            accuracy = y_true[high_risk_mask].mean()
            count = high_risk_mask.sum()
            actual_burned = y_true[high_risk_mask].sum()
            print(f"  {label:6s} (>{threshold:.0%}): {actual_burned}/{count} burned ({accuracy:.1%} accurate)")
    
    # Sample correct and incorrect predictions
    print(f"\n Sample CORRECT predictions (predicted burn + actually burned):")
    correct_burns = [(predictions[i], y_prob[i]) for i in range(len(predictions)) 
                     if y_true[i] == 1 and y_pred[i] == 1]
    for pred, prob in correct_burns[:3]:
        print(f"  ({pred['lat']:.4f}, {pred['lon']:.4f}): {prob:.1%} - BURNED ✓")
    
    print(f"\n Sample MISSED predictions (predicted no burn but actually burned):")
    missed = [(predictions[i], y_prob[i]) for i in range(len(predictions)) 
              if y_true[i] == 1 and y_pred[i] == 0]
    for pred, prob in missed[:3]:
        print(f"  ({pred['lat']:.4f}, {pred['lon']:.4f}): {prob:.1%} - BURNED but missed ⚠️")
    
    print(f"\n  Sample FALSE ALARMS (predicted burn but didn't burn):")
    false_alarms = [(predictions[i], y_prob[i]) for i in range(len(predictions)) 
                    if y_true[i] == 0 and y_pred[i] == 1]
    for pred, prob in false_alarms[:3]:
        print(f"  ({pred['lat']:.4f}, {pred['lon']:.4f}): {prob:.1%} - False alarm")
    
    print("=" * 60)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'actual_burned': int(y_true.sum()),
        'predicted_burned': int(y_pred.sum()),
        'total_predictions': len(predictions)
    }

def validate_multiple_days():
    """Validate across multiple consecutive days"""
    print("\n" + "=" * 60)
    print("MULTI-DAY VALIDATION")
    print("=" * 60)
    
    # Test pairs: (predict_date, actual_date)
    test_pairs = [
        ("2025-01-07", "2025-01-08"),
        ("2025-01-08", "2025-01-09"),
        ("2025-01-09", "2025-01-10"),
        ("2025-01-10", "2025-01-11"),
        ("2025-01-11", "2025-01-12"),
    ]
    
    results = []
    
    for predict_date, actual_date in test_pairs:
        result = validate_prediction(predict_date, actual_date)
        if result:
            results.append({
                'predict_date': predict_date,
                'actual_date': actual_date,
                **result
            })
        print("\n")
    
    # Summary
    if results:
        print("=" * 60)
        print("SUMMARY ACROSS ALL DAYS")
        print("=" * 60)
        
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        
        print(f"\nAverage Metrics:")
        print(f"  Precision: {avg_precision:.3f}")
        print(f"  Recall:    {avg_recall:.3f}")
        print(f"  F1 Score:  {avg_f1:.3f}")
        
        print(f"\nDay-by-Day Performance:")
        print(f"{'Date':12s} | {'Actual':6s} | {'Predicted':9s} | {'F1':4s}")
        print("-" * 50)
        for r in results:
            print(f"{r['predict_date']} | {r['actual_burned']:6d} | {r['predicted_burned']:9d} | {r['f1']:.3f}")
        
        # Save results
        output_file = config.DATA_DIR / "validation_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'avg_precision': float(avg_precision),
                    'avg_recall': float(avg_recall),
                    'avg_f1': float(avg_f1)
                },
                'daily_results': results
            }, f, indent=2)
        
        print(f"\nResults saved: data/validation_results.json")
        print("=" * 60)

        create_validation_visualizations(results)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_validation_visualizations(results):
    """Create summary visualizations"""
    print("\n📊 Generating visualizations...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Extract data
    dates = [r['predict_date'] for r in results]
    f1_scores = [r['f1'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    actual_burned = [r['actual_burned'] for r in results]
    predicted_burned = [r['predicted_burned'] for r in results]
    
    # 1. F1 Score over time
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(dates, f1_scores, marker='o', linewidth=2, markersize=8, color='#ff6b35')
    ax1.set_title('F1 Score Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('F1 Score', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # 2. Precision vs Recall
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(dates, precisions, marker='s', label='Precision', linewidth=2, markersize=8, color='#4CAF50')
    ax2.plot(dates, recalls, marker='^', label='Recall', linewidth=2, markersize=8, color='#2196F3')
    ax2.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Score', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # 3. Actual vs Predicted Burns
    ax3 = plt.subplot(2, 3, 3)
    x = range(len(dates))
    width = 0.35
    ax3.bar([i - width/2 for i in x], actual_burned, width, label='Actual', color='#FF5722', alpha=0.8)
    ax3.bar([i + width/2 for i in x], predicted_burned, width, label='Predicted', color='#FFC107', alpha=0.8)
    ax3.set_title('Actual vs Predicted Fire Spreads', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_ylabel('Number of Cells', fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(dates, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Performance Summary Table
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    avg_f1 = np.mean([r['f1'] for r in results if r['f1'] > 0])
    avg_precision = np.mean([r['precision'] for r in results if r['precision'] > 0])
    avg_recall = np.mean([r['recall'] for r in results if r['recall'] > 0])
    
    summary_text = f"""
    VALIDATION SUMMARY
    {'='*40}
    
    Average F1 Score:      {avg_f1:.3f}
    Average Precision:     {avg_precision:.3f}
    Average Recall:        {avg_recall:.3f}
    
    Total Predictions:     {sum(r['total_predictions'] for r in results)}
    Total Actual Burns:    {sum(r['actual_burned'] for r in results)}
    Total Predicted Burns: {sum(r['predicted_burned'] for r in results)}
    
    Best F1 Day:  {dates[np.argmax(f1_scores)]} ({max(f1_scores):.3f})
    Worst F1 Day: {dates[np.argmin(f1_scores)]} ({min(f1_scores):.3f})
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 5. Daily Performance Heatmap
    ax5 = plt.subplot(2, 3, 5)
    metrics_matrix = np.array([precisions, recalls, f1_scores])
    im = ax5.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax5.set_xticks(range(len(dates)))
    ax5.set_xticklabels(dates, rotation=45, ha='right')
    ax5.set_yticks([0, 1, 2])
    ax5.set_yticklabels(['Precision', 'Recall', 'F1'])
    ax5.set_title('Daily Metrics Heatmap', fontsize=14, fontweight='bold')
    
    # Add values to heatmap
    for i in range(3):
        for j in range(len(dates)):
            text = ax5.text(j, i, f'{metrics_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax5, label='Score')
    
    # 6. Model Confidence Distribution
    ax6 = plt.subplot(2, 3, 6)
    
    # Create risk level summary
    risk_levels = ['Excellent\n(F1 > 0.8)', 'Good\n(F1 > 0.6)', 'Fair\n(F1 > 0.4)', 'Poor\n(F1 < 0.4)']
    counts = [
        sum(1 for f1 in f1_scores if f1 > 0.8),
        sum(1 for f1 in f1_scores if 0.6 < f1 <= 0.8),
        sum(1 for f1 in f1_scores if 0.4 < f1 <= 0.6),
        sum(1 for f1 in f1_scores if f1 <= 0.4)
    ]
    colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF5722']
    
    ax6.bar(risk_levels, counts, color=colors, alpha=0.8)
    ax6.set_title('Performance Distribution', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Number of Days', fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_file = config.DATA_DIR / "validation_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Visualization saved: data/validation_summary.png")
    
    plt.show()
    
    return output_file


if __name__ == "__main__":
    validate_multiple_days()