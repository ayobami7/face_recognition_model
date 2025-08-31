import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your training log
df = pd.read_csv('utkface_finetuning_log.csv')

# Create comprehensive plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('UTKFace Model Training Progress', fontsize=16, fontweight='bold')

# 1. Age Prediction Accuracy
axes[0, 0].plot(df['epoch'], df['age_output_sparse_categorical_accuracy'], 
                'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
axes[0, 0].plot(df['epoch'], df['val_age_output_sparse_categorical_accuracy'], 
                'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
axes[0, 0].set_title('Age Prediction Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0.4, 0.9])

# 2. Gender Prediction Accuracy
axes[0, 1].plot(df['epoch'], df['gender_output_sparse_categorical_accuracy'], 
                'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
axes[0, 1].plot(df['epoch'], df['val_gender_output_sparse_categorical_accuracy'], 
                'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
axes[0, 1].set_title('Gender Prediction Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0.8, 1.0])

# 3. Race Prediction Accuracy
axes[1, 0].plot(df['epoch'], df['race_output_sparse_categorical_accuracy'], 
                'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
axes[1, 0].plot(df['epoch'], df['val_race_output_sparse_categorical_accuracy'], 
                'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
axes[1, 0].set_title('Race Prediction Accuracy', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0.6, 1.0])

# 4. Overall Loss
axes[1, 1].plot(df['epoch'], df['loss'], 
                'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
axes[1, 1].plot(df['epoch'], df['val_loss'], 
                'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
axes[1, 1].set_title('Overall Model Loss', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final performance summary
print("="*60)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("="*60)

final_epoch = df.iloc[-1]
print(f"Epoch: {int(final_epoch['epoch'])}")
print(f"Learning Rate: {final_epoch['learning_rate']:.2e}")
print()

print("TRAINING ACCURACIES:")
print(f"  Age:    {final_epoch['age_output_sparse_categorical_accuracy']:.4f} ({final_epoch['age_output_sparse_categorical_accuracy']*100:.2f}%)")
print(f"  Gender: {final_epoch['gender_output_sparse_categorical_accuracy']:.4f} ({final_epoch['gender_output_sparse_categorical_accuracy']*100:.2f}%)")
print(f"  Race:   {final_epoch['race_output_sparse_categorical_accuracy']:.4f} ({final_epoch['race_output_sparse_categorical_accuracy']*100:.2f}%)")
print()

print("VALIDATION ACCURACIES:")
print(f"  Age:    {final_epoch['val_age_output_sparse_categorical_accuracy']:.4f} ({final_epoch['val_age_output_sparse_categorical_accuracy']*100:.2f}%)")
print(f"  Gender: {final_epoch['val_gender_output_sparse_categorical_accuracy']:.4f} ({final_epoch['val_gender_output_sparse_categorical_accuracy']*100:.2f}%)")
print(f"  Race:   {final_epoch['val_race_output_sparse_categorical_accuracy']:.4f} ({final_epoch['val_race_output_sparse_categorical_accuracy']*100:.2f}%)")
print()

print("LOSSES:")
print(f"  Training Loss:   {final_epoch['loss']:.4f}")
print(f"  Validation Loss: {final_epoch['val_loss']:.4f}")

# Calculate improvement from first to last epoch
first_epoch = df.iloc[0]
print()
print("IMPROVEMENT FROM EPOCH 0 TO FINAL EPOCH:")
print(f"  Age Accuracy:    {(final_epoch['val_age_output_sparse_categorical_accuracy'] - first_epoch['val_age_output_sparse_categorical_accuracy'])*100:.2f}% points")
print(f"  Gender Accuracy: {(final_epoch['val_gender_output_sparse_categorical_accuracy'] - first_epoch['val_gender_output_sparse_categorical_accuracy'])*100:.2f}% points")
print(f"  Race Accuracy:   {(final_epoch['val_race_output_sparse_categorical_accuracy'] - first_epoch['val_race_output_sparse_categorical_accuracy'])*100:.2f}% points")

# Check for overfitting
print()
print("OVERFITTING ANALYSIS:")
for task in ['age', 'gender', 'race']:
    train_acc = final_epoch[f'{task}_output_sparse_categorical_accuracy']
    val_acc = final_epoch[f'val_{task}_output_sparse_categorical_accuracy']
    gap = (train_acc - val_acc) * 100
    
    if gap > 5:
        status = "⚠️  Potential overfitting"
    elif gap > 2:
        status = "⚡ Slight overfitting"
    else:
        status = "✅ Good generalization"
    
    print(f"  {task.capitalize():6}: Train-Val gap = {gap:.2f}% - {status}")