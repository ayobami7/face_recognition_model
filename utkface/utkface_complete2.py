import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, metrics, Model
from keras_facenet import FaceNet
import pandas as pd
import numpy as np
import os
from pathlib import Path
import zipfile
import gdown
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class UTKFaceDataProcessor:
    def __init__(self, img_dir, img_size=160, sample_size=None, is_validation=False):
        self.img_size = img_size
        self.img_dir = img_dir
        self.is_validation = is_validation
        
        # Parse UTKFace filenames to extract labels
        self.parse_utkface_filenames()
        
        # Filter dataset to keep only valid entries
        original_len = len(self.df)
        # Remove entries with invalid ages (keep ages 0-116)
        self.df = self.df[(self.df['age'] >= 0) & (self.df['age'] <= 116)].reset_index(drop=True)
        print(f"Removed {original_len - len(self.df)} rows with invalid ages")
        
        # Sample data if specified
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print(f"Sampled {'validation' if is_validation else 'training'}: {len(self.df)} samples")
        
        # Create age bins for classification (similar to FairFace)
        self.create_age_bins()
        
        # Initialize label encoders
        if not hasattr(self, 'age_encoder'):
            self.age_encoder = LabelEncoder()
            self.gender_encoder = LabelEncoder()
            self.race_encoder = LabelEncoder()
            
            # Fit encoders
            self.age_encoder.fit(self.df['age_bin'])
            self.gender_encoder.fit(self.df['gender'])
            self.race_encoder.fit(self.df['race'])
        
        # Encode labels
        self.df['age_encoded'] = self.age_encoder.transform(self.df['age_bin'])
        self.df['gender_encoded'] = self.gender_encoder.transform(self.df['gender'])
        self.df['race_encoded'] = self.race_encoder.transform(self.df['race'])
        
        self.num_classes = {
            'age': len(self.age_encoder.classes_),
            'gender': len(self.gender_encoder.classes_),
            'race': len(self.race_encoder.classes_)
        }
        
        print(f"Classes - Age: {self.num_classes['age']}, Gender: {self.num_classes['gender']}, Race: {self.num_classes['race']}")
        if not is_validation:
            print(f"Age range: {min(self.df['age'])}-{max(self.df['age'])}")
            print(f"Age bins: {list(self.age_encoder.classes_)}")
            print(f"Gender groups: {list(self.gender_encoder.classes_)}")
            print(f"Race groups: {list(self.race_encoder.classes_)}")
    
    def create_age_bins(self):
        """Create age bins similar to FairFace format"""
        def age_to_bin(age):
            if age <= 2:
                return '0-2'
            elif age <= 9:
                return '3-9'
            elif age <= 19:
                return '10-19'
            elif age <= 29:
                return '20-29'
            elif age <= 39:
                return '30-39'
            elif age <= 49:
                return '40-49'
            elif age <= 59:
                return '50-59'
            elif age <= 69:
                return '60-69'
            else:
                return '70+'
        
        self.df['age_bin'] = self.df['age'].apply(age_to_bin)
        print(f"Age bin distribution:\n{self.df['age_bin'].value_counts().sort_index()}")
    
    def parse_utkface_filenames(self):
        """Parse UTKFace filenames to extract age, gender, race labels"""
        data = []
        
        for filename in os.listdir(self.img_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                parts = filename.split('_')
                if len(parts) >= 3:
                    try:
                        age = int(parts[0])
                        gender_code = parts[1]  # 0: female, 1: male
                        race_code = parts[2]    # 0: White, 1: Black, 2: Asian, 3: Indian, 4: Others
                        
                        # Map gender codes to labels
                        gender_mapping = {'0': 'Female', '1': 'Male'}
                        gender_label = gender_mapping.get(gender_code, 'Unknown')
                        
                        # Map race codes to labels
                        race_mapping = {
                            '0': 'White', 
                            '1': 'Black', 
                            '2': 'East Asian', 
                            '3': 'Indian', 
                            '4': 'Others'
                        }
                        race_label = race_mapping.get(race_code, 'Unknown')
                        
                        # Skip unknown labels
                        if gender_label != 'Unknown' and race_label != 'Unknown':
                            data.append({
                                'file': filename,
                                'age': age,
                                'gender': gender_label,
                                'race': race_label
                            })
                    except (ValueError, IndexError):
                        print(f"Skipping invalid filename: {filename}")
                        continue
        
        self.df = pd.DataFrame(data)
        print(f"Loaded {len(self.df)} valid images from UTKFace dataset")
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Age distribution: min={self.df['age'].min()}, max={self.df['age'].max()}, mean={self.df['age'].mean():.1f}")
        print(f"Gender distribution:\n{self.df['gender'].value_counts()}")
        print(f"Race distribution:\n{self.df['race'].value_counts()}")
    
    def filter_missing_files(self):
        """Filter out rows where image files don't exist"""
        print(f"Original dataset size: {len(self.df)}")
        
        existing_files = []
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.img_dir, row['file'])
            if os.path.exists(img_path):
                existing_files.append(idx)
        
        self.df = self.df.loc[existing_files].reset_index(drop=True)
        print(f"Filtered dataset size: {len(self.df)}")
        
        return self
    
    def load_and_preprocess_image(self, image_path, augment=False):
        """Load and preprocess single image"""
        try:
            if not tf.io.gfile.exists(image_path):
                print(f"Warning: File not found: {image_path}")
                return tf.zeros([self.img_size, self.img_size, 3], dtype=tf.float32)
            
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, [self.img_size, self.img_size])
            image = tf.cast(image, tf.float32) / 255.0
            
            if augment:
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_brightness(image, 0.1)
                image = tf.image.random_contrast(image, 0.9, 1.1)
                image = tf.image.random_saturation(image, 0.9, 1.1)
                image = tf.image.random_hue(image, 0.05)
            
            image = tf.image.per_image_standardization(image)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return tf.zeros([self.img_size, self.img_size, 3], dtype=tf.float32)
    
    def create_dataset(self, batch_size=32, augment=False, shuffle=True):
        """Create TensorFlow dataset"""
        def generator():
            indices = np.arange(len(self.df))
            if shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                row = self.df.iloc[idx]
                img_path = os.path.join(self.img_dir, row['file'])
                
                image = self.load_and_preprocess_image(img_path, augment)
                
                yield (
                    image,
                    {
                        'age_output': tf.cast(row['age_encoded'], dtype=tf.int32),
                        'gender_output': tf.cast(row['gender_encoded'], dtype=tf.int32),
                        'race_output': tf.cast(row['race_encoded'], dtype=tf.int32)
                    }
                )
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.img_size, self.img_size, 3), dtype=tf.float32),
                {
                    'age_output': tf.TensorSpec(shape=(), dtype=tf.int32),
                    'gender_output': tf.TensorSpec(shape=(), dtype=tf.int32),
                    'race_output': tf.TensorSpec(shape=(), dtype=tf.int32)
                }
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

class FaceNetMultiTask(tf.keras.Model):
    def __init__(self, num_age_classes, num_gender_classes, num_race_classes,
                 freeze_backbone=False):
        super(FaceNetMultiTask, self).__init__()

        self.backbone = FaceNet().model

        # Freeze backbone
        if freeze_backbone:
            self.backbone.trainable = False

        # Shared dense layer
        self.shared_dense = layers.Dense(512, activation='relu', name='shared_dense')
        self.shared_dropout = layers.Dropout(0.7, name='shared_dropout')

        # Task-specific classification heads
        # age branch
        self.age_classifier = tf.keras.Sequential([
            layers.Dense(128, activation='relu', name='age_dense'),
            layers.Dense(num_age_classes, activation='softmax', name='age_output')
        ], name='age_head')

        # gender branch
        self.gender_classifier = tf.keras.Sequential([
            layers.Dense(num_gender_classes, activation='softmax', name='gender_output')
        ], name='gender_head')

        # race branch
        self.race_classifier = tf.keras.Sequential([
            layers.Dense(num_race_classes, activation='softmax', name='race_output')
        ], name='race_head')

    def call(self, inputs, training=None):
        # Get FaceNet embeddings
        embeddings = self.backbone(inputs, training=training)

        # Shared processing
        shared_features = self.shared_dense(embeddings, training=training)
        shared_features = self.shared_dropout(shared_features, training=training)

        # Task-specific predictions
        age_pred = self.age_classifier(shared_features, training=training)
        gender_pred = self.gender_classifier(shared_features, training=training)
        race_pred = self.race_classifier(shared_features, training=training)

        return {
            'age_output': age_pred,
            'gender_output': gender_pred,
            'race_output': race_pred
        }

def create_and_compile_model(num_age_classes, num_gender_classes, num_race_classes, freeze_backbone=False):
    """Create and compile the multi-task model"""

    model = FaceNetMultiTask(
        num_age_classes=num_age_classes,
        num_gender_classes=num_gender_classes,
        num_race_classes=num_race_classes,
        freeze_backbone=freeze_backbone
    )

    # Build the model
    model.build((None, 160, 160, 3))

    # Optimizer with learning rate schedule
    initial_lr = 0.001

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss={
            'age_output': losses.SparseCategoricalCrossentropy(),
            'gender_output': losses.SparseCategoricalCrossentropy(),
            'race_output': losses.SparseCategoricalCrossentropy()
        },
        loss_weights={
            'age_output': 1.0,
            'gender_output': 1.0,
            'race_output': 1.0
        },
        metrics={
            'age_output': ['sparse_categorical_accuracy'],
            'gender_output': ['sparse_categorical_accuracy'],
            'race_output': ['sparse_categorical_accuracy']
        }
    )

    print(f"\nModel compiled successfully!")
    print(f"   - Backbone frozen: {freeze_backbone}")
    print(f"   - Learning rate: {initial_lr}")

    return model

# Usage example with UTKFace dataset
def train_utkface_model():
    # Set your UTKFace dataset path
    DATASET_PATH = "path/to/your/UTKFace_dataset"
    
    BATCH_SIZE = 32
    IMG_SIZE = 160
    EPOCHS = 50
    SAMPLE_SIZE = None  
    
    print("Creating training data processor...")
    # Split the dataset manually since UTKFace doesn't come pre-split
    all_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(('.jpg', '.jpeg', '.png'))]
    train_files = all_files[:int(0.8 * len(all_files))]
    val_files = all_files[int(0.8 * len(all_files)):]
    
    # Create temporary directories for train/val split
    import shutil
    train_dir = os.path.join(DATASET_PATH, 'train_temp')
    val_dir = os.path.join(DATASET_PATH, 'val_temp')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Copy files to train/val directories (or create symlinks)
    for f in train_files:
        src = os.path.join(DATASET_PATH, f)
        dst = os.path.join(train_dir, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    for f in val_files:
        src = os.path.join(DATASET_PATH, f)
        dst = os.path.join(val_dir, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    train_processor = UTKFaceDataProcessor(
        train_dir,
        img_size=IMG_SIZE,
        sample_size=SAMPLE_SIZE,
        is_validation=False
    ).filter_missing_files()
    
    print("\nCreating validation data processor...")
    val_processor = UTKFaceDataProcessor(
        val_dir,
        img_size=IMG_SIZE,
        sample_size=SAMPLE_SIZE//5 if SAMPLE_SIZE else None,
        is_validation=True
    ).filter_missing_files()
    
    # Copy encoders from training to validation processor
    val_processor.age_encoder = train_processor.age_encoder
    val_processor.gender_encoder = train_processor.gender_encoder
    val_processor.race_encoder = train_processor.race_encoder
    val_processor.num_classes = train_processor.num_classes
    
    # Re-encode validation labels with training encoders
    val_processor.df['age_encoded'] = val_processor.age_encoder.transform(val_processor.df['age_bin'])
    val_processor.df['gender_encoded'] = val_processor.gender_encoder.transform(val_processor.df['gender'])
    val_processor.df['race_encoded'] = val_processor.race_encoder.transform(val_processor.df['race'])
    
    print("\nCreating datasets...")
    train_dataset = train_processor.create_dataset(
        batch_size=BATCH_SIZE,
        augment=True,
        shuffle=True
    )
    
    val_dataset = val_processor.create_dataset(
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=False
    )
    
    print("Building model...")
    model = create_and_compile_model(
        num_age_classes=train_processor.num_classes['age'],
        num_gender_classes=train_processor.num_classes['gender'],
        num_race_classes=train_processor.num_classes['race'],
        freeze_backbone=True 
    )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_utkface_facenet_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('utkface_training_log.csv')
    ]
    
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Clean up temporary directories
    shutil.rmtree(train_dir)
    shutil.rmtree(val_dir)
    
    return model, history, train_processor, val_processor

# Prediction function for UTKFace
def predict_sample_utkface(model, img_path, train_processor, img_size=(160, 160)):
    """
    Predict demographics for a single image using UTKFace model
    """
    # Load and preprocess image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.per_image_standardization(img)
    img = tf.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img, verbose=0)

    # Extract predictions
    age_pred = np.argmax(preds['age_output'][0])
    gender_pred = np.argmax(preds['gender_output'][0])
    race_pred = np.argmax(preds['race_output'][0])

    # Decode labels
    age_label = train_processor.age_encoder.inverse_transform([age_pred])[0]
    gender_label = train_processor.gender_encoder.inverse_transform([gender_pred])[0]
    race_label = train_processor.race_encoder.inverse_transform([race_pred])[0]

    # Display
    display_img = tf.io.read_file(img_path)
    display_img = tf.image.decode_image(display_img, channels=3)
    display_img = tf.image.resize(display_img, img_size)
    display_img = tf.cast(display_img, tf.float32) / 255.0

    plt.figure(figsize=(8, 6))
    plt.imshow(display_img)
    plt.title(f"Predictions:\nAge: {age_label}\nGender: {gender_label}\nRace: {race_label}")
    plt.axis('off')
    plt.show()

    return age_label, gender_label, race_label

def plot_training_history(history):
    """Plot training history for all tasks"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss plots
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Age accuracy
    axes[0, 1].plot(history.history['age_output_sparse_categorical_accuracy'], label='Train Acc')
    axes[0, 1].plot(history.history['val_age_output_sparse_categorical_accuracy'], label='Val Acc')
    axes[0, 1].set_title('Age Classification Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Gender accuracy
    axes[0, 2].plot(history.history['gender_output_sparse_categorical_accuracy'], label='Train Acc')
    axes[0, 2].plot(history.history['val_gender_output_sparse_categorical_accuracy'], label='Val Acc')
    axes[0, 2].set_title('Gender Classification Accuracy')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Race accuracy
    axes[1, 0].plot(history.history['race_output_sparse_categorical_accuracy'], label='Train Acc')
    axes[1, 0].plot(history.history['val_race_output_sparse_categorical_accuracy'], label='Val Acc')
    axes[1, 0].set_title('Race Classification Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Individual loss plots
    axes[1, 1].plot(history.history['age_output_loss'], label='Age Loss')
    axes[1, 1].plot(history.history['gender_output_loss'], label='Gender Loss')
    axes[1, 1].plot(history.history['race_output_loss'], label='Race Loss')
    axes[1, 1].set_title('Task-specific Losses (Train)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Learning rate
    if 'lr' in history.history:
        axes[1, 2].plot(history.history['lr'])
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('LR')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True)
    else:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, val_dataset, val_processor):
    """Evaluate model performance and create confusion matrices"""
    print("Evaluating model...")
    
    # Get predictions
    y_true_age = []
    y_true_gender = []
    y_true_race = []
    y_pred_age = []
    y_pred_gender = []
    y_pred_race = []
    
    for batch_x, batch_y in val_dataset:
        preds = model.predict(batch_x, verbose=0)
        
        y_true_age.extend(batch_y['age_output'].numpy())
        y_true_gender.extend(batch_y['gender_output'].numpy())
        y_true_race.extend(batch_y['race_output'].numpy())
        
        y_pred_age.extend(np.argmax(preds['age_output'], axis=1))
        y_pred_gender.extend(np.argmax(preds['gender_output'], axis=1))
        y_pred_race.extend(np.argmax(preds['race_output'], axis=1))
    
    # Calculate accuracies
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    age_acc = accuracy_score(y_true_age, y_pred_age)
    gender_acc = accuracy_score(y_true_gender, y_pred_gender)
    race_acc = accuracy_score(y_true_race, y_pred_race)
    
    print(f"\nValidation Accuracies:")
    print(f"Age: {age_acc:.4f}")
    print(f"Gender: {gender_acc:.4f}")
    print(f"Race: {race_acc:.4f}")
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Age confusion matrix
    age_cm = confusion_matrix(y_true_age, y_pred_age)
    sns.heatmap(age_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=val_processor.age_encoder.classes_,
                yticklabels=val_processor.age_encoder.classes_)
    axes[0].set_title(f'Age Confusion Matrix (Acc: {age_acc:.3f})')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Gender confusion matrix
    gender_cm = confusion_matrix(y_true_gender, y_pred_gender)
    sns.heatmap(gender_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=val_processor.gender_encoder.classes_,
                yticklabels=val_processor.gender_encoder.classes_)
    axes[1].set_title(f'Gender Confusion Matrix (Acc: {gender_acc:.3f})')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    # Race confusion matrix
    race_cm = confusion_matrix(y_true_race, y_pred_race)
    sns.heatmap(race_cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=val_processor.race_encoder.classes_,
                yticklabels=val_processor.race_encoder.classes_)
    axes[2].set_title(f'Race Confusion Matrix (Acc: {race_acc:.3f})')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed classification reports
    print("\nDetailed Classification Reports:")
    print("\n=== AGE CLASSIFICATION ===")
    print(classification_report(y_true_age, y_pred_age, 
                              target_names=val_processor.age_encoder.classes_))
    
    print("\n=== GENDER CLASSIFICATION ===")
    print(classification_report(y_true_gender, y_pred_gender, 
                              target_names=val_processor.gender_encoder.classes_))
    
    print("\n=== RACE CLASSIFICATION ===")
    print(classification_report(y_true_race, y_pred_race, 
                              target_names=val_processor.race_encoder.classes_))

def save_model_and_processors(model, train_processor, model_name="utkface_facenet_model"):
    """Save trained model and label encoders"""
    # Save the complete model
    model.save(f"{model_name}.h5")
    print(f"Model saved as {model_name}.h5")
    
    # Save label encoders
    import pickle
    encoders = {
        'age_encoder': train_processor.age_encoder,
        'gender_encoder': train_processor.gender_encoder,
        'race_encoder': train_processor.race_encoder,
        'num_classes': train_processor.num_classes
    }
    
    with open(f"{model_name}_encoders.pkl", 'wb') as f:
        pickle.dump(encoders, f)
    print(f"Encoders saved as {model_name}_encoders.pkl")

def load_model_and_processors(model_path, encoders_path):
    """Load trained model and label encoders"""
    # Load model
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    # Load encoders
    import pickle
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    print("Encoders loaded successfully")
    return model, encoders

def predict_batch_images(model, image_paths, train_processor):
    """Predict demographics for multiple images"""
    predictions = []
    
    for img_path in image_paths:
        if os.path.exists(img_path):
            age_label, gender_label, race_label = predict_sample_utkface(
                model, img_path, train_processor
            )
            predictions.append({
                'image_path': img_path,
                'predicted_age': age_label,
                'predicted_gender': gender_label,
                'predicted_race': race_label
            })
        else:
            print(f"Image not found: {img_path}")
    
    return pd.DataFrame(predictions)

# Complete main execution
if __name__ == "__main__":
    DATASET_PATH = "C:/Users/DELL/Downloads/datasets/UTKface_inthewild/part1"
    
    BATCH_SIZE = 32
    IMG_SIZE = 160
    EPOCHS = 20
    SAMPLE_SIZE = 20000
    
    print("="*60)
    print("UTKFace Multi-task Demographic Classification Training")
    print("="*60)
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path does not exist: {DATASET_PATH}")
        print("Please update DATASET_PATH to point to your UTKFace dataset directory")
        exit(1)
    
    print("Creating training data processor...")
    train_processor = UTKFaceDataProcessor(
        DATASET_PATH,
        img_size=IMG_SIZE,
        sample_size=SAMPLE_SIZE,
        is_validation=False
    ).filter_missing_files()
    
    # Split data for validation using sklearn
    train_df, val_df = train_test_split(
        train_processor.df, 
        test_size=0.2, 
        random_state=42,
        stratify=train_processor.df['age_bin']  # Stratify by age bin for balanced split
    )
    
    # Create validation processor
    print("\nCreating validation data processor...")
    val_processor = UTKFaceDataProcessor(
        DATASET_PATH,
        img_size=IMG_SIZE,
        sample_size=None,
        is_validation=True
    )
    
    # Use validation split
    val_processor.df = val_df.reset_index(drop=True)
    
    # Copy encoders from training to validation processor
    val_processor.age_encoder = train_processor.age_encoder
    val_processor.gender_encoder = train_processor.gender_encoder
    val_processor.race_encoder = train_processor.race_encoder
    val_processor.num_classes = train_processor.num_classes
    
    # Re-encode validation labels with training encoders
    val_processor.df['age_encoded'] = val_processor.age_encoder.transform(val_processor.df['age_bin'])
    val_processor.df['gender_encoded'] = val_processor.gender_encoder.transform(val_processor.df['gender'])
    val_processor.df['race_encoded'] = val_processor.race_encoder.transform(val_processor.df['race'])
    
    # Update training processor to use only training split
    train_processor.df = train_df.reset_index(drop=True)
    
    print(f"\nDataset split:")
    print(f"Training samples: {len(train_processor.df)}")
    print(f"Validation samples: {len(val_processor.df)}")
    
    print("\nCreating datasets...")
    train_dataset = train_processor.create_dataset(
        batch_size=BATCH_SIZE,
        augment=True,
        shuffle=True
    )
    
    val_dataset = val_processor.create_dataset(
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=False
    )
    
    print("\nBuilding and compiling model...")
    model = create_and_compile_model(
        num_age_classes=train_processor.num_classes['age'],
        num_gender_classes=train_processor.num_classes['gender'],
        num_race_classes=train_processor.num_classes['race'],
        freeze_backbone=True  # Start with frozen backbone for transfer learning
    )
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_utkface_facenet_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('utkface_training_log.csv')
    ]
    
    print("\n" + "="*60)
    print("Starting Training Phase 1: Frozen Backbone")
    print("="*60)
    
    # Phase 1: Train with frozen backbone
    history_phase1 = model.fit(
        train_dataset,
        epochs=EPOCHS//2,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("Starting Training Phase 2: Fine-tuning")
    print("="*60)
    
    # Phase 2: Unfreeze backbone and fine-tune with lower learning rate
    model.backbone.trainable = True
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
        loss={
            'age_output': losses.SparseCategoricalCrossentropy(),
            'gender_output': losses.SparseCategoricalCrossentropy(),
            'race_output': losses.SparseCategoricalCrossentropy()
        },
        loss_weights={
            'age_output': 1.0,
            'gender_output': 1.0,
            'race_output': 1.0
        },
        metrics={
            'age_output': ['sparse_categorical_accuracy'],
            'gender_output': ['sparse_categorical_accuracy'],
            'race_output': ['sparse_categorical_accuracy']
        }
    )
    
    # Update callbacks for phase 2
    callbacks[0] = tf.keras.callbacks.ModelCheckpoint(
        'best_utkface_facenet_finetuned_model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks[3] = tf.keras.callbacks.CSVLogger('utkface_finetuning_log.csv')
    
    # Continue training with unfrozen backbone
    history_phase2 = model.fit(
        train_dataset,
        epochs=EPOCHS//2,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine training histories
    combined_history = {}
    for key in history_phase1.history.keys():
        combined_history[key] = history_phase1.history[key] + history_phase2.history[key]
    
    # Create a history object for plotting
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    combined_history_obj = CombinedHistory(combined_history)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(combined_history_obj)
    
    # Evaluate final model
    print("\nEvaluating final model...")
    evaluate_model(model, val_dataset, val_processor)
    
    # Save model and processors
    print("\nSaving model and encoders...")
    save_model_and_processors(model, train_processor, "utkface_facenet_final")
    
    print("\n" + "="*60)
    print("Training Pipeline Complete!")
    print("="*60)
    print("Files created:")
    print("- best_utkface_facenet_finetuned_model.h5 (best model)")
    print("- utkface_facenet_final.h5 (final model)")
    print("- utkface_facenet_final_encoders.pkl (label encoders)")
    print("- utkface_training_log.csv (phase 1 training log)")
    print("- utkface_finetuning_log.csv (phase 2 training log)")
    print("- training_history.png (training plots)")
    print("- confusion_matrices.png (evaluation plots)")
    
    # Example prediction on a sample image
    if len(train_processor.df) > 0:
        print("\n" + "="*60)
        print("Testing Prediction on Sample Image")
        print("="*60)
        
        # Get a random sample from training data
        sample_row = train_processor.df.sample(n=1).iloc[0]
        sample_img_path = os.path.join(DATASET_PATH, sample_row['file'])
        
        print(f"Testing on: {sample_row['file']}")
        print(f"True labels - Age: {sample_row['age_bin']}, Gender: {sample_row['gender']}, Race: {sample_row['race']}")
        
        try:
            predicted_age, predicted_gender, predicted_race = predict_sample_utkface(
                model, sample_img_path, train_processor
            )
            print(f"Predicted - Age: {predicted_age}, Gender: {predicted_gender}, Race: {predicted_race}")
        except Exception as e:
            print(f"Error in prediction: {e}")
    
    print("\nTraining pipeline completed successfully!")
    
    # Return values for interactive use (when called from other scripts)
    # Note: This only works if called as a function, not when run as main script
    if 'model' in locals():
        pass  # Variables are available in local scope for further use

# Additional utility functions
def predict_from_folder(model_path, encoders_path, image_folder, output_csv="predictions.csv"):
    """Predict demographics for all images in a folder"""
    # Load model and encoders
    model, encoders = load_model_and_processors(model_path, encoders_path)
    
    # Create a dummy processor with the encoders
    class DummyProcessor:
        pass
    
    processor = DummyProcessor()
    processor.age_encoder = encoders['age_encoder']
    processor.gender_encoder = encoders['gender_encoder']
    processor.race_encoder = encoders['race_encoder']
    
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    
    print(f"Processing {len(image_files)} images...")
    
    # Process images
    results = []
    for i, img_path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"Processed {i}/{len(image_paths)} images...")
        
        try:
            age, gender, race = predict_sample_utkface(model, img_path, processor, img_size=(160, 160))
            results.append({
                'filename': os.path.basename(img_path),
                'predicted_age': age,
                'predicted_gender': gender,
                'predicted_race': race
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append({
                'filename': os.path.basename(img_path),
                'predicted_age': 'Error',
                'predicted_gender': 'Error',
                'predicted_race': 'Error'
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    
    return results_df

# Example usage for inference
def inference_example():
    """Example of how to use the trained model for inference"""
    MODEL_PATH = "utkface_facenet_final.h5"
    ENCODERS_PATH = "utkface_facenet_final_encoders.pkl"
    
    # Load model and encoders
    try:
        model, encoders = load_model_and_processors(MODEL_PATH, ENCODERS_PATH)
        
        # Create processor object for predictions
        class InferenceProcessor:
            def __init__(self, encoders):
                self.age_encoder = encoders['age_encoder']
                self.gender_encoder = encoders['gender_encoder']
                self.race_encoder = encoders['race_encoder']
        
        processor = InferenceProcessor(encoders)
        
        # Example: predict on a single image
        sample_image = "path/to/your/test/image.jpg"
        if os.path.exists(sample_image):
            age, gender, race = predict_sample_utkface(model, sample_image, processor)
            print(f"Predictions: Age={age}, Gender={gender}, Race={race}")
        
        # Example: predict on a folder of images
        test_folder = "path/to/test/images"
        if os.path.exists(test_folder):
            results = predict_from_folder(MODEL_PATH, ENCODERS_PATH, test_folder)
            print(f"Processed {len(results)} images")
            
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Please run the training first or check file paths")

# Training configuration options
class TrainingConfig:
    """Configuration class for easy parameter tuning"""
    def __init__(self):
        # Dataset parameters
        self.dataset_path = "C:/Users/DELL/Downloads/datasets/UTKface_inthewild/part1"
        self.img_size = 160
        self.sample_size = None  # Use None for full dataset
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 20
        self.initial_lr = 0.001
        self.finetune_lr = 0.0001
        
        # Model parameters
        self.freeze_backbone_initially = True
        self.dropout_rate = 0.7
        self.shared_dense_units = 512
        self.age_dense_units = 128
        
        # Loss weights (adjust based on task importance)
        self.loss_weights = {
            'age_output': 1.0,
            'gender_output': 1.0,
            'race_output': 1.0
        }
        
        # Callback parameters
        self.early_stopping_patience = 10
        self.reduce_lr_patience = 5
        self.reduce_lr_factor = 0.5

def train_with_config(config):
    """Train model using configuration object"""
    print("Training with configuration:")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Image size: {config.img_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Sample size: {config.sample_size}")
    
    # Create data processors
    train_processor = UTKFaceDataProcessor(
        config.dataset_path,
        img_size=config.img_size,
        sample_size=config.sample_size,
        is_validation=False
    ).filter_missing_files()
    
    # Split data
    train_df, val_df = train_test_split(
        train_processor.df, 
        test_size=0.2, 
        random_state=42,
        stratify=train_processor.df['age_bin']
    )
    
    # Update processors with split data
    train_processor.df = train_df.reset_index(drop=True)
    
    val_processor = UTKFaceDataProcessor(
        config.dataset_path,
        img_size=config.img_size,
        sample_size=None,
        is_validation=True
    )
    val_processor.df = val_df.reset_index(drop=True)
    
    # Copy encoders
    val_processor.age_encoder = train_processor.age_encoder
    val_processor.gender_encoder = train_processor.gender_encoder
    val_processor.race_encoder = train_processor.race_encoder
    val_processor.num_classes = train_processor.num_classes
    
    # Re-encode validation labels
    val_processor.df['age_encoded'] = val_processor.age_encoder.transform(val_processor.df['age_bin'])
    val_processor.df['gender_encoded'] = val_processor.gender_encoder.transform(val_processor.df['gender'])
    val_processor.df['race_encoded'] = val_processor.race_encoder.transform(val_processor.df['race'])
    
    # Create datasets
    train_dataset = train_processor.create_dataset(
        batch_size=config.batch_size,
        augment=True,
        shuffle=True
    )
    
    val_dataset = val_processor.create_dataset(
        batch_size=config.batch_size,
        augment=False,
        shuffle=False
    )
    
    # Create and compile model
    model = create_and_compile_model(
        num_age_classes=train_processor.num_classes['age'],
        num_gender_classes=train_processor.num_classes['gender'],
        num_race_classes=train_processor.num_classes['race'],
        freeze_backbone=config.freeze_backbone_initially
    )
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_utkface_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    # Train the model
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate and save
    plot_training_history(history)
    evaluate_model(model, val_dataset, val_processor)
    save_model_and_processors(model, train_processor)
    
    return model, history, train_processor, val_processor

# Run training if this script is executed directly
if __name__ == "__main__":
    try:
        # Option 1: Use configuration-based training
        print("Starting UTKFace training...")
        config = TrainingConfig()
        model, history, train_processor, val_processor = train_with_config(config)
        
        # Option 2: Use custom configuration (uncomment to use)
        # config = TrainingConfig()
        # config.epochs = 30
        # config.sample_size = 50000  # Use subset for faster training
        # model, history, train_processor, val_processor = train_with_config(config)
        
        print("\nTraining completed successfully!")
        print("All files saved. You can now use the trained model for inference.")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    # Uncomment to run inference example after training
    # inference_example()