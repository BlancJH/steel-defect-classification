import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Use the locally unzipped data
    data_dir = os.path.join(os.getcwd(), 'data', 'severstal-steel-defect-detection')
    csv_path = os.path.join(data_dir, "train.csv")
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}. Make sure the dataset is extracted correctly.")
        sys.exit(1)
        
    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total rows in train.csv: {len(df)}")
    
    if 'ImageId' in df.columns:
        image_col = 'ImageId'
    elif 'ImageId_ClassId' in df.columns:
        image_col = 'ImageId_ClassId'
    else:
        print("Unknown image column. Columns:", df.columns)
        sys.exit(1)
        
    # We want to split safely so an image entirely belongs to only one split
    if image_col == 'ImageId_ClassId':
        df['BaseImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    else:
        df['BaseImageId'] = df[image_col]

    unique_images = df['BaseImageId'].unique()
    print(f"Total unique images in train: {len(unique_images)}")
    
    # 80/10/10 Split
    # First, split 80% Train and 20% Temp (Val + Test)
    train_imgs, temp_imgs = train_test_split(unique_images, test_size=0.2, random_state=42)
    # Then split Temp 50/50 so Val and Test each get 10% of total
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
    
    print(f"Train images: {len(train_imgs)}, Validation images: {len(val_imgs)}, Test images: {len(test_imgs)}")
    
    # Create the dataframes for the splits
    train_df = df[df['BaseImageId'].isin(train_imgs)].drop(columns=['BaseImageId'])
    val_df = df[df['BaseImageId'].isin(val_imgs)].drop(columns=['BaseImageId'])
    test_df = df[df['BaseImageId'].isin(test_imgs)].drop(columns=['BaseImageId'])
    
    print(f"Train split size: {len(train_df)} rows")
    print(f"Validation split size: {len(val_df)} rows")
    print(f"Test split size: {len(test_df)} rows")
    
    # Define save paths (saving to data/ directly)
    save_dir = os.path.join(os.getcwd(), 'data')
    train_save_path = os.path.join(save_dir, 'train_split.csv')
    val_save_path = os.path.join(save_dir, 'val_split.csv')
    test_save_path = os.path.join(save_dir, 'test_split.csv')
    
    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)
    
    print(f"\nSaved training split to: {train_save_path}")
    print(f"Saved validation split to: {val_save_path}")
    print(f"Saved test split to: {test_save_path}")

if __name__ == '__main__':
    main()
