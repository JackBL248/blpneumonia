import argparse

parser = argparse.ArgumentParser(
    description="Train CNNs on Chest X-ray images for detection of pneumonia")
# ========================= Data Configs ==========================
parser.add_argument('--datafolder', default="toy_dataset", type=str,
                    help='path to data folder')

# ========================= Model Configs ==========================
parser.add_argument('--model', default="resnet34", type=str,
                    help='Type of CNN architecture')

# ===================== Data Augmentation Configs ==================
parser.add_argument('--rotation', default=10, type=int,
                    help='Range for random rotation')
parser.add_argument('--scale', default=[0.9, 1.1], type=list,
                    help='Range for random scaling')
parser.add_argument('--shear', default=10, type=int,
                    help='Range for random shear')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=25, type=int,
                    help='number of total epochs to run')
parser.add_argument('--patience', default=10, type=int,
                    help='early stopping patience')
parser.add_argument('-b', '--batch', default=8, type=int,
                    help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--dropout', default=0.25, type=float,
                    help='model dropout')
parser.add_argument('--workers', default=4, type=int,
                    help='number of workers')

# ========================= Log Configs ==========================
parser.add_argument('--log', default="log.txt", type=str,
                    help='Destination file for logging results')
