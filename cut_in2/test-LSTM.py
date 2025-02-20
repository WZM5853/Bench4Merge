import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.LSTM_model import PredModel
from dataset.single_traj_dataset2 import PredDataset

# Set which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Define paths and sample index
data_path = '/data/wangzm/merge/final_data/IDM_71+DJI_27_withright+DJI_13.5_noright.pkl'
model_path = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output/LSTM-three_vehicle-to-ego/9ij5zw3w/checkpoints/epoch=51-val_fde=0.16.ckpt'
output_path = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output/LSTM-test'
sample_index = 3000

class Args:
    data_path = data_path
    pred_length = 40  # Set the appropriate prediction length
    hist_length = 10  # Set the appropriate history length
    num_layers = 2  # Set the number of LSTM layers
    lr = 1e-3  # Learning rate
    batch_size = 32  # Batch size
    max_epochs = 4000  # Max epochs
    num_workers = 1  # Number of workers

def plot_trajectory(gt, pred, ego_stat, ahead_stat, right_stat, ego_map, label, output_path):
    plt.figure(figsize=(10, 5))
    
    # Plot ground truth and predicted trajectory
    plt.plot(gt[:, 0], gt[:, 1], label='Ground Truth', marker='o', color = 'blue')
    plt.plot(pred[:, 0], pred[:, 1], label='Predicted path', marker='x', color = 'red')
    
    # Plot ego_stat, ahead_stat, right_stat
    plt.plot(ego_stat[:, 0], ego_stat[:, 1], label='Ego State', linestyle='-', marker='o', color = 'darkorange')
    plt.plot(ahead_stat[:, 0], ahead_stat[:, 1], label='Ahead State', linestyle='-', marker='s', color = 'darkorange')
    plt.plot(right_stat[:, 0], right_stat[:, 1], label='Right State', linestyle='-', marker='d', color = 'darkorange')
    
    # Plot ego_map with dashed lines
    plt.plot(ego_map[:, 0], ego_map[:, 1], label='Ego Map', linestyle='--', color='gray')
    plt.plot(ego_map[:, 0], ego_map[:, 2], linestyle='--', color='gray')

    # Add annotation based on the label in the bottom-left corner
    if label == 1:
        plt.annotate('ego type is aggressive', xy=(0.01, 0.01), xycoords='axes fraction', ha='left', fontsize=12, color='red')
    else:
        plt.annotate('ego type is friendly', xy=(0.01, 0.01), xycoords='axes fraction', ha='left', fontsize=12, color='green')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-5,20)
    plt.ylim(-2,5.5)
    plt.title('LSTM Trajectory Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f'LSTM_{sample_index}_IDM+DJI.png'))
    plt.show()

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load dataset
    args = Args()
    dataset = PredDataset(args)
    sample_data = dataset[sample_index]

    # Extract the label from the last column of ego_stat
    label = sample_data['ego_stat'][0, -1].item()

    # Load model
    model = PredModel(args)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()  # Move model to GPU
    model.eval()

    # Prepare data for prediction
    with torch.no_grad():
        sample_batch = {
            'ego_stat': sample_data['ego_stat'].unsqueeze(0).float().cuda(),
            'ego_map': sample_data['ego_map'].unsqueeze(0).float().cuda(),
            'ego_map_mask': torch.ones_like(sample_data['ego_map'][:, 0]).unsqueeze(0).float().cuda(),
            'ahead_stat': sample_data['ahead_stat'].unsqueeze(0).float().cuda(),
            'right_stat': sample_data['right_stat'].unsqueeze(0).float().cuda()
        }

        # Get predictions
        output = model(sample_batch)

    # Extract ground truth and predictions for comparison
    gt_traj = sample_data['ego_fut'][:, :2].numpy()
    pred_traj = output['pos'].squeeze(0).cpu().numpy()

    # Extract ego_stat, ahead_stat, right_stat, and ego_map
    ego_stat = sample_data['ego_stat'][:, :2].numpy()
    ahead_stat = sample_data['ahead_stat'][:, :2].numpy()
    right_stat = sample_data['right_stat'][:, :2].numpy()
    ego_map = sample_data['ego_map'].numpy()

    # Plot and save the trajectory comparison
    plot_trajectory(gt_traj, pred_traj, ego_stat, ahead_stat, right_stat, ego_map, label, output_path)

    # Save the ground truth and predicted trajectories
    # np.save(os.path.join(output_path, 'ground_truth.npy'), gt_traj)
    # np.save(os.path.join(output_path, 'predicted.npy'), pred_traj)

if __name__ == '__main__':
    main()

