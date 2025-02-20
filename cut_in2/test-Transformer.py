import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.Transformer_model import AgentPredictor
from dataset.single_traj_dataset2 import PredDataset

# Set which GPU to use
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# Define paths and sample index
data_path = '/data/wangzm/merge/IDM_data/IDM_50.pkl'
model_path = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output/Transformer-three-to-three/86p56nx1/checkpoints/last.ckpt'
output_path = '/data/wangzm/merge/Merge-HighwayEnv-RL17/cut_in2/output/Transformer-test'
sample_index = 2050

class Args:
    data_path = data_path
    pred_length = 40  # Set the appropriate prediction length
    hist_length = 10  # Set the appropriate history length
    num_layers = 2  # Set the number of LSTM layers
    lr = 1e-3  # Learning rate
    batch_size = 32  # Batch size
    max_epochs = 4000  # Max epochs
    num_workers = 1  # Number of workers

def plot_trajectory(gt_traj, gt_traj_ahead, gt_traj_right, pred_traj,pred_pos_ahead,pred_pos_right, ego_stat, ahead_stat, right_stat, ego_map, label, output_path):
    plt.figure(figsize=(10, 5))
    
    # Plot ahead_stat, right_stat
    plt.plot(ahead_stat[:, 0], ahead_stat[:, 1], label='Ahead State', linestyle='-', marker='s', color = 'darkorange')
    plt.plot(gt_traj_ahead[:, 0], gt_traj_ahead[:, 1], label='Ahead Ground Truth', marker='s', color = 'blue')
    plt.plot(pred_pos_ahead[:, 0], pred_pos_ahead[:, 1], label='Ahead Predicted path', marker='x', color = 'purple')

    plt.plot(right_stat[:, 0], right_stat[:, 1], label='Right State', linestyle='-', marker='d', color = 'darkorange')
    plt.plot(gt_traj_right[:, 0], gt_traj_right[:, 1], label='Right Ground Truth', marker='d', color = 'blue')
    plt.plot(pred_pos_right[:, 0], pred_pos_right[:, 1], label='Right Predicted path', marker='x', color = 'purple')
    
    # Plot ego_map with dashed lines
    plt.plot(ego_map[:, 0], ego_map[:, 1], label='Center line', linestyle='--', color='gray')
    plt.plot(ego_map[:, 0], ego_map[:, 2], linestyle='--', color='gray')

    # Plot ground truth and predicted trajectory
    plt.plot(ego_stat[:, 0], ego_stat[:, 1], label='Ego State', linestyle='-', marker='o', color = 'darkorange')
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], label='Ego Ground Truth', marker='o', color = 'blue')
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], label='Ego Predicted path', marker='x', color = 'red')

    # Add annotation based on the label
    if label == 1:
        plt.annotate('ego type is aggressive', xy=(0.01, 0.01), xycoords='axes fraction', ha='left', fontsize=12, color='red')
    else:
        plt.annotate('ego type is friendly', xy=(0.01, 0.01), xycoords='axes fraction', ha='left', fontsize=12, color='green')
    
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    plt.xlim(-5,20)
    plt.ylim(-2,5.5)
    plt.title('Transformer Trajectory Comparison (IDM12)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f'Transformer_{sample_index}_HD40.png'))
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
    label = sample_data['ego_stat'][0, -2].item()

    # Load model
    config = {
        'map_input_size': 3,
        'map_hidden_size': 256,
        'agent_his_input_dim': 5,
        'agent_his_hidden_size': 256,
        'fusion_hidden_size': 256,
        'num_fusion_layer': args.num_layers,
        'dropout_rate': 0.1,
        'in_hidden_dim': 256,
        'head_hidden_dim': 256,
        'out_traj_dim': args.pred_length * 5,
        'num_queries': 40,
        'num_decoder_layer': args.num_layers,
        'aux_loss_temponet': True,
        'aux_loss_spanet': True,
        'lr': args.lr
    }

    model = AgentPredictor(config)
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
            'right_stat': sample_data['right_stat'].unsqueeze(0).float().cuda(),
            'ego_fut': sample_data['ego_fut'].unsqueeze(0).float().cuda(),  # Added
            'ahead_fut': sample_data['ahead_fut'].unsqueeze(0).float().cuda(),  # Added
            'right_fut': sample_data['right_fut'].unsqueeze(0).float().cuda()  # Added
        }

        # Get predictions
        output, _ = model(sample_batch)

    # Extract ground truth and predictions for comparison
    gt_traj = sample_data['ego_fut'][:, :2].numpy()
    gt_traj_ahead = sample_data['ahead_fut'][:, :2].numpy()
    gt_traj_right = sample_data['right_fut'][:, :2].numpy()
    gt_vel = sample_data['ego_fut'][:,3:5]
    print(gt_vel)

    pred_traj = output['pos'].squeeze(0).cpu().numpy()
    pred_pos_ahead = output['agent'][:, 1,:,:2].squeeze(0).cpu().numpy()
    pred_pos_right = output['agent'][:, 2,:,:2].squeeze(0).cpu().numpy()
    pred_vel = output['vel'].squeeze(0).cpu().numpy()
    print(pred_vel)

    # Extract ego_stat, ahead_stat, right_stat, and ego_map
    ego_stat = sample_data['ego_stat'][:, :2].numpy()
    ahead_stat = sample_data['ahead_stat'][:, :2].numpy()
    right_stat = sample_data['right_stat'][:, :2].numpy()
    ego_map = sample_data['ego_map'].numpy()

    # Plot and save the trajectory comparison
    plot_trajectory(gt_traj, gt_traj_ahead, gt_traj_right, pred_traj,pred_pos_ahead,pred_pos_right, ego_stat, ahead_stat, right_stat, ego_map, label, output_path)

    # Save the ground truth and predicted trajectories
    # np.save(os.path.join(output_path, 'ground_truth.npy'), gt_traj)
    # np.save(os.path.join(output_path, 'predicted.npy'), pred_traj)

if __name__ == '__main__':
    main()


