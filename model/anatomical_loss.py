import torch
import torch.nn as nn
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


class AnatomicalLoss(nn.Module):
    def __init__(self, min_angs, max_angs, min_lens, max_lens, root_index):
        super(AnatomicalLoss, self).__init__()
        self.min_angs = min_angs
        self.max_angs = max_angs

        self.min_lens = min_lens
        self.max_lens = max_lens

        self.root_index = root_index

    def _calc_loss(self, x, mins, maxs):
        min_mask = ((x - mins) < 0.0).float()
        max_mask = ((x - maxs) > 0.0).float()

        min_loss = ((x - mins) * min_mask).norm(dim=-1, p=2).mean()
        max_loss = ((x - maxs) * max_mask).norm(dim=-1, p=2).mean()

        anat_loss = (min_loss + max_loss) / 2.0
        
        return anat_loss

    def forward(self, x):
        angs = calc_joints_angles(x, self.root_index)
        lens = calc_segments_lengths(x)

        ang_loss = self._calc_loss(angs, self.min_angs, self.max_angs)
        len_loss = self._calc_loss(lens, self.min_lens, self.max_lens)

        return ang_loss, len_loss


def calc_segments_lengths(hand_poses):
    '''
    calculates the length of the segment between different hand joints.

    Params:
        - hand_poses (tensor): a batch of hand poses sequences, shape (batch_size, num_frames, num_joints, 3)

    Returns:
        - segments_lengths (tensor): a tensor of shape (B, T, N, N)
         contains the length of the segment between each pair of hand joints.
    '''
    ## compute differences between all pairs of joints
    differences = hand_poses.unsqueeze(2) - hand_poses.unsqueeze(3)
    
    ## compute segment lengths matrix
    segments_lengths = torch.norm(differences, dim=-1)
    
    return segments_lengths

def calc_joints_angles(hand_poses, root_index):
    '''
    Calculates the angles in radians between different hand joints given the root as reference.
    For each pair of joints, computes the angle between the vectors (root, joint_i) and (root, joint_j).

    Params:
        - hand_poses (tensor): shape (B, T, N, 3), batch of hand pose sequences.
        - root_index (int): index of the root joint.

    Returns:
        - thetas (tensor): shape (B, T, N, N), angles between pairs of joints in radians.
    '''
    # Extract root pose and joint coordinates
    root_pose = hand_poses[:, :, root_index, :2].unsqueeze(2)  # Shape: (B, T, 1, 2)
    poses_xy = hand_poses[:, :, :, :2]  # Shape: (B, T, N, 2)

    # Compute vectors from root to each joint
    vectors = poses_xy - root_pose  # Shape: (B, T, N, 2)

    # Normalize vectors to unit length
    norms = torch.norm(vectors, dim=-1, keepdim=True) + 1e-6  # Avoid division by zero
    unit_vectors = vectors / norms  # Shape: (B, T, N, 2)

    # Compute dot product between all pairs of unit vectors
    dot_products = torch.matmul(unit_vectors, unit_vectors.transpose(-2, -1))  # Shape: (B, T, N, N)

    # Clamp values to [-1, 1] to avoid numerical errors in acos
    dot_products = torch.clamp(dot_products, -1.0, 1.0)

    # Compute angles using arccos
    thetas = torch.acos(dot_products)  # Shape: (B, T, N, N)

    return thetas




def get_data_stats(data_loader, root_index):
    '''
    Returns the ranges of the variation of the length and angles of hand joints.
    '''
    all_angles = []
    all_lengths = []
    
    for d in tqdm(data_loader, desc='calculating data stats....' ):
        P = d['Sequence']
        
        angles = calc_joints_angles(P, root_index)
        all_angles.extend(angles.view(-1, angles.shape[-2], angles.shape[-1]))
            
        lengths = calc_segments_lengths(P)
        all_lengths.extend(lengths.view(-1, lengths.shape[-2], lengths.shape[-1]))
            
    all_angles = torch.stack(all_angles)
    all_lengths = torch.stack(all_lengths)
    
    min_angles = all_angles.min(axis=0)[0]    
    max_angles = all_angles.max(axis=0)[0] 
    
    min_lenghts = all_lengths.min(axis=0)[0]     
    max_lenghts = all_lengths.max(axis=0)[0] 
    
    angles_stats = {'min': min_angles, 
                    'max': max_angles} 

    lengths_stats = {'min': min_lenghts,
                     'max': max_lenghts} 
    
    #assert False , angles_stats
    return angles_stats, lengths_stats


def plot_data_stats(angles_stats, lengths_stats, show=False, fname=None):
    '''
    Plots the ranges of the angles and lengths between different hand joints
    '''

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    sns.heatmap(angles_stats['min'],  ax=ax1, square=True)
    sns.heatmap(angles_stats['max'],  ax=ax2, square=True)
    sns.heatmap(lengths_stats['min'], ax=ax3, square=True)
    sns.heatmap(lengths_stats['max'], ax=ax4, square=True)
    ax1.set_title('Min Angles')
    ax2.set_title('Max Angles')
    ax3.set_title('Min Lengths')
    ax4.set_title('Max Lengths')
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)
    if show:
        plt.show()