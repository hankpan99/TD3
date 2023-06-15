import glob
import numpy as np
import matplotlib.pyplot as plt

def smoothing(scalars, weight=0.6):
    last = scalars[0] 
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed

def load_rewards(envs_list, ablation_list):
    rewards_dict = {}

    for env in envs_list:
        rewards_dict[env] = {}

        for ablation in ablation_list:
            pth_list = sorted(glob.glob('results/TD3_{}_?_{}.npy'.format(env, ablation)))
            
            tmp_arr = np.array([np.load(pth) for pth in pth_list])

            try:
                avg_rewards = np.sum(tmp_arr, axis=0) / tmp_arr.shape[0]
            except:
                tmp_arr = tmp_arr[:-1]
                avg_rewards = np.sum(tmp_arr, axis=0) / tmp_arr.shape[0]
            
            rewards_dict[env][ablation] = avg_rewards.copy()
    
    return rewards_dict

if __name__ == '__main__':
    envs_list = ['Walker2d-v3', 'Ant-v3']
    ablation_list = ['delay', 'noise', 'clip', 'noise_clip', 'delay_clip', 'delay_noise', 'delay_noise_clip', 'all']

    rewards_dict = load_rewards(envs_list, ablation_list)

    component_list = [['delay', 'noise', 'clip', 'all', 'delay_noise_clip'], ['noise_clip', 'delay_clip', 'delay_noise', 'all', 'delay_noise_clip']]
    color_list = [['purple', 'red', 'brown', 'blue', 'green'], ['purple', 'red', 'brown', 'blue', 'green']]
    comp2label_dict = {'delay': 'TD3 - DP',
                       'noise': 'TD3 - TPS',
                       'clip': 'TD3 - CDQ',
                       'noise_clip': 'AHE + DP',
                       'delay_clip': 'AHE + TPS',
                       'delay_noise': 'AHE + CDQ',
                       'all': 'TD3',
                       'delay_noise_clip': 'AHE'}
    fig_title_list = ['Remove 1 Component', 'Remove 2 Components']
    x_axis = np.arange(0, 1e6 + 1, 5e3, dtype=int)

    for env in envs_list:
        fig = plt.figure(figsize=(13, 20))
        # fig.suptitle("{}".format(env), fontsize=20)
        ax_list = [fig.add_subplot(2, 1, i + 1) for i in range(2)]

        for i in range(2):
            for cnt, comp in enumerate(component_list[i]):
                # smooth the rewards
                rewards_smooth = smoothing(rewards_dict[env][comp], 0.9)
                ax_list[i].plot(x_axis, rewards_smooth, label=comp2label_dict[comp], linewidth=3, color=color_list[i][cnt])
            
            # set title
            ax_list[i].set_title(fig_title_list[i], size=30)

            # set x, y label
            ax_list[i].set_xlabel('Time Steps (1e6)', fontsize=25)
            ax_list[i].set_ylabel('Average Rewards', fontsize=25)

            # set tick font size
            ax_list[i].tick_params(axis='both', labelsize=25)

            # legend
            ax_list[i].legend(fontsize=25)

            # modify spacing
            fig.tight_layout()

        plt.savefig('{}.png'.format(env))
        plt.show()