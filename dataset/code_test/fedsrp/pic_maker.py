import matplotlib.pyplot as plt
import numpy as np

def mk_nse_pic(basin_id, nse_list):
    local_nse = []
    mutil_fune_nse = []
    transfer_a_nse = []
    transfer_b_nse = []
    srp_nse = []
    for item in nse_list:
        local_nse.append(item[0])
        mutil_fune_nse.append(item[1])
        transfer_a_nse.append(item[2])
        transfer_b_nse.append(item[3])
        srp_nse.append(item[4])
    plt.clf()
    width = 1
    plt.plot(local_nse, color=(210/255,180/255,140/255), linewidth=width, marker='o')
    plt.plot(mutil_fune_nse, color=(56/255,232/255,176/255), linewidth=width, marker='*')
    plt.plot(transfer_a_nse, color=(27/255,175/255,208/255), linewidth=width, marker='^')
    plt.plot(transfer_b_nse, color=(105/255,103/255,206/255), linewidth=width, marker='D')
    plt.plot(srp_nse, color=(253/255,99/255,107/255), linewidth=width, marker='s')
    # plt.legend()
    # plt.gca().set_xticklabels([''] * len(plt.gca().get_xticks()))
    # plt.gca().set_yticklabels([''] * len(plt.gca().get_yticks()))
    plt.savefig('/home/FedHydro/dataset/code_test/fedsrp/nse_' + basin_id + '.pdf', dpi=300)
    print('basin:' + basin_id + 'is finished')

def mk_rmse_pic(basin_id, rmse_list, value):
    local_rmse = []
    mutil_fune_rmse = []
    transfer_a_rmse = []
    transfer_b_rmse = []
    srp_rmse = []
    avg_rmse = []
    for item in rmse_list:
        local_rmse.append(item[0])
        mutil_fune_rmse.append(item[1])
        transfer_a_rmse.append(item[2])
        transfer_b_rmse.append(item[3])
        srp_rmse.append(item[4])
        avg_rmse.append(item[5])
    plt.clf()
    width = 1
    plt.plot(local_rmse, color=(210/255,180/255,140/255), linewidth=width, marker='o')
    plt.plot(mutil_fune_rmse, color=(56/255,232/255,176/255), linewidth=width, marker='*')
    plt.plot(transfer_a_rmse, color=(27/255,175/255,208/255), linewidth=width, marker='^')
    plt.plot(transfer_b_rmse, color=(105/255,103/255,206/255), linewidth=width, marker='D')
    plt.plot(avg_rmse, color=(169/255,169/255,169/255), linewidth=width, marker='p')
    plt.plot(srp_rmse, color=(253/255,99/255,107/255), linewidth=width, marker='s')

    num_ticks = 6  # 设置刻度的个数
    plt.xlim(-0.5, 5.5)
    min = 0.002
    max = 0.006
    plt.ylim(min, max)

    y_ticks = np.linspace(min, max, num_ticks)
    plt.yticks(y_ticks)


    plt.grid(linewidth=0.5)

    if not value:
        plt.gca().set_xticklabels([''] * len(plt.gca().get_xticks()))
        plt.gca().set_yticklabels([''] * len(plt.gca().get_yticks()))
    plt.savefig('/home/FedHydro/dataset/code_test/fedsrp/yyy_rmse_' + basin_id + '.jpg', dpi=600)
    print('basin:' + basin_id + 'is finished')

def mk_mae_pic(basin_id, mae_list, value):
    local_mae = []
    mutil_fune_mae = []
    transfer_a_mae = []
    transfer_b_mae = []
    srp_mae = []
    avg_mae = []
    for item in mae_list:
        local_mae.append(item[0])
        mutil_fune_mae.append(item[1])
        transfer_a_mae.append(item[2])
        transfer_b_mae.append(item[3])
        srp_mae.append(item[4])
        avg_mae.append(item[5])
    plt.clf()
    width = 1
    plt.plot(local_mae, color=(210/255,180/255,140/255), linewidth=width, marker='o')
    plt.plot(mutil_fune_mae, color=(56/255,232/255,176/255), linewidth=width, marker='*')
    plt.plot(transfer_a_mae, color=(27/255,175/255,208/255), linewidth=width, marker='^')
    plt.plot(transfer_b_mae, color=(105/255,103/255,206/255), linewidth=width, marker='D')
    plt.plot(avg_mae, color=(169/255,169/255,169/255), linewidth=width, marker='p')
    plt.plot(srp_mae, color=(253/255,99/255,107/255), linewidth=width, marker='s')

    # plt.legend()

    #plt.xlim(-0.5, 5.5)
    #plt.ylim(0.0015, 0.006)
    #plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=6))

    #num_ticks = 6
    #max_num = 0.006
    #min_num = 0.0015
    #step = ((max_num - min_num) / (num_ticks - 1))
    #yticks = np.arange(min_num, max_num + step, step)
    #plt.yticks(yticks)


    num_ticks = 6  # 设置刻度的个数
    plt.xlim(-0.5, 5.5)
    min = 0.5
    max = 5.5
    plt.ylim(min, max)

    y_ticks = np.linspace(min, max, num_ticks)
    plt.yticks(y_ticks)


    plt.grid(linewidth=0.5)

    if not value:
        plt.gca().set_xticklabels([''] * len(plt.gca().get_xticks()))
        plt.gca().set_yticklabels([''] * len(plt.gca().get_yticks()))
    plt.savefig('/home/FedHydro/dataset/code_test/fedsrp/vvv_mae_' + basin_id + '.jpg', dpi=600)
    print('basin:' + basin_id + 'is finished')


if __name__ == '__main__':
    basin03 = '02479155'
    basin01 = '01169000'
    basin18 = '11151300' # 18
    basin17 = '13331500' # 17

    rmse_list3 = []

    # mk_nse_pic(basin_id, nse_list_03)
    mk_rmse_pic(basin18, rmse18_h, 0)
    #mk_mae_pic(basin01, mae01_b, 0)
