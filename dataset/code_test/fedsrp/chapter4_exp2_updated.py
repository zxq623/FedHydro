import sys
sys.path.append("/home/FedHydro/")

from dataset.code_test.hydro_lstm_test.save_script_ import cal_nse_rmse_mae, cal_nse_rmse_rae
from dataset.code_test.fedsrp.chapter4_exp2 import get_merge_data, get_date_range, get_sparse_data, train_test_data, mutil_fune, transfer_a, transfer_b, fed_hydro
import pandas as pd
import nn
from utils.constants import local_basic_lstm
import numpy as np
import matplotlib.pyplot as plt

local_basic_model_name = "/home/FedHydro/dataset/code_test/fedsrp/models/basic_local_model.model"

rich_date_range = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1995-09-30", format="%Y-%m-%d")  # 2年训练期
        },
        'val_date': {
            'start_date': pd.to_datetime("2010-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2005-09-30", format="%Y-%m-%d")
        },
        'test_date': {
            'start_date': pd.to_datetime("2006-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2008-09-30", format="%Y-%m-%d")
        },
    }

"""实验均使用val_date时期数据进行测试，test_date不用管"""
# 01047000
date_range1 = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1982-09-30", format="%Y-%m-%d")  # 2年训练期
        },
        'val_date': {
            'start_date': pd.to_datetime("2008-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2010-09-30", format="%Y-%m-%d")
        },
        'test_date': {
            'start_date': pd.to_datetime("2002-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2004-09-30", format="%Y-%m-%d")
        },
    }

# 01054200
date_range2 = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1982-09-30", format="%Y-%m-%d")  # 2年训练期
        },
        'val_date': {
            'start_date': pd.to_datetime("2007-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2009-09-30", format="%Y-%m-%d")
        },
        'test_date': {
            'start_date': pd.to_datetime("2000-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2002-09-30", format="%Y-%m-%d")
        },
    }

# 01055000
date_range3 = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1982-09-30", format="%Y-%m-%d")  # 2年训练期
        },
        'val_date': {
            'start_date': pd.to_datetime("2007-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2009-09-30", format="%Y-%m-%d")  # 测试日期1，差异性大， 数据太少不设置验证期
        },
        'test_date': {
            'start_date': pd.to_datetime("2004-10-01", format="%Y-%m-%d"),  # 测试日期2， 差异性小
            'end_date': pd.to_datetime("2006-09-30", format="%Y-%m-%d")
        },
    }


# def fedsrp_test(rich_basin_ids, sparse_basin_ids, train_data_list, test_data_list, ds_test_list):
#     """废弃"""
#     round = 70
#     eval_round = 2  # 5*5个epoch验证一次
#     local_epoch = 3
#     batch_size = 256
#     target_basin_batch_size = 64
#     num_basins = len(rich_basin_ids) + len(sparse_basin_ids)
#     train_split = 0.7
#     model_path = local_basic_model_name
#     fed_fomo = FedFomo(model_path, round, eval_round, local_epoch, batch_size, target_basin_batch_size,
#                        num_basins, train_data_list, test_data_list, ds_test_list, train_split, len(sparse_basin_ids), sparse_basin_ids)
#     fed_fomo.run()


def correct_date_range_test():
    # a = ["1996-10-01", "1998-09-30"]
    # b = ["1998-10-01", "2000-09-30"]
    c = ["2000-10-01", "2002-09-30"]
    d = ["2002-10-01", "2004-09-30"]
    e = ["2004-10-01", "2006-09-30"]
    f = ["2006-10-01", "2008-09-30"]
    g = ["2008-10-01", "2010-09-30"]
    # h = ["20-10-01", "2010-09-30"]
    # i = ["2008-10-01", "2010-09-30"]
    date_range_list = []
    # date_range_list.append(a)
    # date_range_list.append(b)
    date_range_list.append(c)
    date_range_list.append(d)
    date_range_list.append(e)
    date_range_list.append(f)
    date_range_list.append(g)

    return date_range_list


def train_local_lstm(local_model, single_basin_data, epoch, batch_size):
    """仅用本地数据训练"""
    train_x = single_basin_data[0]
    train_y = single_basin_data[1]
    local_model.fit(train_x, train_y, epoch=epoch, batch_size=batch_size)
    return local_model


def local_lstm(basin_id, test_date, local_basic_model_name, single_basin_data, test_x, test_y, single_ds_test, epoch,  batch_size):
    """生成本地模型"""
    local_basic_model = nn.model.SequentialModel.load(local_basic_model_name)
    model = train_local_lstm(local_basic_model, single_basin_data, epoch, batch_size)
    predict_y = model.predict(test_x)
    predict_y = single_ds_test.local_rescale(predict_y, variable='output')
    nse, rmse, mae = (test_y, predict_y)
    model.save('/home/FedHydro/dataset/code_test/fedsrp/models/local_BLSTM'+str(basin_id)+'_2years.model')
    print("Local_BLSTM, basin:", basin_id, ",date:", test_date, ",nse:", nse)
    return model




def mk_pic_test(basin_id, the_date_range, model_path):
    single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin_id, the_date_range)
    fed_srp_model_path = model_path
    fed_srp_model = nn.model.SequentialModel.load(fed_srp_model_path)
    fed_srp_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=3, batch_size=64)
    model = fed_srp_model
    predict_y = model.predict(val_x)
    predict_y = ds_val.local_rescale(predict_y, variable='output')
    nse, rmse, mae = (val_y, predict_y)
    print('basin:', basin_id)
    print(len(val_y))
    print('nse:', nse)
    print('rmse:', rmse)
    print('mae:', mae)
    return nse, rmse, mae


def generate_compare_model():
    """将FedSRP模型和对比方法进行对比"""
    basin_ids = ['02479155']
    the_date_range = [date_range1]
    i = 0
    epoch = 50
    fune_epoch = 5
    batch_size = 64
    for basin_id in basin_ids:
        # "在chapter4_exp2.py中的train_global_model_with_sparse函数生成全局模型"
        global_model_name = '/home/FedHydro/dataset/code_test/fedsrp/models/global_model_8basins_50epoch_2years.model'
        # global_model_name = './model/global_model_6basins_70epoch_2years.model'
        single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin_id, the_date_range[i])

        """Local-BLSTM"""
        local_lstm_model = local_lstm(basin_id, the_date_range[i], local_basic_model_name, single_basin_data, val_x, val_y, ds_val,
                   epoch, batch_size)

        """mutil-fune"""
        mutil_fune_model =mutil_fune(global_model_name, basin_id, single_basin_data, fune_epoch, batch_size, the_date_range[i])

        """Transfer-A"""
        tl_a_model = transfer_a(global_model_name, basin_id, single_basin_data, fune_epoch, batch_size, val_x, val_y, ds_val)

        """Transfer-B"""
        tl_b_model = transfer_b(global_model_name, basin_id, single_basin_data, fune_epoch, batch_size, val_x, val_y, ds_val)
        i += 1

        """FedFRP，之前称为fed_hydro，为了与FedSRP区分，改为FedFRP"""
        # fed_model_name = './model/MODEL-fed_hydro_6basins_t30_210-N(0).model'
        # fed_hydro_model = fed_hydro(fed_model_name, single_basin_data, fune_epoch, batch_size, basin_id, val_x, val_y, ds_val)

        """fed_srp"""
        # fed_srp_model_path = './model/fed_srp/fedsrp_0414/MODEL-fed_srp' + str(basin_id) + '-N(7).model'
        fed_srp_model_path = '/home/FedHydro/dataset/code_test/fedsrp/models/MODEL-fed_srp03z-N(7).model'
        fed_srp_model = nn.model.SequentialModel.load(fed_srp_model_path)
        fed_srp_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=3, batch_size=64)
        """eval_model"""
        model_list = [local_lstm_model, mutil_fune_model, tl_a_model, tl_b_model, fed_srp_model]
        nse_list = []
        rmse_list = []
        mae_list = []
        for model in model_list:
            predict_y = model.predict(val_x)
            predict_y = ds_val.local_rescale(predict_y, variable='output')
            nse, rmse, mae = (val_y, predict_y)
            nse_list.append(nse)
            rmse_list.append(rmse)
            mae_list.append(mae)
            # np.savetxt()
        print("======================================compare result:", basin_id, "========================================")
        print(nse_list)
        print(rmse_list)
        print(mae_list)


def generate_compare_model_S2():
    """废弃"""
    """S2测试集"""
    basin_ids = ['01047000', '01054200', '01055000']
    the_date_range = [date_range1, date_range2, date_range3]
    i = 0
    epoch = 70
    fune_epoch = 3
    batch_size = 64
    for basin_id in basin_ids:
        global_model_name = './model/global_model_0413/global_model_8basins_70epoch_2years'+str(basin_id)+'.model'
        # global_model_name = './model/global_model_6basins_70epoch_2years.model'
        single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin_id, the_date_range[i])

        """Local-BLSTM"""
        local_lstm_model = local_lstm(basin_id, the_date_range[i], local_basic_model_name, single_basin_data, test_x, test_y, ds_test,
                   epoch, batch_size)

        """mutil-fune"""
        mutil_fune_model =mutil_fune(global_model_name, basin_id, single_basin_data, fune_epoch, batch_size, the_date_range[i])

        """Transfer-A"""
        tl_a_model = transfer_a(global_model_name, basin_id, single_basin_data, fune_epoch, batch_size, test_x, test_y, ds_test)

        """Transfer-B"""
        tl_b_model = transfer_b(global_model_name, basin_id, single_basin_data, fune_epoch, batch_size, test_x, test_y, ds_test)
        i += 1

        """fed_hydro"""
        fed_model_name = './model/MODEL-fed_hydro_6basins_t30_210-N(0).model'
        fed_hydro_model = fed_hydro(fed_model_name, single_basin_data, fune_epoch, batch_size, basin_id, test_x, test_y, ds_test)

        """fed_srp"""
        fed_srp_model_path = './model/fed_srp/fedsrp_8basins_spilt_rate0.7_client8_70epoch_' + str(basin_id) + '.model'
        fed_srp_model = nn.model.SequentialModel.load(fed_srp_model_path)
        """eval_model"""
        model_list = [local_lstm_model, mutil_fune_model, tl_a_model, tl_b_model, fed_hydro_model, fed_srp_model]
        nse_list = []
        rmse_list = []
        mae_list = []
        for model in model_list:
            predict_y = model.predict(test_x)
            predict_y = ds_test.local_rescale(predict_y, variable='output')
            nse, rmse, mae = (test_y, predict_y)
            nse_list.append(nse)
            rmse_list.append(rmse)
            mae_list.append(mae)
        print("======================================compare result:", basin_id, "========================================")
        print(nse_list)
        print(rmse_list)
        print(mae_list)


def fed_srp_model_test():
    """用于测试FedSRP在单个数据稀缺流域上的预测性能"""
    basin_ids = ['01047000', '01054200', '01055000']
    the_date_range = [date_range1, date_range2, date_range3]
    i = 0
    for basin in basin_ids:
        fed_fomo_path = './model/MODEL-fed_srp_epoch100_' + str(basin_id) + '-N(7).model'
        single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin, the_date_range[i])
        model = nn.model.SequentialModel.load(fed_fomo_path)
        predict_y = model.predict(test_x)
        predict_y = ds_test.local_rescale(predict_y, variable='output')
        nse, rmse, mae = (test_y, predict_y)
        print("basin:", basin, "nse:", nse)


def fed_srp_model_test2():
    basin_ids = ['01047000']
    # basin_ids = ['01047000', '01054200', '01055000']
    # the_date_range = [date_range1, date_range2, date_range3]
    the_date_range = [date_range1]
    i = 0
    for basin in basin_ids:
        fed_fomo_path = './model/MODEL-fed_srp-N(7).model'
        single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin, the_date_range[i])
        model = nn.model.SequentialModel.load(fed_fomo_path)
        model.compile(nn.gradient_descent.ADAMOptimizer(alpha=0.001))
        model.fit(x=single_basin_data[0], label=single_basin_data[1], batch_size=64, epoch=3)
        predict_y = model.predict(test_x)
        predict_y = ds_test.local_rescale(predict_y, variable='output')
        nse, rmse, mae = (test_y, predict_y)
        print("basin:", basin, "nse:", nse)


def sparse_basin_date_test():
    basin_ids = ['01047000', '01054200', '01055000']
    test_date_lsit = correct_date_range_test()
    epoch = 70
    batch_size = 64
    for test_date_range in test_date_lsit:
        for basin in basin_ids:
            the_date_range = get_date_range(test_date_range)
            train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_sparse_data(basin, the_date_range)
            single_basin_data = []
            single_basin_data.append(train_x)
            single_basin_data.append(train_y)
            print("date:", test_date_range)
            local_lstm(basin, test_date_range, local_basic_model_name, single_basin_data, val_x, val_y, ds_val, epoch, batch_size)


# def fedsrp():
#     """废弃"""
#     """7个数据丰富流域， 1个数据稀缺流域"""
#     rich_basin_ids1 = ['01013500', '01030500', '01031500', '01052500', '01055000', '01057000', '01054200']  # 01047000
#     rich_basin_ids2 = ['01013500', '01030500', '01031500', '01052500', '01055000', '01057000', '01047000']  # 01054200
#     rich_basin_ids3 = ['01013500', '01030500', '01031500', '01052500', '01057000', '01047000', '01054200']  # 01055000
#     rich_basin_id_list = []
#     rich_basin_id_list.append(rich_basin_ids1)
#     rich_basin_id_list.append(rich_basin_ids2)
#     rich_basin_id_list.append(rich_basin_ids3)
#
#     sparse_basin_ids1 = ['01047000']
#     sparse_basin_ids2 = ['01054200']
#     sparse_basin_ids3 = ['01055000']
#     sparse_basin_ids_list = []
#     sparse_basin_ids_list.append(sparse_basin_ids1)
#     sparse_basin_ids_list.append(sparse_basin_ids2)
#     sparse_basin_ids_list.append(sparse_basin_ids3)
#     # rich_date_range = rich_date_range
#     sparse_date_range = []
#     for i in range(len(rich_basin_id_list)):
#         if i == 0:
#             sparse_date_range.append(date_range1)
#         elif i == 1:
#             sparse_date_range.append(date_range2)
#         else:
#             sparse_date_range.append(date_range3)
#         rich_basin_ids = rich_basin_id_list[i]
#         sparse_basin_ids = sparse_basin_ids_list[i]
#         train_data_list, test_data_list, ds_test_list = get_merge_data(rich_basin_ids, sparse_basin_ids,
#                                                                        rich_date_range, sparse_date_range)
#         sparse_date_range = []
#         fedsrp_test(rich_basin_ids, sparse_basin_ids, train_data_list, test_data_list, ds_test_list)


def cal_improvement_rate():
    nse1 = [0.42880656875694767, 0.6944220306944396, 0.699552594765654, 0.6679974149387493, 0.685382037160466, 0.7840945656230599]
    nse2 = [0.5377958448125016, 0.6062751244407893, 0.5953955944945282, 0.568731843587551, 0.5509495362565724, 0.6578815325430056]
    nse3 = [0.43719541933217865, 0.6033053560607651, 0.5943046177297746, 0.5704297203090295, 0.5553835543162655, 0.6618573245112089]
    improve1 = [(x - nse1[0])/nse1[0]*100 for x in nse1[1:]]
    improve2 = [(x - nse2[0])/nse2[0]*100 for x in nse2[1:]]
    improve3 = [(x - nse3[0])/nse3[0]*100 for x in nse3[1:]]

    improve1 = np.asarray(improve1)
    improve2 = np.asarray(improve2)
    improve3 = np.asarray(improve3)
    print(improve1)
    print(improve2)
    print(improve3)
    improve = improve1 + improve2 + improve3
    print(improve)
    print(improve.shape)
    print("结果1， mean:", improve/3)

    # 结果2
    pos = len(nse1)-1
    improve1_list = []
    improve2_list = []
    improve3_list = []
    for i in range(1, len(nse1)-1):
        improve1 = (nse1[pos] - nse1[i])/nse1[i]*100
        improve2 = (nse2[pos] - nse2[i])/nse2[i]*100
        improve3 = (nse3[pos] - nse3[i])/nse3[i]*100
        improve1_list.append(improve1)
        improve2_list.append(improve2)
        improve3_list.append(improve3)
    print("FedSRP对比其他方法对比结果:")
    print(improve1_list)
    print(improve2_list)
    print(improve3_list)
    improve1_list = np.asarray(improve1_list)
    improve2_list = np.asarray(improve2_list)
    improve3_list = np.asarray(improve3_list)
    improve = improve1_list + improve2_list + improve3_list
    print(improve/3)


def cal_improvement_rate2(index1, index2, index3):
    # 结果2
    pos = len(index1)-1
    improve1_list = []
    improve2_list = []
    improve3_list = []
    for i in range(1, len(index1)-1):
        improve1 = -(index1[pos] - index1[i])/index1[pos]*100
        improve2 = -(index2[pos] - index2[i])/index2[pos]*100
        improve3 = -(index3[pos] - index3[i])/index3[pos]*100
        improve1_list.append(improve1)
        improve2_list.append(improve2)
        improve3_list.append(improve3)
    print("FedSRP对比其他方法对比结果:")
    print(improve1_list)
    print(improve2_list)
    print(improve3_list)
    improve1_list = np.asarray(improve1_list)
    improve2_list = np.asarray(improve2_list)
    improve3_list = np.asarray(improve3_list)
    improve = improve1_list + improve2_list + improve3_list
    print(improve/3)
# def eval_model():
#     basin_ids = ["01047000", "01054200", "01055000"]
#     the_date_range = [date_range1, date_range2]
#     flag = [1, 2]
#     i = 0
#     for basin in basin_ids:
#         single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin, the_date_range[i])
#         # eval_model_2years(basin, flag[i], val_x, val_y, ds_val)
#         eval_model_2years2(basin, flag[i], val_x, val_y, ds_val)
#         print("下面是相似度大的测试集结果：====================================")
#         eval_model_2years2(basin, flag[i], test_x, test_y, ds_test)
#         i += 1

def compare_model(basin_id, the_date_range):
    """将FedSRP模型和对比方法进行对比"""

    single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin_id, the_date_range)

    """Local-BLSTM"""
    local_lstm_model_path = '/home/FedHydro/dataset/code_test/fedsrp/models/local_BLSTM02479155_2years_1.model'
    local_lstm_model = nn.model.SequentialModel.load(local_lstm_model_path)
    local_lstm_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=5, batch_size=64)

    """mutil-fune"""
    mutil_fune_model_path = '/home/FedHydro/dataset/code_test/fedsrp/models/mutil_fune02479155_2years.model'
    mutil_fune_model = nn.model.SequentialModel.load(mutil_fune_model_path)
    mutil_fune_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=5, batch_size=64)

    """Transfer-A"""
    tl_a_model_path = '/home/FedHydro/dataset/code_test/fedsrp/models/tl_a_basin02479155_2years.model'
    tl_a_model = nn.model.SequentialModel.load(tl_a_model_path)
    tl_a_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=5, batch_size=64)

    """Transfer-B"""
    tl_b_model_path = '/home/FedHydro/dataset/code_test/fedsrp/models/tl_b_basin02479155_2years.model'
    tl_b_model = nn.model.SequentialModel.load(tl_b_model_path)
    tl_b_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=5, batch_size=64)

    """fed_srp"""
    fed_srp_model_path = '/home/FedHydro/dataset/code_test/fedsrp/models/MODEL-fed_srp03-N(7).model'
    fed_srp_model = nn.model.SequentialModel.load(fed_srp_model_path)
    fed_srp_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=5, batch_size=64)

    """eval_model"""
    model_list = [local_lstm_model, mutil_fune_model, tl_a_model, tl_b_model, fed_srp_model]
    nse_list = []
    rmse_list = []
    mae_list = []
    for model in model_list:
        predict_y = model.predict(val_x)
        predict_y = ds_val.local_rescale(predict_y, variable='output')
        nse, rmse, mae = (val_y, predict_y)
        nse_list.append(nse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        # np.savetxt()
    print("======================================compare result:", basin_id, "========================================")
    print(nse_list)
    print(rmse_list)
    print(mae_list)

def mk_pic(basin_id, the_date_range, model_path):
    single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin_id, the_date_range)
    fed_srp_model_path = model_path
    fed_srp_model = nn.model.SequentialModel.load(fed_srp_model_path)
    fed_srp_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=5, batch_size=64)
    model = fed_srp_model
    predict_y = model.predict(val_x)
    print(type(predict_y))
    predict_y = ds_val.local_rescale(predict_y, variable='output')
    nse, rmse, mae = (val_y, predict_y)
    print('basin:', basin_id)
    print('nse:', nse)
    width = 1
    plt.plot(val_y, color=(255/255,0/255,0/255), linewidth=width)# 红色是observation
    plt.plot(predict_y, color=(0/255,111/255,192/255), linewidth=width)# 蓝色是prediction
    plt.grid(linewidth=0.5)
    plt.gca().set_xticklabels([''] * len(plt.gca().get_xticks()))
    plt.gca().set_yticklabels([''] * len(plt.gca().get_yticks()))
    plt.savefig('/home/FedHydro/dataset/code_test/fedsrp/nnn_' + basin_id + '.pdf', dpi=300)
    print('basin:' + basin_id + 'is finished')
    

def generalization(basin_id, the_date_range):
    single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin_id, the_date_range)

    """Local-BLSTM"""
    local_lstm_model_path = '/home/FedHydro/dataset/code_test/fedsrp/models/local_BLSTM02479155_2years_1.model'
    local_lstm_model = nn.model.SequentialModel.load(local_lstm_model_path)
    local_lstm_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=5, batch_size=64)

    """mutil-fune"""
    mutil_fune_model_path = '/home/FedHydro/dataset/code_test/fedsrp/models/mutil_fune02479155_2years.model'
    mutil_fune_model = nn.model.SequentialModel.load(mutil_fune_model_path)
    mutil_fune_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=5, batch_size=64)

    """Transfer-A"""
    tl_a_model_path = '/home/FedHydro/dataset/code_test/fedsrp/models/tl_a_basin02479155_2years.model'
    tl_a_model = nn.model.SequentialModel.load(tl_a_model_path)
    tl_a_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=5, batch_size=64)

    """Transfer-B"""
    tl_b_model_path = '/home/FedHydro/dataset/code_test/fedsrp/models/tl_b_basin02479155_2years.model'
    tl_b_model = nn.model.SequentialModel.load(tl_b_model_path)
    tl_b_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=5, batch_size=64)

    """fed_srp"""
    fed_srp_model_path = '/home/FedHydro/dataset/code_test/fedsrp/models/MODEL-fed_srp03-N(7).model'
    fed_srp_model = nn.model.SequentialModel.load(fed_srp_model_path)
    fed_srp_model.fit(x=single_basin_data[0], label=single_basin_data[1], epoch=5, batch_size=64)

    """eval_model"""
    model_list = [local_lstm_model, mutil_fune_model, tl_a_model, tl_b_model, fed_srp_model]
    nse_list = []
    rmse_list = []
    mae_list = []
    for model in model_list:
        predict_y = model.predict(val_x)
        predict_y = ds_val.local_rescale(predict_y, variable='output')
        nse, rmse, mae = (val_y, predict_y)
        nse_list.append(nse)
        rmse_list.append(rmse)
        mae_list.append(mae)
        # np.savetxt()
    print("======================================compare result:", basin_id, "========================================")
    print(nse_list)
    print(rmse_list)
    print(mae_list)



if __name__ == '__main__':

    # generate_compare_model()

    # sparse_basin_date_test()

    """生成FedSRP及其对比模型，及对比结果"""
    """"调用此方法前，需要先生成以下模型
    1、在chapter4_exp2.py中的train_global_model_with_sparse函数生成全局模型
    2、FedFRP模型
    3、FedSRP模型
    然后将生成的模型放入fedsrp/model文件夹下"""
    # generate_compare_model()
    # generate_compare_model_S2()
    # fed_srp_model_test()

    # 计算NSE提高比率
    # cal_improvement_rate()

    # 计算rmse, mae降低比例
    # rmse1 = [1.7714547745329172, 1.2956851678362848, 1.284762038617284, 1.3505454361234568, 1.3147107812109056, 1.089106258976917]
    # rmse2 = [3.3448626234892296, 3.087150414948174, 3.1295123266331095, 3.2309858722116767, 3.29692401143775, 2.8777272007700376]
    # rmse3 = [3.223357685582805, 2.706187334249237, 2.736715934863723, 2.816091760992707, 2.8649856429406735, 2.498501882475088]
    # # cal_improvement_rate2(rmse1, rmse2, rmse3)
    #
    # mae1 = [0.7314426486873685, 0.5863496881795864, 0.5955581233963576, 0.6498322393202761, 0.5174121835580218, 0.49873475525332334]
    # mae2 = [1.5611482316940521, 1.4075683964989523, 1.42474841915122, 1.4243162055765788, 1.5830128432426982, 1.3364599210812476]
    # mae3 = [1.5618057076046716, 1.175705598765456, 1.181536202744692, 1.2062192686409605, 1.3405968676342832, 1.1850863914180942]
    # cal_improvement_rate2(mae1, mae2, mae3)


    basin_id = '02465493'
    used_03 = ['02046000', '02051000', '02051500', '02053200', '02053800', '02055100', '02059500', '02064000', '02065500', '02069700', '02070000', '02297310', '02111500', '02108000', '02216180', '02231000', '02212600', '02215100', '02216180', '02231000', '02231342', '02235200', '02245500', '02246000', '02296500', '02297155', '02297310', '02464360']
    basin03 = '02479155'
    basin01 = '01169000'
    basin18 = '11151300'
    basin17 = '13331500'

    the_date_range = {
        'train_date': {
            'start_date': pd.to_datetime("1981-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1983-09-30", format="%Y-%m-%d")
        },
        'val_date': {
            'start_date' : pd.to_datetime("2008-10-01", format="%Y-%m-%d"),
            'end_date' : pd.to_datetime("2009-09-30", format="%Y-%m-%d")
        },
        'test_date': {
            'start_date': pd.to_datetime("2004-10-01", format="%Y-%m-%d"),# 没用
            'end_date': pd.to_datetime("2006-09-30", format="%Y-%m-%d")
        },
    }

    #generalization(basin_id, the_date_range)
    #compare_model(basin01, the_date_range)
    mk_pic(basin17, the_date_range, '/home/FedHydro/dataset/code_test/fedsrp/models/MODEL-fed_srp17-N(7).model')
    
    #     # srp模型的地址
    #     model_path = '/home/FedHydro/Node-7-Retrieve/MODEL-fed_srp03z-N(7).model'
    #     nse, rmse, mae = mk_pic_test(basin_id, the_date_range, model_path)
    #     nse_list.append(nse)
    #     rmse_list.append(rmse)
    #     mae_list.append(mae)

    # print("--------------------最终结果----------------------")
    # print(nse_list)
    # print(rmse_list)
    # print(mae_list)
