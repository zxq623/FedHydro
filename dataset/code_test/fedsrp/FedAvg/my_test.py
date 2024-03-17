import torch
from models.lstm import LSTM

if __name__ == '__main__':
    # 实例化模型
    model_path='/root/best_model.pt'
    model = LSTM(input_size = 5,hidden_size = 64,num_layers = 2,output_size = 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # for param in model.parameters():
    #     print(param)
    
    # 设置输入数据
    input_data = torch.randn(3, 30, 5) # 输入数据，根据你的实际情况进行设置

    # 将输入数据转换为张量
    input_tensor = torch.Tensor(input_data)  # 调整输入数据的维度

    # 使用模型进行预测
    with torch.no_grad():
        model.eval()  # 设置模型为评估模式
        prediction = model(input_tensor)

    # 打印预测结果
    print(prediction)