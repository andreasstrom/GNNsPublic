from nets.load_net import gnn_model 
from data.data import LoadData 
from train.train_data_my_new_task import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network

DATASET_NAME = 'SBM_PATTERN'
dataset = LoadData(DATASET_NAME)

MODEL_NAME = 'MyGNN'
model = gnn_model(MODEL_NAME, net_params)



optimizer = optim.Adam(model.parameters())
train_loader = DataLoader(trainset, batch_size=128, shuffle=True, collate_fn=dataset.collate)
epoch_train_loss, epoch_train_acc = train_epoch(model, optimizer, device, train_loader, epoch)   