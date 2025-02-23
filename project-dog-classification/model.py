 transform = transforms.Compose([
        transforms.CenterCrop((224,224)),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        
    train_data = CustomDataset('dogImages/train', transform)
    valid_data = CustomDataset('dogImages/valid', transform)
    test_data = CustomDataset('dogImages/test', transform)


    batch_size = 16
    num_workers = 5
    dataloader_train = DataLoader(train_data, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    dataloader_valid= DataLoader(valid_data, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)  
    dataloader_test = DataLoader(test_data, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)  

    # define the CNN architecture
    class Net(nn.Module):
        ### TODO: choose an architecture, and complete the class
        def __init__(self):
            super(Net, self).__init__()
            ## Define layers of a CNN 
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) # in: 3x224x224 out: 8x112x112
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # in: 8x112x112 out:16x56x56
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1) # in: 16x56x56 out: 24x28x28
            self.conv4 = nn.Conv2d(64, 128, 3, padding=1) # in: 24x28x28 out: 32x14x14
            self.conv5 = nn.Conv2d(128, 256, 3, padding=1) # in: 32x14x14 out: 64x7x7
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(in_features=256 * 7 * 7, out_features=512)
            self.fc2 = nn.Linear(512, 133)
            self.dropout = nn.Dropout(p=0.3)
        
        def forward(self, x):
            ## Define forward behavior
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool1(F.relu(self.conv2(x)))
            x = self.pool1(F.relu(self.conv3(x)))
            x = self.pool1(F.relu(self.conv4(x)))
            x = self.pool1(F.relu(self.conv5(x)))
            # flatten image input
            x = x.view(-1, 256 * 7 * 7)
            # add dropout layer
            #x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    #-#-# You do NOT have to modify the code below this line. #-#-#
    # instantiate the CNN
    model_scratch = Net()

    # move tensors to GPU if CUDA is available
    if use_cuda:
        model_scratch.cuda()

    ### TODO: select loss function
    criterion_scratch = nn.CrossEntropyLoss()

    ### TODO: select optimizer
    optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.1)

    def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
        """returns trained model"""
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf 
        print("CUDA:", use_cuda)

        for epoch in tqdm(range(1, n_epochs+1),desc='Epochs'):
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            
            ###################
            # train the model #
            ###################
            model.train()
            for batch_idx, (data, target) in enumerate(tqdm(loaders['train'], desc='Train')):
                # move to GPU
                
                data, target = data.cuda(), target.cuda()
                ## find the loss and update the model parameters accordingly
                ## record the average training loss, using something like
                ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            ######################    
            # validate the model #
            ######################
            model.eval()
            for batch_idx, (data, target) in enumerate(tqdm(loaders['valid'], desc='Valid')):
                # move to GPU
                data, target = data.cuda(), target.cuda()
                ## update the average validation loss
                output = model(data)
                loss = criterion(output, target)
                valid_loss += ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
                
            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, 
                train_loss,
                valid_loss
                ))
            
             ## TODO: save the model if validation loss has decreased
            if valid_loss < valid_loss_min:
                valid_loss_min = valid_loss
                torch.save(model.state_dict(), save_path)
            
        # return trained model
        return model

    loaders_scratch = {'train': dataloader_train,
                       'valid': dataloader_valid,
                       'test': dataloader_test}
    # train the model
    model_scratch = train(100,
                          loaders_scratch,
                          model_scratch,
                          optimizer_scratch, 
                          criterion_scratch,
                          use_cuda,
                          'model_scratch.pt')
    model_scratch.load_state_dict(torch.load('model_scratch.pt'))
