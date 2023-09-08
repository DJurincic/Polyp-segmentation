import segmentation_models_pytorch as smp

BEST_MODEL_PATH = './best_model.pth'

def train(model, loss, metrics, goal_metric, goal_comparison, optimizer, device, verbose, n_epochs, train_dataloader, valid_dataloader, lr_schedule):
    """
    A function for training models.

    Parameters
    ----------

    model : trainable model object
    loss : loss object
    metrics : list
            list of prediction metrics
    goal_metric : str
                name of the metric which will be used to determine the best model
    goal_comparison : callable
                callable object which recives the current best score and the current score and returns True if the current score is better 
    optimizer : optimizer object
    device : str
            'cuda' or 'cpu'
    verbose : bool
            option to get more detailed output while training
    n_epochs : int
            number of training epochs
    train_dataloader : DataLoader
                    loader of training data
    valid_dataloader : DataLoader
                    loader of validation data
    lr_schedule : list
                list object in which the first element is a boolean value which determines if the learning rate should change
                and the second object determines the number of epochs after which the learning rate is multiplied by 0.1
    """

    ## Initialize the train and valid epochs
    #--------------------------------------------------------------------------

    train_epoch = smp.utils.train.TrainEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=verbose
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=verbose
    )

    #--------------------------------------------------------------------------

    best_score = 0

    train_goal_metrics = []
    valid_goal_metrics = []

    # Run the training and validation over n_epoch epochs
    #--------------------------------------------------------------------------

    for epoch in range(n_epochs):

        print(f'Epoch: {epoch}')

        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)

        # Keep track of the goal_metric for visualization
        #-------------------------------------------------

        train_goal_metrics.append(train_logs[goal_metric])
        valid_goal_metrics.append(valid_logs[goal_metric])

        #-------------------------------------------------

        current_score = valid_logs[goal_metric]

        if epoch == 0:
            best_score = current_score
        
        # Save model if it is the best so far
        #--------------------------------------------------

        else:
            if goal_comparison(best_score, current_score):
                best_score = current_score
                torch.save(model, BEST_MODEL_PATH)
                print('New best model saved!')

        #-------------------------------------------------

        # Adjust learning rate if necessary
        #---------------------------------------------------------------------
        
        if lr_schedule[0]:
            scheduled_epochs = lr_schedule[1]

            if epoch % scheduled_epochs == 0 and epoch != 0:

                current_lr = optimizer.param_group[0]['lr']
                new_lr = current_lr*0.1
                optimizer.param_group[0]['lr'] = new_lr
                print(f'Learning rate lowered to {new_lr} on epoch {epoch}!')
        
        #----------------------------------------------------------------------

    #--------------------------------------------------------------------------

    # Visualize the goal metric over all epochs
    #---------------------------------------------

    plt.plot(train_goal_metrics, label = 'Train')
    plt.plot(valid_goal_metrics, label = 'Valid')
    plt.xlabel('Epoch')
    plt.ylabel(goal_metric)
    plt.legend()
    plt.show()

    #---------------------------------------------