
from pytorch_forecasting.metrics import SMAPE
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from torch import cat

## Evaluate performance
def evaluate(
    trainer: Trainer,
    model: LightningModule,
    test_dataloader: DataLoader
    ):
    ### Best performers

    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = model.load_from_checkpoint(best_model_path)


    # calcualte mean absolute error on validation set
    actuals = cat([y[0] for x, y in iter(test_dataloader)])
    predictions = best_tft.predict(test_dataloader)
    (actuals - predictions).abs().mean()


    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions, x = best_tft.predict(test_dataloader, mode="raw", return_x=True)

    for idx in range(10):  # plot 10 examples
        best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);


    ### Worst performers

    # calcualte metric by which to display
    predictions = best_tft.predict(test_dataloader)
    mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
    indices = mean_losses.argsort(descending=True)  # sort losses
    for idx in range(10):  # plot 10 examples
        best_tft.plot_prediction(
            x, raw_predictions,
            idx=indices[idx], 
            add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles)
        );

    print("Press Enter to close all plots and continue ...")
    input()