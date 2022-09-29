# %%

from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torch import cat

## Evaluate performance
def evaluate(
    checkpoint_path: str,
    model: LightningModule,
    test_dataloader: DataLoader
    ):
    ### Best performers

    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_tft = model.load_from_checkpoint(checkpoint_path)

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

def get_file_path(
        path: str,
        notebook: str="evaluation.py",
    ):
    import os
    notebook_path = os.path.abspath(notebook)
    parent_folder_path = '..'
    # NOTEBOOK = os.environ['NOTEBOOK']
    # if NOTEBOOK and NOTEBOOK =="false":
    #     parent_folder_path ='.'
    file_path = os.path.join(os.path.dirname(notebook_path), parent_folder_path, path)
    return file_path

def load_best_model_path(
        log_dir: str='lightning_logs'
    ):
    import json
    json_path = log_dir+'/'+'best_model_path.json'
    full_json_path = get_file_path(json_path)
    with open(full_json_path) as f:
        d = json.load(f)
        print(d)
        checkpoint_path = d.get("checkpoint_path")
    full_json_path = get_file_path(checkpoint_path)
    return full_json_path

def main():
    validation_path = get_file_path("examples/data/validation.pkl")
    # load datasets
    validation = TimeSeriesDataSet.load(validation_path)
    batch_size = 64
    num_workers = 0
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers
    )
    # find_best_model_path
    checkpoint_path = load_best_model_path()
    # evaluate model
    evaluate(
        checkpoint_path=checkpoint_path,
        model=TemporalFusionTransformer,
        test_dataloader=val_dataloader
        )
    
    print("Press Enter to close all plots and continue ...")
    # input()

main()

# %%
