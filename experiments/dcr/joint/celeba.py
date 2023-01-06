from torch.nn import BCELoss
import pytorch_lightning as pl
import torch
import os
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dcr.data.celeba import load_celeba
from dcr.models import DeepConceptReasoner


def main():
    epochs = 100
    learning_rate = 0.008
    emb_size = 16
    batch_size = 32
    limit_batches = 1.0

    train_data, test_data, in_concepts, out_concepts, concept_names, class_names = load_celeba()
    train_dl = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)

    for fold in range(5):
        results_dir = f"./results/celeba/{fold}/"
        os.makedirs(results_dir, exist_ok=True)
        model_path = os.path.join(results_dir, 'model.pt')

        model = DeepConceptReasoner(in_concepts=in_concepts, out_concepts=out_concepts, emb_size=emb_size,
                                    concept_names=concept_names, class_names=class_names,
                                    learning_rate=learning_rate, loss_form=BCELoss(),
                                    reasoner=True, temperature=1)
        if not os.path.exists(model_path):
            print(f'Running epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}')
            logger = TensorBoardLogger(save_dir=results_dir, name="lightning_logs")
            trainer = pl.Trainer(max_epochs=epochs,
                                 # precision=16,
                                 # check_val_every_n_epoch=5,
                                 # accumulate_grad_batches=4,
                                 devices=1, accelerator="gpu",
                                 limit_train_batches=limit_batches,
                                 limit_val_batches=limit_batches,
                                 # callbacks=[checkpoint_callback],
                                 logger=logger)
            trainer.fit(model=model, train_dataloaders=train_dl)
            torch.save(model.state_dict(), model_path)

        model.load_state_dict(torch.load(model_path))
        trainer = pl.Trainer(devices=1, accelerator="gpu",
                             # precision=16,
                             limit_test_batches=limit_batches)
        trainer.test(model, dataloaders=test_dl)


if __name__ == '__main__':
    main()
