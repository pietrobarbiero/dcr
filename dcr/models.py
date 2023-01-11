import torchvision
import torch
import pytorch_lightning as pl

from .nn import ConceptReasoningLayer, ConceptEmbedding
from .semantics import Logic, GodelTNorm


class DeepConceptReasoner(pl.LightningModule):
    def __init__(self, in_concepts, out_concepts, emb_size, concept_names, class_names,
                 learning_rate, loss_form, concept_loss_weight: float = 1., class_loss_weight: float = 1.,
                 temperature_pos: float = 1., temperature_neg: float = 1., reasoner: bool = True,
                 logic: Logic = GodelTNorm()):
        super().__init__()
        self.concept_loss_weight = concept_loss_weight
        self.class_loss_weight = class_loss_weight
        self.resnet = torchvision.models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        n_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_features),
            torch.nn.LeakyReLU(),
        )
        self.concept_embedder = ConceptEmbedding(n_features, in_concepts, emb_size)
        self.reasoner = reasoner
        if self.reasoner:
            self.predictor = ConceptReasoningLayer(emb_size, out_concepts, logic, temperature_pos, temperature_neg)
        else:
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(in_concepts, 10),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(10, out_concepts),
                torch.nn.Sigmoid()
            )
        self.concept_names = concept_names
        self.class_names = class_names
        self.learning_rate = learning_rate
        self.loss_form = loss_form

    def forward(self, x):
        h = self.resnet.forward(x)
        c_emb, c_pred = self.concept_embedder(h)
        if self.reasoner:
            y_pred = self.predictor(c_emb, c_pred)
        else:
            y_pred = self.predictor(c_pred)
        return c_pred, y_pred

    def training_step(self, batch, batch_idx):
        x, t = batch
        c, y = t
        c_pred, y_pred = self.forward(x)
        concept_loss = self.loss_form(c_pred, c.float())
        class_loss = self.loss_form(y_pred, y.float())
        loss = self.concept_loss_weight * concept_loss + self.class_loss_weight * class_loss
        self.log("train_loss", loss.item())
        self.log("concept_loss", concept_loss.item())
        self.log("class_loss", class_loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        c, y = t[0], t[1]
        c_pred, y_pred = self.forward(x)
        concept_loss = self.loss_form(c_pred, c.float())
        class_loss = self.loss_form(y_pred, y.float())
        loss = self.concept_loss_weight * concept_loss + self.class_loss_weight * class_loss
        self.log("test_loss", loss.item())
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, t = batch
        return self.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)    # TODO: consider whether we need this or not
        return optimizer
