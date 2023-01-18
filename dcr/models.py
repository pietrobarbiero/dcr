import torchvision
import torch
import pytorch_lightning as pl

from .nn import ConceptReasoningLayer, ConceptEmbedding
from .semantics import Logic, GodelTNorm


class DeepConceptReasoner(pl.LightningModule):
    def __init__(
        self,
        in_concepts,
        out_concepts,
        emb_size,
        concept_names,
        class_names,
        learning_rate,
        loss_form,
        loss_form_concept=None,
        concept_loss_weight: float = 1.,
        class_loss_weight: float = 1.,
        temperature: float = 1.,
        reasoner: bool = True,
        logic: Logic = GodelTNorm(),
        intervention_idxs=None,
        training_intervention_prob=0.25,
    ):
        super().__init__()
        self.concept_loss_weight = concept_loss_weight
        self.class_loss_weight = class_loss_weight
        self.resnet = torchvision.models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        n_features = self.resnet.fc.in_features
        self.intervention_idxs = intervention_idxs
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(n_features, n_features),
            torch.nn.LeakyReLU(),
        )
        
        self.concept_embedder = ConceptEmbedding(
            in_features=n_features,
            n_concepts=in_concepts,
            emb_size=emb_size,
            intervention_idxs=self.intervention_idxs,
            training_intervention_prob=training_intervention_prob,
        )
        self.reasoner = reasoner
        if self.reasoner:
            self.predictor = ConceptReasoningLayer(emb_size, out_concepts, logic, temperature)
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
        self.loss_form_concept = loss_form_concept or loss_form

    def _forward(self, x, intervention_idxs=None, c=None, train=False):
        h = self.resnet.forward(x)
        c_emb, c_pred = self.concept_embedder(
            h,
            intervention_idxs=intervention_idxs,
            c=c,
            train=train,
        )
        if self.reasoner:
            y_pred = self.predictor(c_emb, c_pred)
        else:
            y_pred = self.predictor(c_pred)
        return c_pred, y_pred
    
    def forward(self, x, intervention_idxs=None, c=None, train=False):
        if intervention_idxs is not None:
            # This takes precedence over the model construction parameter
            c_pred, y_pred = self._forward(
                x,
                intervention_idxs=intervention_idxs,
                c=c,
                train=train,
            )
        elif self.intervention_idxs is not None:
            c_pred, y_pred = self._forward(
                x,
                intervention_idxs=self.intervention_idxs,
                c=c,
                train=train,
            )
        else:
            c_pred, y_pred = self._forward(x, c=c, train=train)
        return c_pred, y_pred
    
    def _unpack_input(self, batch):
        if len(batch) == 2:
            x, (c, y) = batch
            intervention_idxs = None
        elif len(batch) == 3:
            # Then we assume we are also given a set of concepts to intervene on as an argument!
            x, (c, y), intervention_idxs = batch
        return x, c, y, intervention_idxs
        
    def training_step(self, batch, batch_idx):
        x, c, y, intervention_idxs = self._unpack_input(batch=batch)
        c_pred, y_pred = self.forward(x, intervention_idxs=intervention_idxs, c=c, train=True)
        concept_loss = self.loss_form_concept(c_pred, c.float())
        class_loss = self.loss_form(y_pred, y.float())
        loss = self.concept_loss_weight * concept_loss + self.class_loss_weight * class_loss
        self.log("train_loss", loss.item())
        self.log("concept_loss", concept_loss.item())
        self.log("class_loss", class_loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        x, c, y, intervention_idxs = self._unpack_input(batch=batch)
        c_pred, y_pred = self.forward(x, intervention_idxs=intervention_idxs, c=c, train=False)
        concept_loss = self.loss_form_concept(c_pred, c.float())
        class_loss = self.loss_form(y_pred, y.float())
        loss = self.concept_loss_weight * concept_loss + self.class_loss_weight * class_loss
        self.log("test_loss", loss.item())
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, c, y, intervention_idxs = self._unpack_input(batch=batch)
        return self.forward(x, intervention_idxs=intervention_idxs, c=c, train=False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)    # TODO: consider whether we need this or not
        return optimizer
