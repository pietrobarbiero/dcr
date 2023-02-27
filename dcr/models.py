import torchvision
import torch
import pytorch_lightning as pl

from .nn import ConceptReasoningLayer, ConceptEmbedding
from .semantics import Logic, GodelTNorm
from cem.models.cbm import ConceptBottleneckModel

class CemEmbedder(torch.nn.Module):
    def __init__(
        self,
        cem,
        n_concepts,
        freeze_model=True,
        soft_concept_labels=True,
        pass_pos_neg_embs=False,
        intervention_idxs=None,
        training_intervention_prob=0.25,
        test_with_soft_labels=True,
    ):
        super().__init__()
        self.cem = cem
        self.cem.training_intervention_prob = training_intervention_prob
        self.n_concepts = n_concepts
        self.soft_concept_labels = soft_concept_labels
        self.pass_pos_neg_embs = pass_pos_neg_embs
        self.intervention_idxs = intervention_idxs
        self.test_with_soft_labels = test_with_soft_labels
        self.freeze_model = freeze_model
        if freeze_model:
            for param in self.cem.parameters():
                param.requires_grad = False

    def forward(self, x, intervention_idxs=None, c=None, train=False):
        if self.pass_pos_neg_embs and intervention_idxs is None:
            # Then we will force the CEM to "intervene" in all concepts
            # so that we get their corresponding positive or negative
            # embeddings based on their predicted ground truth labels
            intervention_idxs = list(range(self.n_concepts))
        c_pred, c_emb, _ = self.cem._forward(
            x=x,
            intervention_idxs=intervention_idxs,
            c=c,
            train=((not self.freeze_model) and train),
        )
        c_used = c_pred
        if (c is not None) and (not train):
            if (not self.soft_concept_labels):
                # Then we simply use the ground truth concept labels rather than
                # their soft representations
                c_used = c
            elif self.intervention_idxs:
                # Then we propagate the intervention change through the predicted
                # probability
                c_used = c_pred.copy()
                c_used[:, intervention_idxs] = c[:, intervention_idxs]
        elif (c is not None) and (not self.test_with_soft_labels):
            c_used = c
        c_emb = torch.reshape(
            c_emb,
            (c_emb.shape[0], self.n_concepts, c_emb.shape[-1]//self.n_concepts)
        )
        return c_emb, c_pred, c_used

class DefaultConceptEmbedder(torch.nn.Module):
    def __init__(
        self,
        n_concepts,
        c_extractor_arch,
        emb_size,
        intervention_idxs=None,
        training_intervention_prob=0.25,
        freeze_pretrain=False,
        pretrain_model=True,
    ):
        super().__init__()
        self.intervention_idxs = intervention_idxs
        self.training_intervention_prob = training_intervention_prob
        
        # Else we assume that it is a callable function which we will
        # need to instantiate here
        try:
            self.latent_code_gen = c_extractor_arch(pretrained=pretrain_model)
            n_features = self.latent_code_gen.fc.in_features
            if freeze_pretrain:
                for param in self.latent_code_gen.parameters():
                    param.requires_grad = False
            self.latent_code_gen.fc = torch.nn.Sequential(
                torch.nn.Linear(n_features, n_features),
                torch.nn.LeakyReLU(),
            )
        except Exception as e:
            n_features = n_concepts
            self.latent_code_gen = c_extractor_arch(output_dim=n_concepts)

        self.concept_embedder = ConceptEmbedding(
            in_features=n_features,
            n_concepts=n_concepts,
            emb_size=emb_size,
            intervention_idxs=intervention_idxs,
            training_intervention_prob=training_intervention_prob,
        )

    def forward(self, x, intervention_idxs=None, c=None, train=False):
        h = self.latent_code_gen.forward(x)
        if intervention_idxs is not None:
            # This takes precedence over the model construction parameter
            c_emb, c_pred = self.concept_embedder(
                h,
                intervention_idxs=intervention_idxs,
                c=c,
                train=train,
            )
        elif self.intervention_idxs is not None:
            c_emb, c_pred = self.concept_embedder(
                h,
                intervention_idxs=self.intervention_idxs,
                c=c,
                train=train,
            )
        else:
            c_emb, c_pred = self.concept_embedder(
                h,
                intervention_idxs=intervention_idxs,
                c=c,
                train=train,
            )
        return c_emb, c_pred


class DeepConceptReasoner(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        emb_size,
        concept_names,
        class_names,
        concept_loss_weight=1,
        task_loss_weight=1,
        temperature=1,
        reasoner=True,
        logic=GodelTNorm(),
        intervention_idxs=None,
        training_intervention_prob=0.25,
        concept_embedder=None,
        c_extractor_arch=torchvision.models.resnet18,
        per_class_models=False,

        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        weight_loss=None,
        normalize_loss=False,
        optimizer="adam",
        top_k_accuracy=2,
        
        **extra_params,
    ):
        pl.LightningModule.__init__(self)
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight
        self.intervention_idxs = intervention_idxs
        if concept_embedder is None:
            self.concept_embedder = DefaultConceptEmbedder(
                n_concepts=n_concepts,
                emb_size=emb_size,
                intervention_idxs=intervention_idxs,
                c_extractor_arch=c_extractor_arch,
                training_intervention_prob=training_intervention_prob,
            )
        else:
            self.concept_embedder = concept_embedder
            
        self.reasoner = reasoner
        if self.reasoner:
            self.predictor = ConceptReasoningLayer(
                emb_size,
                n_tasks,
                logic,
                temperature,
                per_class_models=per_class_models,
            )
        else:
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(n_concepts, 10),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(10, n_tasks),
                torch.nn.Sigmoid()
            )
        self.concept_names = concept_names or [
            "concept_{i}" for i in range(0, n_concepts)
        ]
        self.class_names = class_names or [
            "class_{i}" for i in range(0, n_tasks)
        ]
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.normalize_loss = normalize_loss
        self.optimizer_name = optimizer
        self.top_k_accuracy = top_k_accuracy
        self.n_tasks = n_tasks
        
        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.loss_task = (
            torch.nn.CrossEntropyLoss()
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss()
        )

    def _forward(self, x, intervention_idxs=None, c=None, train=False):
        if (self.intervention_idxs is not None) and (intervention_idxs is None):
            intervention_idxs = self.intervention_idxs
        embs_outputs = self.concept_embedder(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            train=train,
        )
        if len(embs_outputs) == 2:
            c_emb, c_sem = embs_outputs
            c_used = c_sem
        else:
            c_emb, c_sem, c_used = embs_outputs
        if self.reasoner:
            y_pred = self.predictor(c_emb, c_used)
        else:
            y_pred = self.predictor(c_sem)
        return c_sem, c_emb, y_pred