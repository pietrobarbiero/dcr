import sklearn.metrics
import torch
import pytorch_lightning as pl
import numpy as np

from cem.metrics.accs import compute_accuracy

################################################################################
## BASELINE MODEL
################################################################################


class ConceptToLabelModel(pl.LightningModule):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        c2y_model=None,
        c2y_layers=None,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=10,
        task_class_weights=None,
        output_latent=False,

        feature_drop_out=0.25,
        intervene_concept_vals=0.5,

        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        top_k_accuracy=None,
        **kwargs,
    ):
        """
        TODO
        """
        super().__init__()
        self.output_latent = output_latent
        self.n_concepts = n_concepts
        self.intervention_policy = intervention_policy
        self.output_interventions = output_interventions
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.feature_drop_out = feature_drop_out
        self.intervene_concept_vals = intervene_concept_vals

        # Now construct the label prediction model
        if c2y_model is not None:
            # Then this method has been provided to us already
            self.c2y_model = c2y_model
        else:
            # Else we construct it here directly
            units = [n_concepts] + (c2y_layers or []) + [n_tasks]
            layers = []
            for i in range(1, len(units)):
                layers.append(torch.nn.Linear(units[i-1], units[i]))
                if i != len(units) - 1:
                    layers.append(torch.nn.LeakyReLU())
            self.c2y_model = torch.nn.Sequential(*layers)

        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                pos_weight=task_class_weights
            )
        )
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.top_k_accuracy = top_k_accuracy
        self.n_tasks = n_tasks
        self.use_concept_groups = use_concept_groups

    def _unpack_batch(self, batch):
        x = batch[0]
        if isinstance(batch[1], list):
            offset = 2
            y, c = batch[1]
        else:
            offset = 3
            y, c = batch[1], batch[2]
        if len(batch) > (offset):
            g = batch[offset]
        else:
            g = None
        if len(batch) > (offset + 1):
            competencies = batch[offset + 1]
        else:
            competencies = None
        if len(batch) > (offset + 2):
            prev_interventions = batch[offset + 2]
        else:
            prev_interventions = None
        return x, y, (c, g, competencies, prev_interventions)

    def _standardize_indices(self, intervention_idxs, batch_size, device='gpu'):
        if getattr(self, 'force_all_interventions', False):
            intervention_idxs = torch.ones(
                (batch_size, self.n_concepts)
            ).to(device)
        if isinstance(intervention_idxs, list):
            intervention_idxs = np.array(intervention_idxs)
        if isinstance(intervention_idxs, np.ndarray):
            intervention_idxs = torch.IntTensor(intervention_idxs)

        if intervention_idxs is None or (
            isinstance(intervention_idxs, torch.Tensor) and
            ((len(intervention_idxs) == 0) or intervention_idxs.shape[-1] == 0)
        ):
            return None
        if not isinstance(intervention_idxs, torch.Tensor):
            raise ValueError(
                f'Unsupported intervention indices {intervention_idxs}'
            )
        if len(intervention_idxs.shape) == 1:
            # Then we will assume that we will do use the same
            # intervention indices for the entire batch!
            intervention_idxs = torch.tile(
                torch.unsqueeze(intervention_idxs, 0),
                (batch_size, 1),
            )
        elif len(intervention_idxs.shape) == 2:
            assert intervention_idxs.shape[0] == batch_size, (
                f'Expected intervention indices to have batch size {batch_size} '
                f'but got intervention indices with '
                f'shape {intervention_idxs.shape}.'
            )
        else:
            raise ValueError(
                f'Intervention indices should have 1 or 2 dimensions. Instead '
                f'we got indices with shape {intervention_idxs.shape}.'
            )
        if intervention_idxs.shape[-1] == self.n_concepts:
            # We still need to check the corner case here where all indices are
            # given...
            elems = torch.unique(intervention_idxs)
            if len(elems) == 1:
                is_binary = (0 in elems) or (1 in elems)
            elif len(elems) == 2:
                is_binary = (0 in elems) and (1 in elems)
            else:
                is_binary = False
        else:
            is_binary = False
        if not is_binary:
            # Then this is an array of indices rather than a binary array!
            intervention_idxs = intervention_idxs.to(dtype=torch.long)
            result = torch.zeros(
                (batch_size, self.n_concepts),
                dtype=torch.bool,
                device=intervention_idxs.device,
            )
            result[:, intervention_idxs] = 1
            intervention_idxs = result
        assert intervention_idxs.shape[-1] == self.n_concepts, (
                f'Unsupported intervention indices with '
                f'shape {intervention_idxs.shape}.'
            )
        if isinstance(intervention_idxs, np.ndarray):
            # Time to make it into a torch Tensor!
            intervention_idxs = torch.BoolTensor(intervention_idxs)
        intervention_idxs = intervention_idxs.to(dtype=torch.bool)
        return intervention_idxs


    def _concept_intervention(
        self,
        c_pred,
        intervention_idxs=None,
        c_true=None,
        train=False,
    ):
        if train and (self.feature_drop_out != 0) and (
            (c_true is not None) and
            (intervention_idxs is None)
        ):
            # Then we will probabilistically simulate dropping some concepts
            mask = torch.bernoulli(
                torch.ones(self.n_concepts) * (1 - self.feature_drop_out),
            )
            intervention_idxs = torch.tile(
                mask,
                (c_true.shape[0], 1),
            )

        if (c_true is None) or (intervention_idxs is None):
            return c_pred
        intervention_idxs = self._standardize_indices(
            intervention_idxs=intervention_idxs,
            batch_size=c_pred.shape[0],
            device=c_pred.device,
        )
        intervention_idxs = intervention_idxs.to(c_pred.device).to(torch.float32)
        # Let's set all the non-intervened values to 0.5
        return intervention_idxs * c_pred + (1 - intervention_idxs) * (
            self.intervene_concept_vals
        )

    def _forward(
        self,
        x,
        intervention_idxs=None,
        competencies=None,
        prev_interventions=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        output_latent=None,
        output_embeddings=False,
        output_interventions=None,
    ):
        # IMPORTANT NOTE: x here are the concepts!
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        if output_embeddings:
            pos_embeddings = torch.ones(x.shape).to(x.device)
            neg_embeddings = torch.zeros(x.shape).to(x.device)
        assert c is not None, "Expected concepts for ConceptToLabel model!"

        c_pred = c_sem = c
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=None,
            )
        else:
            c_int = c
        c_pred = self._concept_intervention(
            c_pred=c_pred,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
            train=train,
        )
        y_pred = self.c2y_model(c_pred)
        tail_results = []
        if output_interventions:
            if intervention_idxs is None:
                intervention_idxs = None
            if isinstance(intervention_idxs, np.ndarray):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings:
            tail_results.append(pos_embeddings)
            tail_results.append(neg_embeddings)
        return tuple([c_sem, c_pred, y_pred] + tail_results)


    def forward(
        self,
        x,
        c=None,
        y=None,
        latent=None,
        intervention_idxs=None,
        competencies=None,
        prev_interventions=None,
        **kwargs
    ):
        if c is None:
            # Then we will assume that x are the concepts themselves
            c = x
        return self._forward(
            x,
            train=False,
            c=c,
            y=y,
            competencies=competencies,
            prev_interventions=prev_interventions,
            intervention_idxs=intervention_idxs,
            latent=latent,
            **kwargs
        )

    def predict_step(
        self,
        batch,
        batch_idx,
        intervention_idxs=None,
        dataloader_idx=0,
    ):
        x, y, (c, g, competencies, prev_interventions) = self._unpack_batch(
            batch
        )
        return self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=False,
            competencies=competencies,
            prev_interventions=prev_interventions,
            output_embeddings=getattr(self, 'output_embeddings', False),
        )

    def _run_step(
        self,
        batch,
        batch_idx,
        train=False,
        intervention_idxs=None,
    ):
        x, y, (c, g, competencies, prev_interventions) = self._unpack_batch(
            batch
        )
        outputs = self._forward(
            x,
            intervention_idxs=intervention_idxs,
            c=c,
            y=y,
            train=train,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        _, _, y_logits = outputs[0], outputs[1], outputs[2]
        loss = self.loss_task(
            y_logits if y_logits.shape[-1] > 1 else y_logits.reshape(-1),
            y,
        )
        # compute accuracy
        _, (y_accuracy, y_auc, y_f1) = compute_accuracy(
            y_pred=y_logits,
            y_true=y,
            c_pred=None,
            c_true=None,
        )
        result = {
            "c_accuracy": 1,
            "c_auc": 1,
            "c_f1": 1,
            "y_accuracy": y_accuracy,
            "y_auc": y_auc,
            "y_f1": y_f1,
            "loss": loss.detach(),
        }
        if self.top_k_accuracy is not None:
            y_true = y.reshape(-1).cpu().detach()
            y_pred = y_logits.cpu().detach()
            labels = list(range(self.n_tasks))
            if isinstance(self.top_k_accuracy, int):
                top_k_accuracy = list(range(1, self.top_k_accuracy))
            else:
                top_k_accuracy = self.top_k_accuracy

            for top_k_val in top_k_accuracy:
                if top_k_val:
                    y_top_k_accuracy = sklearn.metrics.top_k_accuracy_score(
                        y_true,
                        y_pred,
                        k=top_k_val,
                        labels=labels,
                    )
                    result[f'y_top_{top_k_val}_accuracy'] = y_top_k_accuracy
        return loss, result

    def training_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=True)
        for name, val in result.items():
            if self.n_tasks <= 2:
                prog_bar = (
                    ("auc" in name)
                )
            else:
                prog_bar = (
                    ("y_accuracy" in name)

                )
            self.log(name, val, prog_bar=prog_bar)
        return {
            "loss": loss,
            "log": {
                "c_accuracy": result.get('c_accuracy', 0),
                "c_auc": result.get('c_auc', 0),
                "c_f1": result.get('c_f1', 0),
                "y_accuracy": result.get('y_accuracy', 0),
                "y_auc": result.get('y_auc', 0),
                "y_f1": result.get('y_f1', 0),
                "concept_loss": result.get('concept_loss', 0),
                "task_loss": result.get('task_loss', 0),
                "loss": result.get('loss', 0),
                "avg_c_y_acc": result.get('avg_c_y_acc', 0),
            },
        }

    def validation_step(self, batch, batch_no):
        _, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            if self.n_tasks <= 2:
                prog_bar = (("auc" in name))
            else:
                prog_bar = (("y_accuracy" in name))
            self.log("val_" + name, val, prog_bar=prog_bar)
        result = {
            "val_" + key: val
            for key, val in result.items()
        }
        return result

    def test_step(self, batch, batch_no):
        loss, result = self._run_step(batch, batch_no, train=False)
        for name, val in result.items():
            self.log("test_" + name, val, prog_bar=True)
        return result['loss']

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        if self.lr_scheduler_patience != 0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                verbose=True,
                patience=self.lr_scheduler_patience,
                factor=self.lr_scheduler_factor,
                min_lr=getattr(self, 'min_lr', 1e-5),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "loss",
            }
        return {
            "optimizer": optimizer,
            "monitor": "loss",
        }
