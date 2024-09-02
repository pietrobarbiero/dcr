import numpy as np
import pytorch_lightning as pl
import torch

from cem.models.cbm import ConceptBottleneckModel


################################################################################
## Posthoc CBM
################################################################################


class PCBM(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        concept_vectors,
        pretrained_model,
        concept_vector_intercepts=None,

        c2y_model=None,
        output_latent=False,
        residual=False,
        residual_model=None,
        reg_strength=1e-5, # regularization strenght for downstream sparse classifier
        l1_ratio=0.99,
        freeze_pretrained_model=True,
        freeze_concept_embeddings=True,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,
        training_intervention_prob=0.0,

        top_k_accuracy=None,
    ):
        """
        TODO
        """
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.n_tasks = n_tasks
        self.top_k_accuracy = top_k_accuracy

        # Let's first set up the embedding extractor from the pretrained model
        # The assumption here is that this model already generates the embedding
        # (i.e., it has been modified so that is output is the layer we are
        # targeting)
        self.embedding_generator = pretrained_model
        self.freeze_pretrained_model = freeze_pretrained_model
        if freeze_pretrained_model:
            # Then let's go ahead and freeze all the parameters in this model
            for child in self.embedding_generator.children():
                for param in child.parameters():
                    param.requires_grad = False

        # Time to initialize our embedding projection matrix
        self.freeze_concept_embeddings = freeze_concept_embeddings
        assert concept_vectors.shape[0] == n_concepts, (
            f'Expected the concept vector matrix to be of shape '
            f'(n_concepts, emb_size) however we got a matrix of size '
            f'{concept_vectors.shape} even though we were told we would expect '
            f'{self.n_concepts} concepts.'
        )
        self.concept_embeddings = torch.nn.Parameter(
            concept_vectors,
            requires_grad=(not freeze_concept_embeddings),
        )
        # Precompute the norms as we will use them during inference
        self.concept_norms = torch.norm(
            self.concept_embeddings,
            p=2,
            dim=1,
            keepdim=True,
        )
        # Handle intercepts (if provided)
        if concept_vector_intercepts is None:
            concept_vector_intercepts = torch.zeros(
                (n_concepts,)
            )
        self.concept_intercepts = torch.nn.Parameter(
            concept_vector_intercepts,
            requires_grad=(not freeze_concept_embeddings),
        )


        # Now construct the downstream interpretable model
        if c2y_model is None:
            self._c2y_model_provided = False
            self.c2y_model = torch.nn.Linear(self.n_concepts, self.n_tasks)
        else:
            self._c2y_model_provided = True
            self.c2y_model = c2y_model

        # Scafolding needed for intervention support here
        self.output_latent = output_latent
        self.use_concept_groups = use_concept_groups
        self.output_interventions = output_interventions
        self.intervention_policy = intervention_policy
        self.training_intervention_prob = training_intervention_prob
        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)

        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.ones(n_concepts)



        # And the losses. Notice we will still have a concept loss so that
        # we can use the CBM base class but this will be a no-op
        self.loss_concept = lambda *args, **kwargs: 0.0
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                weight=task_class_weights
            )
        )
        self.reg_strength = reg_strength
        self.l1_ratio = l1_ratio

        # Finally, stuff for the actual optimization
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer


    def _generate_concept_scores(
        self,
        x,
        latent=None,
        training=False,
    ):
        if latent is None:
            latent = self.embedding_generator(x)

        # Embeddings are (B, m) and concept vectors are (k, m)
        projections = torch.matmul(latent, self.concept_embeddings.T)
        projections = (projections + self.concept_intercepts)/self.concept_norms
        return projections, latent


    def _extra_losses(
        self,
        x,
        y,
        c,
        y_pred,
        c_sem,
        c_pred,
        competencies=None,
        prev_interventions=None,
    ):
        if self._c2y_model_provided:
            # Then we use all weights for the regularization
            l1_norm = 0.0
            l2_norm = 0.0
            for child in self.c2y_model.children():
                for param in child.parameters():
                    if param.requires_grad:
                        l1_norm += torch.norm(param, p=1)
                        l2_norm += torch.pow(param, 2)
            l2_norm = torch.square(l2_norm)
        else:
            # Otherwise this is the detault sparse network for which we only
            # need to normalize its weight vector
            l1_norm = torch.norm(self.c2y_model.weight, p=1)
            l2_norm = torch.norm(self.c2y_model.weight, p=2)
        elastic_net = l1_norm * self.l1_ratio + (1 - self.l1_ratio) * l2_norm

        # And normalize by classes and concepts while also considering the
        # regularizer strength
        return elastic_net * self.reg_strength / (
            self.n_concepts * self.n_tasks
        )


    def _forward(
        self,
        x,
        intervention_idxs=None,
        c=None,
        y=None,
        train=False,
        latent=None,
        competencies=None,
        prev_interventions=None,
        output_embeddings=False,
        output_latent=None,
        output_interventions=None,
    ):
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )

        c_pred, latent = self._generate_concept_scores(
            x=x,
            latent=latent,
            training=train,
        )

        # Now include any interventions that we may want to perform!
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            horizon = self.intervention_policy.num_groups_intervened
            if hasattr(self.intervention_policy, "horizon"):
                horizon = self.intervention_policy.horizon
            active_intervention_values = \
                self.active_intervention_values.to(
                    c_pred.device
                )
            pos_embs = torch.tile(
                active_intervention_values,
                (c.shape[0], 1),
            ).to(active_intervention_values.device)
            inactive_intervention_values = \
                self.inactive_intervention_values.to(
                    c_pred.device
                )
            neg_embs = torch.tile(
                inactive_intervention_values,
                (c.shape[0], 1),
            ).to(inactive_intervention_values.device)

            pos_embs = torch.unsqueeze(pos_embs, dim=-1)
            neg_embs = torch.unsqueeze(neg_embs, dim=-1)

            prior_distribution = self._prior_int_distribution(
                prob=c_pred,
                pos_embeddings=pos_embs,
                neg_embeddings=neg_embs,
                competencies=competencies,
                prev_interventions=prev_interventions,
                c=c,
                train=train,
                horizon=horizon,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_pred,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )
        else:
            c_int = c

        if not train:
            intervention_idxs = self._standardize_indices(
                intervention_idxs=intervention_idxs,
                batch_size=x.shape[0],
            )

        c_pred = self._concept_intervention(
            c_pred=c_pred,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
        )

        y_pred = self.c2y_model(c_pred)
        tail_results = []
        if output_interventions:
            if (
                (intervention_idxs is not None) and
                isinstance(intervention_idxs, np.ndarray)
            ):
                intervention_idxs = torch.FloatTensor(
                    intervention_idxs
                ).to(x.device)
            tail_results.append(intervention_idxs)
        if output_latent:
            tail_results.append(latent)
        if output_embeddings and (not pos_embs is None) and (
            not neg_embs is None
        ):
            tail_results.append(pos_embs)
            tail_results.append(neg_embs)
        return tuple([c_pred, c_pred, y_pred] + tail_results)
