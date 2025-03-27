import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torchvision.models import resnet50
from tqdm import tqdm

from cem.models.cbm import ConceptBottleneckModel
import cem.train.utils as utils



def update_osr_thresholds(
    model,
    dataloader,
    device='cuda',
    rec_percentile=95,
    proto_percentile=95,
):
    """
    Computes self.thr_rec and self.thr_y for a GlanceNet model.

    Arguments:
        model: Trained GlanceNet instance (with `eval()` mode, no OSR yet).
        dataloader: DataLoader over the training set.
        device: 'cuda' or 'cpu'.
        rec_percentile: Percentile for reconstruction error threshold.
        proto_percentile: Percentile for prototype distance thresholds.
    """
    model.eval()
    model.to(device)

    all_rec_errors = []
    all_proto_dists = [[] for _ in range(model.n_tasks)]  # per-class distances

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing OSR thresholds"):
            x, y = batch[:2]  # assuming batch = (x, y, ...) or (x, y)
            x, y = x.to(device), y.to(device)

            # Forward pass to get latent code
            latent = model.x2c_model(x)

            # Reconstruction error
            x_recon = model.recon_decoder(latent)
            x_flat = x.view(x.size(0), -1)
            x_recon_flat = x_recon.view(x_recon.size(0), -1)
            rec_error = model.reconstruction_loss(x_recon_flat, x_flat).mean(dim=1)
            all_rec_errors.append(rec_error.cpu().numpy())

            # Prototype distances
            # y_onehot = F.one_hot(y, num_classes=model.n_tasks).float()
            # z_proto = model.enc_z_from_y(y_onehot.to(device))  # [batch_size, z_dim]
            z_proto = model.enc_z_from_y(y)  # [batch_size, z_dim]
            proto_dists = F.pairwise_distance(latent, z_proto, p=2)

            for i in range(len(y)):
                class_id = y[i].item()
                all_proto_dists[class_id].append(proto_dists[i].item())

    # Aggregate and compute thresholds
    all_rec_errors = np.concatenate(all_rec_errors, axis=0)
    thr_rec = np.percentile(all_rec_errors, rec_percentile)

    thr_y = torch.zeros(model.n_tasks)
    for c in range(model.n_tasks):
        if len(all_proto_dists[c]) > 0:
            thr_y[c] = np.percentile(all_proto_dists[c], proto_percentile)
        else:
            # If class wasn't seen (shouldn't happen in training), set to a large value
            thr_y[c] = 1e6

    # Set thresholds on model
    model.thr_rec.data = torch.tensor(thr_rec, device=device)
    model.thr_y.data = thr_y.to(device)

    print(f"Set model.thr_rec to: {model.thr_rec.item():.4f}")
    print(f"Set model.thr_y to: {model.thr_y.tolist()}")


################################################################################
## GlanceNets
################################################################################


class GlanceNet(ConceptBottleneckModel):
    def __init__(
        self,
        n_concepts,
        n_tasks,
        input_shape,
        concept_loss_weight=0.01,
        task_loss_weight=1,
        reconstruction_loss=None,
        decoder_layers=[512, 256, 128, 64, 3],
        decoder_output_activation=None,
        beta=1,
        recon_weight=1,
        prior_loss_weight=1,
        conditional_prior=True,

        extra_dims=0,
        hidden_dim=64,
        output_latent=False,

        x2c_model=None,
        c_extractor_arch=utils.wrap_pretrained_model(resnet50),
        c2y_model=None,
        c2y_layers=None,

        optimizer="adam",
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        lr_scheduler_factor=0.1,
        lr_scheduler_patience=10,
        weight_loss=None,
        task_class_weights=None,

        active_intervention_values=None,
        inactive_intervention_values=None,
        intervention_policy=None,
        output_interventions=False,
        use_concept_groups=False,

        top_k_accuracy=None,
    ):
        """
        TODO
        """
        pl.LightningModule.__init__(self)
        self.n_concepts = n_concepts
        self.intervention_policy = intervention_policy
        self.output_latent = output_latent
        self.output_interventions = output_interventions
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.z_dim = n_concepts + extra_dims
        if x2c_model is not None:
            # Then this is assumed to be a module already provided as
            # the input to concepts method
            self.x2c_model = x2c_model
        else:
            self.backbone = c_extractor_arch(
                output_dim=hidden_dim,
            )
            self.back_to_prob_params = torch.nn.Sequential(
                self.backbone,
                torch.nn.LeakyReLU(),
                torch.nn.Linear(
                    hidden_dim,
                    # Maps to the mean and variance of the latent code z
                    2 * self.z_dim,
                )
            )
            self.x2c_model = self._x2c_model

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
        # Intervention-specific fields/handlers:
        if active_intervention_values is not None:
            self.active_intervention_values = torch.FloatTensor(
                active_intervention_values
            )
        else:
            # Setting to 5 for prob = 1 (as that would result in its sigmoid
            # value being very close to 1) and -5 if prob=0 (as that will
            # go to zero when applied a sigmoid)
            self.active_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]
            ) * 5.0
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.FloatTensor(
                inactive_intervention_values
            )
        else:
            # Setting to 5 for prob = 1 (as that would result in its sigmoid
            # value being very close to 1) and -5 if prob=0 (as that will
            # go to zero when applied a sigmoid)
            self.inactive_intervention_values = torch.FloatTensor(
                [1 for _ in range(n_concepts)]
            ) * -5.0

        self.loss_concept = torch.nn.BCELoss(weight=weight_loss)
        self.reconstruction_loss = (
            torch.nn.MSELoss(reduction='none') if reconstruction_loss is None
            else reconstruction_loss
        )
        self.loss_task = (
            torch.nn.CrossEntropyLoss(weight=task_class_weights)
            if n_tasks > 1 else torch.nn.BCEWithLogitsLoss(
                pos_weight=task_class_weights
            )
        )
        self.concept_loss_weight = concept_loss_weight
        self.task_loss_weight = task_loss_weight
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.extra_dims = extra_dims
        self.beta = beta
        self.recon_weight = recon_weight
        self.prior_loss_weight = prior_loss_weight
        self.conditional_prior = conditional_prior
        self.top_k_accuracy = top_k_accuracy
        self.n_tasks = n_tasks
        self.bool = False
        self.sigmoidal_prob = False
        self.sigmoidal_extra_capacity = False
        self.use_concept_groups = use_concept_groups

        # The prior model
        self.prior_means = torch.nn.Parameter(
            torch.zeros(n_tasks, self.z_dim),
            requires_grad=True,
        )
        # self.enc_z_from_y = torch.nn.Linear(n_tasks, self.z_dim)

        # Make the decoder
        current_size = self.z_dim
        recon_decoder_layers = [
            torch.nn.Linear(current_size, decoder_layers[0] * 4 * 4),
            torch.nn.Unflatten(-1, (decoder_layers[0], 4, 4)),
        ]
        current_size = decoder_layers[0]
        for idx, num_acts in enumerate(decoder_layers):
            recon_decoder_layers.extend([
                torch.nn.ConvTranspose2d(current_size, num_acts, kernel_size=4, stride=2, padding=1),
                # torch.nn.BatchNorm2d(num_acts),
            ])
            if idx != (len(decoder_layers) - 1):
                recon_decoder_layers.append(torch.nn.LeakyReLU())
            current_size = num_acts
        # recon_decoder_layers.append(torch.nn.Flatten())
        # size = 2**(len(decoder_layers) + 2)
        # current_size = size * size * current_size
        # recon_decoder_layers.append(
        #     torch.nn.Linear(
        #         current_size,
        #         np.prod(input_shape[-3:]),
        #     )
        # )
        if decoder_output_activation is not None:
            recon_decoder_layers.append(decoder_output_activation)
        self.recon_decoder = torch.nn.Sequential(
            *recon_decoder_layers,
            # torch.nn.Unflatten(dim=-1, unflattened_size=input_shape[-3:])
            torch.nn.Upsample(size=input_shape[-2:], mode='bilinear', align_corners=False),
        )

        # Loss variables
        self._means = None
        self._vars = None
        self._latent_code = None

        # OSR Variables
        self.cluster_loss = torch.nn.BCELoss(reduction='mean')
        self._osr_variables_set = True
        self.thr_rec = torch.nn.Parameter(
            1000*torch.ones(size=(),),
            requires_grad=False,
        )
        self.thr_y = torch.nn.Parameter(
            1000*torch.ones(size=(n_tasks,),),
            requires_grad=False,
        )

    def enc_z_from_y(self, y):
        if not self.conditional_prior:
            # Then we assume we just want a standard gaussian prior
            return torch.zeros((y.shape[0], self.z_dim)).to(y.device)
        return torch.cat(
            [
                self.prior_means[label, :].unsqueeze(0) for label in y
            ],
            dim=0,
        )

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
        loss = 0.0
        if self.recon_weight and (self._latent_code is not None):
            # Then compute the reconstruction loss
            x_recon = self.recon_decoder(self._latent_code)
            recon_losses = self.recon_weight * self.reconstruction_loss(
                x_recon.view(x_recon.size(0), -1),
                x.view(x.size(0), -1),
            )
            recon_losses = torch.mean(recon_losses)
            # print("recon_losses =", recon_losses)
            loss += recon_losses

        # And align the prior distribution using the same approach
        # as the original GlanceNet code base
        if self.conditional_prior and self.prior_loss_weight and (
            y is not None
        ):
            # y_onehot = F.one_hot(y, self.n_tasks).to(
            #     dtype=torch.float,
            #     device=y.device,
            # )
            # mu_cluster = self.enc_z_from_y(y_onehot)
            mu_cluster = self.enc_z_from_y(y)
            concepts = (
                1 + torch.tanh(mu_cluster[:, :self.n_concepts] / 2)
            ) / 2
            prior_loss = self.cluster_loss(
                concepts,
                c,
            )
            # print("prior_loss =", prior_loss)
            loss += self.prior_loss_weight * prior_loss

        if (self.beta != 0) and (self._means is not None) and (
            self._logvars is not None
        ):
            # Then add the KL divergence loss
            # y_onehot = F.one_hot(y, self.n_tasks).to(
            #     dtype=torch.float,
            #     device=y.device,
            # )
            # mu_target = self.enc_z_from_y(y_onehot)
            mu_target = self.enc_z_from_y(y)
            kl_loss = (
                -0.5 * torch.sum(
                    1 + self._logvars -
                    (self._means - mu_target).pow(2) -
                    self._logvars.exp()
                )
            )
            # print("kl_loss =", self.beta * kl_loss)
            loss += self.beta * kl_loss
        return loss

    def _x2c_model(self, x):
        # First compute the distribution's mean and variance
        mean_logvar_concat = self.back_to_prob_params(x)
        # Split them up into their corresponding tensors
        mean = mean_logvar_concat[:, :self.z_dim]
        logvar = mean_logvar_concat[:, self.z_dim:]
        # Save the means as we will use this for our losses and predictions
        self._means = mean
        self._logvars = logvar
        # Generate a single latent code
        sigma = torch.exp(0.5 * self._logvars)
        eps = torch.randn_like(sigma)
        latent_code = mean + eps * sigma
        self._latent_code = latent_code
        return latent_code

    def _update_prediction_with_osr(
        self,
        x,
        y_pred,
    ):
        if self.training or (not self._osr_variables_set):
            # Then nothing to do here
            return y_pred

        # Step 1: Reconstruction error
        x_recon = self.recon_decoder(self._latent_code)
        x_flat = x.view(x.size(0), -1)
        x_recon_flat = x_recon.view(x_recon.size(0), -1)
        rec_error = self.reconstruction_loss(x_recon_flat, x_flat).mean(dim=1)
        unknown_rec_mask = rec_error > self.thr_rec  # shape: [batch_size]

        # Step 2: Prototype distance thresholding
        # Determine predicted class from y_pred
        if self.n_tasks == 1:
            # Binary classification: threshold at 0.5 after sigmoid
            pred_class = (torch.sigmoid(y_pred) > 0.5).long().squeeze()
        else:
            # Multi-class classification: take argmax over logits
            pred_class = torch.argmax(y_pred, dim=1)

        # Get prototype latent vectors
        # y_onehot = F.one_hot(pred_class, num_classes=self.n_tasks).float().to(x.device)
        # z_proto = self.enc_z_from_y(y_onehot)  # shape: [batch_size, z_dim]
        z_proto = self.enc_z_from_y(pred_class)  # shape: [batch_size, z_dim]

        # Compute distances to prototype
        latent = self._latent_code  # shape: [batch_size, z_dim]
        proto_dists = F.pairwise_distance(latent, z_proto, p=2)  # shape: [batch_size]

        # Thresholding on class-specific thresholds
        thr_y_vals = self.thr_y[pred_class]  # shape: [batch_size]
        unknown_proto_mask = proto_dists > thr_y_vals  # shape: [batch_size]

        # Step 3: Combine both masks
        unknown_mask = unknown_rec_mask | unknown_proto_mask  # shape: [batch_size]

        # Step 4: Replace unknowns in y_pred
        new_y_pred = y_pred.clone()
        if self.n_tasks == 1:
            new_y_pred[unknown_mask] = torch.tensor(0.0, device=new_y_pred.device)
        else:
            new_y_pred[unknown_mask] = torch.zeros(self.n_tasks, device=new_y_pred.device)
        return new_y_pred

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
        output_interventions = (
            output_interventions if output_interventions is not None
            else self.output_interventions
        )
        output_latent = (
            output_latent if output_latent is not None
            else self.output_latent
        )
        if latent is None:
            latent = self.x2c_model(x)
        mu_processed = torch.tanh(latent[:, :self.n_concepts] / 2)
        if self.extra_dims:
            # Then we only sigmoid on the probability bits but
            # let the other entries up for grabs
            # c_pred_probs = torch.sigmoid(latent[:, :self.n_concepts])
            # c_pred_probs = (1 + mu_processed)/2
            c_pred_probs = torch.sigmoid(self._means[:, :self.n_concepts])
            # c_others = latent[:, self.n_concepts:]
            # c_pred =  torch.cat([c_pred_probs, c_others], axis=-1)
            c_pred =  latent
            c_sem = c_pred_probs
        else:
            # c_pred = torch.sigmoid(latent)
            # c_pred = (1 + mu_processed)/2
            c_pred = torch.sigmoid(self._means[:, :self.n_concepts])
            c_sem = c_pred

        if output_embeddings or (
            (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        )):
            pos_embeddings = torch.ones(c_sem.shape).to(x.device)
            neg_embeddings = torch.zeros(c_sem.shape).to(x.device)
            if (
                (self.active_intervention_values is not None) and
                (self.inactive_intervention_values is not None)
            ):
                active_intervention_values = \
                    self.active_intervention_values.to(
                        c_pred.device
                    )
                pos_embeddings = torch.tile(
                    active_intervention_values,
                    (c.shape[0], 1),
                ).to(active_intervention_values.device)
                inactive_intervention_values = \
                    self.inactive_intervention_values.to(
                        c_pred.device
                    )
                neg_embeddings = torch.tile(
                    inactive_intervention_values,
                    (c.shape[0], 1),
                ).to(inactive_intervention_values.device)
            else:
                out_embs = c_pred.detach().cpu().numpy()
                for concept_idx in range(self.n_concepts):
                    pos_embeddings[:, concept_idx] = np.percentile(
                        out_embs[:, concept_idx],
                        95,
                    )
                    neg_embeddings[:, concept_idx] = np.percentile(
                        out_embs[:, concept_idx],
                        5,
                    )
            pos_embeddings = torch.unsqueeze(pos_embeddings, dim=-1)
            neg_embeddings = torch.unsqueeze(neg_embeddings, dim=-1)

        # Now include any interventions that we may want to include
        if (intervention_idxs is None) and (c is not None) and (
            self.intervention_policy is not None
        ):
            prior_distribution = self._prior_int_distribution(
                c=c,
                prob=c_sem,
                pos_embeddings=pos_embeddings,
                neg_embeddings=neg_embeddings,
                competencies=competencies,
                prev_interventions=prev_interventions,
                train=train,
                horizon=1,
            )
            intervention_idxs, c_int = self.intervention_policy(
                x=x,
                c=c,
                pred_c=c_sem,
                y=y,
                competencies=competencies,
                prev_interventions=prev_interventions,
                prior_distribution=prior_distribution,
            )
        else:
            c_int = c
        c_pred = self._concept_intervention(
            c_pred=self._means,
            intervention_idxs=intervention_idxs,
            c_true=c_int,
        )
        # We make the task prediction based only on the aligned concepts/latent
        # codes
        # y_pred = self.c2y_model(mu_processed[:, :self.n_concepts])
        # y_pred = self.c2y_model(self._means[:, :self.n_concepts])
        y_pred = self.c2y_model(c_pred[:, :self.n_concepts])
        y_pred = self._update_prediction_with_osr(
            x=x,
            y_pred=y_pred,
        )
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
        tail_results += self._extra_tail_results(
            x=x,
            y=y,
            c=c,
            c_sem=c_sem,
            competencies=competencies,
            prev_interventions=prev_interventions,
        )
        return tuple([c_sem, c_pred, y_pred] + tail_results)
