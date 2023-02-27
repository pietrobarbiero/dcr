import numpy as np
import torch
from torch import nn
from dcr.semantics import ProductTNorm, GodelTNorm, Logic
from dcr.nn import softselect
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, BCELoss
from collections import Counter

# class MNISTEncoder(nn.Module):
#     def __init__(self, output_size):
#         super(MNISTEncoder, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 6, 5),
#             nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
#             nn.ReLU(True),
#             nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
#             nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
#             nn.ReLU(True),
#         )
#
#         self.mlp = nn.Sequential(
#             nn.Linear(16 * 4 * 4, 128),
#             nn.ReLU(True),
#             nn.Linear(128, output_size),
#             nn.ReLU(True)
#         )
#
#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.view(-1, 16 * 4 * 4)
#         x = self.mlp(x)
#         return x



class MNISTEncoder(nn.Module):

    def __init__(self, output_size):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x



class MNISTDecoder(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
        unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
            padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class MNISTAutoEncoder(nn.Module):

    def __init__(self, hidden_size):
        super(MNISTAutoEncoder, self).__init__()

        self.encoder = MNISTEncoder(output_size=hidden_size)
        self.decoder = MNISTDecoder(input_size=hidden_size)

    def forward(self, x):
        x = self.encoder(x)
        x= self.decoder(x)
        return x


class MNISTClassifier(nn.Module):
    def __init__(self, input_size, num_classes=10, with_softmax=True):
        super(MNISTClassifier, self).__init__()
        super().__init__()
        self.with_softmax = with_softmax
        if with_softmax:
            self.softmax = nn.Softmax(1)
        self.classifier = nn.Sequential(nn.Linear(input_size, num_classes))

    def forward(self, x):
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x

class MNISTGumbelClassifier(nn.Module):
    def __init__(self, input_size, num_classes=10, temperature = 1):
        super(MNISTGumbelClassifier, self).__init__()
        super().__init__()
        self.temperature = temperature
        self.classifier = nn.Sequential(nn.Linear(input_size, input_size),
                                        nn.LeakyReLU(),
                                        nn.Linear(input_size, num_classes))

    def forward(self, x):
        x = self.classifier(x)
        x = torch.nn.functional.gumbel_softmax(x, tau=self.temperature, hard=False, dim=- 1)
        return x


class ConceptEmbedder(torch.nn.Module):
    def __init__(self, input_size, num_concepts, emb_size):
        super().__init__()
        self.num_concepts = num_concepts
        self.emb_size = emb_size
        self.model = nn.Sequential(
            torch.nn.Linear(input_size * 2, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, 2 * num_concepts * emb_size),
        )

    def forward(self, x):
        x = self.model.forward(x)
        x = x.view(-1, 2 * self.num_concepts, self.emb_size)
        return x


class ConceptReasoningLayer(torch.nn.Module):
    def __init__(self, emb_size, n_classes, logic: Logic, temperature: float = 1.):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.logic = logic
        self.filter_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )
        self.sign_nn = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )
        self.temperature = temperature

    def forward(self, x, c, return_attn=False, sign_attn=None, filter_attn=None):
        values = c.unsqueeze(-1).repeat(1, 1, self.n_classes)

        if sign_attn is None:
            # compute attention scores to build logic sentence
            # each attention score will represent whether the concept should be active or not in the logic sentence
            sign_attn = torch.sigmoid(self.sign_nn(x))  # TODO: might be independent of input x (but requires OR)

            # s = self.sign_nn(x)
            # s1 = torch.softmax(s[:,:10], dim=1)
            # s2 = torch.softmax(s[:,10:], dim=1)
            # sign_attn = torch.concat((s1,s2), dim=1)

        # attention scores need to be aligned with predicted concept truth values (attn <-> values)
        # (not A or V) and (A or not V) <-> (A <-> V)
        sign_terms = self.logic.iff_pair(sign_attn, values)    # TODO: temperature sharp here?

        if filter_attn is None:
            # compute attention scores to identify only relevant concepts for each class
           filter_attn = softselect(self.filter_nn(x), self.temperature)
           # filter_attn = torch.ones_like(softselect(self.filter_nn(x), self.temperature))

        # filter values
        # filtered implemented as "or(a, not b)", corresponding to "b -> a"
        filtered_values = self.logic.disj_pair(sign_terms, self.logic.neg(filter_attn))
        # filtered_values = self.logic.disj_pair(sign_terms, filter_attn)   # TODO: avoid negation

        # generate minterm
        preds = self.logic.conj(filtered_values, dim=1).squeeze(1).float()

        # TODO: add OR for global explanations

        if return_attn:
            return preds, sign_attn, filter_attn
        else:
            return preds

    def explain(self, x, c, mode, concept_names=None, class_names=None):
        assert mode in ['local', 'global', 'exact']

        if concept_names is None:
            concept_names = [f'c_{i}' for i in range(c.shape[1])]
        if class_names is None:
            class_names = [f'y_{i}' for i in range(self.n_classes)]

        # make a forward pass to get predictions and attention weights
        y_preds, sign_attn_mask, filter_attn_mask = self.forward(x, c, return_attn=True)

        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(x)):
            prediction = y_preds[sample_idx] > 0.5
            active_classes = torch.argwhere(prediction).ravel()

            if len(active_classes) == 0:
                # TODO: explain even not active classes!
                # if no class is active for this sample, then we cannot extract any explanation
                explanations.append({
                    'class': -1,
                    'explanation': '',
                    'attention': [],
                })
            else:
                # else we can extract an explanation for each active class!
                for target_class in active_classes:
                    attentions = []
                    minterm = []
                    for concept_idx in range(len(concept_names)):
                        c_pred = c[sample_idx, concept_idx]
                        sign_attn = sign_attn_mask[sample_idx, concept_idx, target_class]
                        filter_attn = filter_attn_mask[sample_idx, concept_idx, target_class]

                        # we first check if the concept was relevant
                        # a concept is relevant <-> the filter attention score is lower than the concept probability
                        at_score = 0
                        sign_terms = self.logic.iff_pair(sign_attn, c_pred).item()
                        if self.logic.neg(filter_attn) < sign_terms:
                            if sign_attn >= 0.5:
                                # if the concept is relevant and the sign is positive we just take its attention score
                                at_score = filter_attn.item()
                                if mode == 'exact':
                                    minterm.append(f'{sign_terms:.3f} ({concept_names[concept_idx]})')
                                else:
                                    minterm.append(f'{concept_names[concept_idx]}')
                            else:
                                # if the concept is relevant and the sign is positive we take (-1) * its attention score
                                at_score = -filter_attn.item()
                                if mode == 'exact':
                                    minterm.append(f'{sign_terms:.3f} (~{concept_names[concept_idx]})')
                                else:
                                    minterm.append(f'~{concept_names[concept_idx]}')
                        attentions.append(at_score)

                    # add explanation to list
                    target_class_name = class_names[target_class]
                    minterm = ' & '.join(minterm)
                    all_class_explanations[target_class_name].append(minterm)
                    explanations.append({
                        'sample-id': sample_idx,
                        'class': target_class_name,
                        'explanation': minterm,
                        'attention': attentions,
                    })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append({
                        'class': class_id,
                        'explanation': explanation,
                        'count': count,
                    })

        return explanations


class DeepConceptReasoner(pl.LightningModule):
    def __init__(self, emb_size, num_concepts, num_digits, num_additions, learning_rate,  logic=GodelTNorm(), temperature=10, temperature_gumbel = 1, num_concepts_supervisions = None, to_explain = None, class_weights = None):
        super().__init__()

        self.logic = logic
        self.emb_size = emb_size
        self.num_concepts = num_concepts
        self.temperature_gumbel = temperature_gumbel


        self.encoder = MNISTEncoder(output_size=emb_size)
        self.gumbel_classifier = MNISTGumbelClassifier(input_size=emb_size, num_classes=10, temperature = temperature_gumbel)
        self.concepts_embedder = ConceptEmbedder(emb_size, num_concepts, emb_size)

        self.reasoner = ConceptReasoningLayer(emb_size, num_additions, self.logic, temperature)
        self.neural_reasoner = nn.Sequential(nn.Linear(1 * 2* self.num_concepts, emb_size),
                                             nn.ReLU(),
                                             nn.Linear(emb_size, num_additions))

        self.learning_rate = learning_rate
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.bce = BCELoss(reduction="mean")
        self.num_concpets_supervisions = num_concepts_supervisions
        self.explain_epochs = 0
        self.to_explain = to_explain
        self.best_val = 0
        self.class_weights = class_weights




    def concept_embeddings_and_predictions(self, X):
        encodings = []
        classes = []
        for x in X:
            x = self.encoder.forward(x)
            # cl = self.classifier.forward(x)
            cl = self.gumbel_classifier.forward(x)
            encodings.append(x)
            classes.append(cl)

        scene_enconding = torch.concat(encodings, dim=-1)
        concepts_predictions = torch.concat(classes, dim=-1)
        concepts_embeddings= self.concepts_embedder(scene_enconding)

        return concepts_predictions, concepts_embeddings

    def forward(self, X, return_attn = False):

        concepts_predictions, concepts_embeddings = self.concept_embeddings_and_predictions(X)
        if return_attn:
            task_predictions_neural = self.neural_reasoner(concepts_predictions)
            task_predictions = torch.zeros_like(task_predictions_neural)
            return concepts_predictions, task_predictions, task_predictions_neural,None, None
            # task_predictions, sign_attn, filter_attn = self.reasoner.forward(concepts_embeddings, concepts_predictions, return_attn=True)
            # return concepts_predictions, task_predictions, task_predictions_neural,sign_attn, filter_attn
        else:
            task_predictions_neural = self.neural_reasoner(concepts_predictions)
            task_predictions = self.reasoner.forward(concepts_embeddings, concepts_predictions, return_attn=False)
            return concepts_predictions, concepts_embeddings, task_predictions, task_predictions_neural




    def explain(self,X):

        concepts_predictions, concepts_embeddings = self.concept_embeddings_and_predictions(X)
        explanations = self.reasoner.explain(concepts_embeddings, concepts_predictions, "global")
        return explanations




    def training_step(self, I, batch_idx):
        X = I[:2]
        y = I[2:4]
        z = I[4]
        y_pred, y_emb, z_pred, z_n_pred = self.forward(X)


        # loss_tasks = self.cross_entropy(z_pred, z)
        loss_tasks = self.bce(z_pred, torch.nn.functional.one_hot(z,19).float())
        loss_tasks_neural = self.cross_entropy(z_n_pred, z)
        # y_pred = y_pred[:, :10], y_pred[:, 10:]


        # loss_concepts_0 = self.cross_entropy(y_pred[0], torch.argmax(y_pred[0], dim=-1))
        # loss_concepts_1 = self.cross_entropy(y_pred[1], torch.argmax(y_pred[1], dim=-1))
        # loss_concepts = 0.0* (loss_concepts_0 + loss_concepts_1)
        # loss_concepts = 0


        # Loss
        loss =  0.001*loss_tasks + loss_tasks_neural
        # loss = loss_tasks
        # self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Tasks
        z_pred = torch.max(z_pred, dim=-1)[1]
        accuracy = (z_pred == z.squeeze()).float().mean()
        self.log("train_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)

        z_n_pred = torch.max(z_n_pred, dim=-1)[1]
        accuracy = (z_n_pred == z.squeeze()).float().mean()
        self.log("train_accuracy_neural", accuracy, prog_bar=True, on_step=False, on_epoch=True)

        # Digits
        # for d in [0,1]:
        #     y_p = torch.max(y_pred[d], dim=-1)[1]
        #     y_t = y[d]
        #     accuracy = (y_p == y_t.squeeze()).float().mean()
        #     self.log("train_accuracy_digit_%d" % d, accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, I, batch_idx):
        X = I[:2]
        y = I[2:4]
        z = I[4]
        y_pred, y_emb, z_pred, z_n_pred = self.forward(X)

        # y_pred = y_pred[:, :10], y_pred[:, 10:]


        # Tasks
        z_pred = torch.max(z_pred, dim=-1)[1]
        accuracy_t = (z_pred == z.squeeze()).float().mean()
        self.log("valid_accuracy", accuracy_t, prog_bar=True, on_step=False, on_epoch=True)

        z_n_pred = torch.max(z_n_pred, dim=-1)[1]
        accuracy_t_n = (z_n_pred == z.squeeze()).float().mean()
        self.log("valid_accuracy_neural", accuracy_t_n, prog_bar=True, on_step=False, on_epoch=True)

        # Digits
        # for d in [0, 1]:
        #     y_p = torch.max(y_pred[d], dim=-1)[1]
        #     y_t = y[d]
        #     accuracy = (y_p == y_t.squeeze()).float().mean()
        #     self.log("valid_accuracy_digit_%d" % d, accuracy, prog_bar=True, on_step=False, on_epoch=True)


    # def validation_epoch_end(self, outputs) -> None:
    #
    #     X1 = []
    #     X2 = []
    #     for X in self.val_dataloader():
    #         X1.append(X[0])
    #         X2.append(X[1])
    #
    #     X1 = torch.concat(X1, dim=0)
    #     X2 = torch.concat(X2, dim=0)
    #     X = [X1,X2]
    #     print(self.explain(X))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


#
# class DeepConceptReasoner(pl.LightningModule):
#     def __init__(self, in_concepts, out_concepts, emb_size, concept_names, class_names,
#                  learning_rate, loss_form, concept_loss_weight: float = 1., class_loss_weight: float = 1.,
#                  temperature: float = 1., reasoner: bool = True,
#                  logic: Logic = GodelTNorm()):
#         super().__init__()
#         self.concept_loss_weight = concept_loss_weight
#         self.class_loss_weight = class_loss_weight
#
#         for param in self.resnet.parameters():
#             param.requires_grad = False
#         n_features = self.resnet.fc.in_features
#         self.resnet.fc = torch.nn.Sequential(
#             torch.nn.Linear(n_features, n_features),
#             torch.nn.LeakyReLU(),
#         )
#         self.concept_embedder = ConceptEmbedding(n_features, in_concepts, emb_size)
#         self.reasoner = reasoner
#         if self.reasoner:
#             self.predictor = ConceptReasoningLayer(emb_size, out_concepts, logic, temperature)
#         else:
#             self.predictor = torch.nn.Sequential(
#                 torch.nn.Linear(in_concepts, 10),
#                 torch.nn.LeakyReLU(),
#                 torch.nn.Linear(10, out_concepts),
#                 torch.nn.Sigmoid()
#             )
#         self.concept_names = concept_names
#         self.class_names = class_names
#         self.learning_rate = learning_rate
#         self.loss_form = loss_form
#
#     def forward(self, x):
#         h = self.resnet.forward(x)
#         c_emb, c_pred = self.concept_embedder(h)
#         if self.reasoner:
#             y_pred = self.predictor(c_emb, c_pred)
#         else:
#             y_pred = self.predictor(c_pred)
#         return c_pred, y_pred
#
#     def training_step(self, batch, batch_idx):
#         x, t = batch
#         c, y = t
#         c_pred, y_pred = self.forward(x)
#         concept_loss = self.loss_form(c_pred, c.float())
#         class_loss = self.loss_form(y_pred, y.float())
#         loss = self.concept_loss_weight * concept_loss + self.class_loss_weight * class_loss
#         self.log("train_loss", loss.item())
#         self.log("concept_loss", concept_loss.item())
#         self.log("class_loss", class_loss.item())
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         x, t = batch
#         c, y = t[0], t[1]
#         c_pred, y_pred = self.forward(x)
#         concept_loss = self.loss_form(c_pred, c.float())
#         class_loss = self.loss_form(y_pred, y.float())
#         loss = self.concept_loss_weight * concept_loss + self.class_loss_weight * class_loss
#         self.log("test_loss", loss.item())
#         return loss
#
#     def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
#         x, t = batch
#         return self.forward(x)
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
#         # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)    # TODO: consider whether we need this or not
#         return optimizer



def pretty_print_explain(explanations, file = None):
    for ex in explanations:
        cl = int(ex["class"].split("_")[1])
        exp = [a for a in ex["explanation"].split("&") if "~" not in a]
        cnt = int(ex["count"])
        print(cl, exp, cnt, file = file)
