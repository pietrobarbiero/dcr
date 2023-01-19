from torch.nn import BCELoss
import pytorch_lightning as pl
import torch
import os
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dcr.data.ba_shapes import load_ba_shapes
from dcr.data.celeba import load_celeba
from dcr.semantics import Logic, GodelTNorm
from dcr.nn import ConceptReasoningLayer, ConceptEmbedding

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class DeepGraphConceptReasoner(pl.LightningModule):
    def __init__(self, num_in_features, in_concepts, out_concepts, emb_size, concept_names, class_names,
                 learning_rate, loss_form, concept_loss_weight: float = 1., class_loss_weight: float = 1.,
                 temperature: float = 1., reasoner: bool = True,
                 logic: Logic = GodelTNorm()):
        super().__init__()
        self.concept_loss_weight = concept_loss_weight
        self.class_loss_weight = class_loss_weight
        # self.resnet = torchvision.models.resnet18(pretrained=True)
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        # n_features = self.resnet.fc.in_features
        # self.resnet.fc = torch.nn.Sequential(
        #     torch.nn.Linear(n_features, n_features),
        #     torch.nn.LeakyReLU(),
        # )

        num_hidden_features = 20

        self.conv0 = GCNConv(num_in_features, num_hidden_features)
        self.conv1 = GCNConv(num_hidden_features, num_hidden_features)
        self.conv2 = GCNConv(num_hidden_features, num_hidden_features)
        self.conv3 = GCNConv(num_hidden_features, num_hidden_features)

        n_features = num_hidden_features

        self.concept_embedder = ConceptEmbedding(n_features, in_concepts, emb_size)
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

    def forward(self, x, edge_index):
        x = self.conv0(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)


        c_emb, c_pred = self.concept_embedder(x)

        if self.reasoner:
            y_pred = self.predictor(c_emb, c_pred)
        else:
            y_pred = self.predictor(c_pred)
        return c_pred, y_pred


def test(c, y, c_pred, y_pred, mask):
    c_pred = torch.round(c_pred)
    y = y.max(dim=1)[1]
    y_pred = y_pred.max(dim=1)[1]

    c_correct = torch.all(torch.eq(c[mask], c_pred[mask]), dim=1).sum().item()
    y_correct = y_pred[mask].eq(y[mask]).sum().item()

    return c_correct / (len(c[mask])), y_correct / (len(y[mask]))


def training_loop(model, data, epochs, lr):
    # get data
    x = data["x"]
    edges = data['edges']
    y = data["y"]
    c = data["concept_labels"]
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.learning_rate)

    # list of accuracies
    train_accuracies_c, train_accuracies_y, test_accuracies_c, test_accuracies_y, train_losses, test_losses = list(), list(), list(), list(), list(), list()

    # iterate for number of epochs
    for epoch in range(epochs):
        # set mode to training
        model.train()

        # input data
        optimizer.zero_grad()

        c_pred, y_pred = model(x, edges)


        concept_loss = model.loss_form(c_pred[train_mask], c[train_mask].float())
        class_loss = model.loss_form(y_pred[train_mask], y[train_mask].float())
        loss = model.concept_loss_weight * concept_loss + model.class_loss_weight * class_loss
        model.log("train_loss", loss.item())
        model.log("concept_loss", concept_loss.item())
        model.log("class_loss", class_loss.item())

        loss.backward()
        optimizer.step()


        model.eval()
        c_pred, y_pred = model(x, edges)

        concept_loss = model.loss_form(c_pred[test_mask], c[test_mask].float())
        class_loss = model.loss_form(y_pred[test_mask], y[test_mask].float())
        test_loss = model.concept_loss_weight * concept_loss + model.class_loss_weight * class_loss
        model.log("test_loss", test_loss.item())

        train_acc_c, train_acc_y = test(c, y, c_pred, y_pred, train_mask)
        test_acc_c, test_acc_y = test(c, y, c_pred, y_pred, test_mask)

        ## add to list and print
        train_accuracies_c.append(train_acc_c)
        train_accuracies_y.append(train_acc_y)
        test_accuracies_c.append(test_acc_c)
        test_accuracies_y.append(test_acc_y)
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        print('Epoch: {:03d}, Loss: {:.5f}, Train Acc C: {:.5f}, Train Acc Y: {:.5f}, Test Acc C: {:.5f}, Test Acc Y: {:.5f}'.
              format(epoch, loss.item(), train_acc_c, train_acc_y, test_acc_c, test_acc_y), end = "\r")

    model.eval()
    return train_accuracies_c, train_accuracies_y, test_accuracies_c, test_accuracies_y


def main():
    epochs = 7000
    learning_rate = 0.001
    emb_size = 32
    batch_size = 1
    # limit_batches = 1.0

    # train_data, test_data, in_concepts, out_concepts, concept_names, class_names = load_celeba()
    # train_dl = DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)
    # test_dl = DataLoader(test_data, batch_size, shuffle=False, pin_memory=True)
    #
    # for d in train_dl:
    #     x, t = d
    #     c, y = t
    #     print("c ", c.shape)
    #     print("y ", y.shape)
    #     print(c)
    #     print(y)
    # return

    data, in_concepts, out_concepts, concept_names, class_names = load_ba_shapes()

    for fold in range(5):
        results_dir = f"./results/ba_shapes/{fold}/"
        os.makedirs(results_dir, exist_ok=True)
        model_path = os.path.join(results_dir, 'model.pt')

        num_in_features = data["x"].shape[1]
        model = DeepGraphConceptReasoner(num_in_features=num_in_features, in_concepts=in_concepts, out_concepts=out_concepts, emb_size=emb_size,
                                    concept_names=concept_names, class_names=class_names,
                                    learning_rate=learning_rate, loss_form=BCELoss(),
                                    concept_loss_weight=1, class_loss_weight=0.1, logic=GodelTNorm(),
                                    reasoner=True, temperature=1)

        if not os.path.exists(model_path):
            print(f'Running epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}')
            logger = TensorBoardLogger(save_dir=results_dir, name="lightning_logs")

            training_loop(model=model, data=data, epochs=epochs, lr=learning_rate)
            torch.save(model.state_dict(), model_path)

        # model.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    main()
