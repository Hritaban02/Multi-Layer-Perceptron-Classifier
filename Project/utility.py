import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def fit(model, X, y, evaluation_set, n_epochs=500, learning_rate=0.001, batch_learning=False, batch_size=10):
    (X_val, y_val) = evaluation_set
    criterion = nn.BCEWithLogitsLoss()
    valid_loss_mini = np.Inf
    # create your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    model.zero_grad()
    model.train()

    for i in range(0, n_epochs):
        valid_loss = 0
        optimizer.zero_grad()  # zero the gradient buffer

        if not batch_learning:
            x_tensor = torch.from_numpy(X).float()
            y_tensor = torch.from_numpy(y).float()
            y_tensor = y_tensor.view(-1, 1)

            output = model(x_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()    # Does the update

            model.eval()  # prep model for evaluation
            x_tensor = torch.from_numpy(X_val).float()
            output = model(x_tensor)
            y_tensor = torch.from_numpy(np.array(y_val)).float()
            y_tensor = y_tensor.view(-1, 1)
            loss = criterion(output, y_tensor)
            valid_loss = loss
        else:
            dataset = np.c_[X, y]
            loader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
            for batch_ndx, sample in enumerate(loader):
                x_tensor = sample[:, 0:-1].float()
                y_tensor = sample[:, -1].float()
                y_tensor = y_tensor.view(-1, 1)

                output = model(x_tensor)
                loss = criterion(output, y_tensor)
                loss.backward()
                optimizer.step()

                model.eval()  # prep model for evaluation
                x_tensor = torch.from_numpy(X_val).float()
                output = model(x_tensor)
                y_tensor = torch.from_numpy(np.array(y_val)).float()
                y_tensor = y_tensor.view(-1, 1)
                loss = criterion(output, y_tensor)
                valid_loss += loss
            valid_loss = valid_loss * batch_size / len(dataset)

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_mini:
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_mini = valid_loss


def eval_model(model, test_set):
    (X_test, y_test) = test_set
    model.eval()  # prep model for evaluation
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_test).float()
        y_test_pred = model(x_tensor)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list = (y_pred_tag.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    return accuracy_score(y_test, y_pred_list)
