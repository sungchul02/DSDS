import torch
import torch.nn as nn


def train(model, train_loader, val_loader, epochs=5, lr=0.0003, device="cpu"):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_train_acc = 0
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        # validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc

    return best_train_acc, best_val_acc