import torch
from torch import nn
from friendship_1dcnn import Friendship1DCNN
from cnn_corr_map_to_friendship_pred.friendship_csv_data_loader import FriendshipCSVDataLoader
import config

def train(model, train_loader, criterion, optimizer, device):
    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch+1}/{config.EPOCHS}')
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            y_batch = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, y_batch)

            # Backward and optimize based on SGD algo
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print(f'Step {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')

    return model

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            y_batch = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def run_trainer():
    device = torch.device(config.DEVICE)
    print(f'Going to train model in {device}')

    model = Friendship1DCNN(num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    data_loader = FriendshipCSVDataLoader(csv_path=config.CSV_PATH)
    train_loader, test_loader = data_loader.get_data_loaders(batch_size=config.BATCH_SIZE, train_test_split=config.TRAIN_TEST_SPLIT)
    trained_model = train(model, train_loader, criterion, optimizer, device)
    test_acc = evaluate(trained_model, test_loader, device)
    print(f'Final Test Accuracy: {test_acc:.2f}%')

if __name__ == "__main__":
    run_trainer()
