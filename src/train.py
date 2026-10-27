import torch
from torch.utils.data import DataLoader
from monai.data import Dataset
from model import build_model
from dataset import build_data_list, train_transforms, LABELS, collate_fn

device = torch.device('cuda')
items = build_data_list('/workspace/data/Data_Entry_2017.csv', '/workspace/data')
ds = Dataset(data=items, transform=train_transforms)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0, collate_fn=collate_fn)
model = build_model(len(LABELS)).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'epoch {epoch} loss {loss.item():.4f}')
    torch.save(model.state_dict(), f'/workspace/outputs/densenet121_epoch{epoch}.pt')
