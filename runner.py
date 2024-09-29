import torch
from torch import nn, optim
import clip
import pdb
from clip_finetune_jscc2 import get_logits
import numpy as np
from torch.optim.lr_scheduler import StepLR
import random
from torch.nn import functional as F
torch.autograd.set_detect_anomaly(True)

# Need: lr, optimizer(Adam)
# No Need: Criterion
 

"""
functions zoo:
train_epoch, train_epoch_gan,  validate_epoch
"""
def train_epoch(args, model, train_loader, device):
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    model.train()
    total_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        feat_decoded, loss = model(images, labels)
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(train_loader)

def train_epoch_straight(args, model, train_loader, device):
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    model.train()
    total_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = model(images, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(train_loader)

def train_epoch_validate(args, model, train_loader, text_features, device):
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        feat_decoded, loss = model(images, labels)
        loss.backward()
        optimizer.step() # 更新参数
        total_loss += loss.item()
        # 进入validate
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            text_features = F.normalize(text_features, p=2, dim=1)
            feat_decoded, loss = model(images, labels)
            logits_images, logits_text = get_logits(feat_decoded, text_features[:800].float(), 1)
            probs = logits_images.softmax(dim=1) 
            predicted = torch.argmax(probs, axis=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    scheduler.step()
    train_accuracy = 100 * correct / total
    print(f'Accuracy on train set: {train_accuracy:.2f}%')
    return total_loss / len(train_loader)

def train_epoch_aug(args, model, train_loader, device):
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    model.train()
    total_loss = 0.0
    for i, (images, aug_images, labels) in enumerate(train_loader):
        images, aug_images, labels = images.to(device), aug_images.to(device), labels.to(device)
        optimizer.zero_grad()
        _, _, loss = model(images, aug_images, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(train_loader)


def train_epoch_snr(args, model, train_loader, device):
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model.train()
    total_loss = 0.0
    # snr_db = np.random.uniform(-10, 10)
    snr_db = 20
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        feat_decoded, loss = model(images, labels, snr_db)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    scheduler.step()

    return total_loss / len(train_loader)


def train_multi_scale(args, model, train_loader, device):
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model.train()
    total_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        feat_decoded, loss = model(images, labels)     
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    scheduler.step()

    return total_loss / len(train_loader)


# def train_epoch_gan(args, model, train_loader, device):
#     if args.optimizer == "Adam":
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
#     scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

#     model.train()
#     total_loss = 0.0
#     for i, (images, labels) in enumerate(train_loader):
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         feat_decoded, model_loss, gan_loss = model(images, labels)     
#         loss = 0.0 * model_loss + 0.5 * gan_loss
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
        
#     scheduler.step()

#     return total_loss / len(train_loader)



def train_epoch_two_stage(args, model, train_loader_1, train_loader_2, device):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    model.train()
    total_loss = 0.0
    for i, (images, labels) in enumerate(train_loader_1):
        images = images.to(device)
        optimizer.zero_grad()
        
        _, loss, all_loss_values = model.forward(images, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 选择高损失样本
        high_loss_indices = torch.topk(all_loss_values, k=int(0.75 * images.shape[0]), largest=True).indices
        
        # 第二阶段训练
        # 在这个地方需要把train_loader_2里面对应high_loss_indices的那些samples拿出来！，因为尽管2个train_loader的每个sample都不同，但是sample的数量是一样的
        # 然后支队这些high_loss_indices所指向的samples进行反向传播
        high_loss_indices = high_loss_indices.cpu().numpy()
        multi_scale_images = []
        for idx in high_loss_indices:
            multi_scale_images.append(train_loader_1.dataset[idx][0])
        
        multi_scale_images = torch.stack(multi_scale_images).to(device)
        optimizer.zero_grad()

        # 第二阶段前向传递和反向传播
        _, loss_stage2, _ = model.forward(multi_scale_images, torch.tensor(high_loss_indices).to(device))
        loss_stage2.backward()
        optimizer.step()
        total_loss += loss_stage2.item()
        
    return total_loss / len(train_loader_1)



def validate_epoch(model, test_loader, text_features, device):
    model.eval()
    # metrics: test_accuracy of CLIP model's inference
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            feat_decoded = model(images, labels)
            
            text_features = F.normalize(text_features, p=2, dim=1)

            # logits_images, logits_text = get_logits(feat_decoded, text_features[-200:].float(), 1)
            logits_images, logits_text = get_logits(feat_decoded, text_features.float(), 1)
            # pdb.set_trace()
            probs = logits_images.softmax(dim=1) 
            predicted = torch.argmax(probs, axis=1)
            """
            # or use logits_text
            probs = logits_text.softmax(dim=0)
            predicted = torch.argmax(probs, axis=0)
            """
            total += labels.size(0)
            # labels = labels - 800
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f'Accuracy on test set: {test_accuracy:.2f}%')
    return test_accuracy


def validate_epoch_straight(model, test_loader, device):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            loss = model(images, labels)
            total_loss += loss.item()
    return total_loss/len(test_loader)

def validate_epoch_classify(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            predicted = model(images, labels)
            total += labels.size(0)
            labels = labels - 800
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f'Accuracy on test set: {test_accuracy:.2f}%')
    return test_accuracy


def validate_epoch_snr(args, model, test_loader, text_features, device):
    model.eval()
    correct = 0
    total = 0
    snr = -10
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            feat_decoded = model(images, labels, snr)
            text_features = F.normalize(text_features, p=2, dim=1)
            # logits_images, logits_text = get_logits(feat_decoded, text_features.float(), 1)
            logits_images, logits_text = get_logits(feat_decoded, text_features[-200:].float(), 1)
            probs = logits_images.softmax(dim=1) 
            predicted = torch.argmax(probs, axis=1)
            total += labels.size(0)
            labels = labels - 800
            correct += (predicted == labels).sum().item()
    test_accuracy = 100 * correct / total
    print(f'Accuracy on test set: {test_accuracy:.2f}%')
    return test_accuracy



def validate_epoch_aug(model, test_loader, text_features, device):
    model.eval()
    # metrics: test_accuracy of CLIP model's inference
    correct = 0
    total = 0
    with torch.no_grad():
        for features, aug_features, labels in test_loader:
            features, aug_features, labels = features.to(device), aug_features.to(device), labels.to(device)
            feat_decoded = model(features, aug_features, labels)
            
            text_features = F.normalize(text_features, p=2, dim=1)

            logits_images, logits_text = get_logits(feat_decoded, text_features[-200:].float(), 1)
            # logits_images, logits_text = get_logits(feat_decoded, text_features.float(), 1)
            # pdb.set_trace()
            probs = logits_images.softmax(dim=1) 
            predicted = torch.argmax(probs, axis=1)
            """
            # or use logits_text
            probs = logits_text.softmax(dim=0)
            predicted = torch.argmax(probs, axis=0)
            """
            total += labels.size(0)
            labels = labels - 800
            correct += (predicted == labels).sum().item()
            # pdb.set_trace()

    test_accuracy = 100 * correct / total
    print(f'Accuracy on test set: {test_accuracy:.2f}%')
    return test_accuracy


def validate_epoch_text(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            val_loss = model(images, labels)
            total_val_loss += val_loss

    final_val_loss = total_val_loss / len(test_loader)
    print(f'Loss on test set: {final_val_loss:.2f}')
    return final_val_loss