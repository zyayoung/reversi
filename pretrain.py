import torch
import torch.nn.functional as F
import reversi_layer_cpp
from torch.optim.lr_scheduler import MultiStepLR
from net import ReversiNet
import os
from tqdm import tqdm
import datetime
from data import read_data
import random
from copy import deepcopy

ITERS = 1000000
BATCH_SIZE = 128
device = 'cuda'

bk_dir = datetime.datetime.now().strftime("backup/%Y%m%d%H%M%S")
os.makedirs(bk_dir, exist_ok=True)

# init
data = read_data('data')
queue = [deepcopy(random.choice(data)) for _ in range(BATCH_SIZE)]


@torch.jit.script
@torch.no_grad()
def convert(w, b):
    x = w.new_zeros((w.size(0), 2, 64))
    x_mask = torch.arange(0, 64).unsqueeze(0).to(x.device)
    x[:, 0] = (w.unsqueeze(1) >> x_mask) & 1
    x[:, 1] = (b.unsqueeze(1) >> x_mask) & 1
    return x.view(-1, 2, 8, 8).float()


def step(data_b, data_w):
    b = data_b.size(0)
    prob = torch.zeros((b, 65))
    prob[..., -1] = 1
    for i in range(b):
        prob[i, queue[i][0]] = 1
    s, win, valid_mask = reversi_layer_cpp.forward(data_b, data_w, prob)
    for i in torch.where(~valid_mask[..., -1])[0]:
        queue[i].pop(0)
    for i in range(b):
        if win[i] != 3 or len(queue[i]) == 0:
            data_b[i] = 0x0000000810000000
            data_w[i] = 0x0000001008000000
            queue[i] = deepcopy(random.choice(data))
    return s, win, valid_mask


def train(model):
    import wandb
    wandb.init(project='reversi')

    optim = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
    sched = MultiStepLR(optim, milestones=[ITERS/10*8, ITERS/10*9])

    data_b = torch.zeros((BATCH_SIZE,), dtype=torch.long)
    data_w = torch.zeros((BATCH_SIZE,), dtype=torch.long)
    data_b[:] = 0x0000000810000000
    data_w[:] = 0x0000001008000000
    for i in range(1, BATCH_SIZE):
        step(data_b[:i], data_w[:i])

    p_prev = None
    q_prev = None
    win_prev = None
    pbar = tqdm(range(ITERS))
    for iter in pbar:
        pred = model(convert(data_b, data_w).to(device)).to('cpu')
        s, win, valid_mask = step(data_b, data_w)

        # Compute and print loss
        if win_prev is not None:
            with torch.no_grad():
                gt = 1 - pred[:, -1]
                gt[win_prev != 3] = win_prev[win_prev != 3].float() / 2
            loss_p = F.binary_cross_entropy(p_prev, gt.detach())
            loss_q = q_prev.mean()
            loss = loss_p + loss_q

            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()

            if iter % 100 == 0:
                wandb.log({"loss_q": loss_q, "loss_p": loss_p})
            pbar.set_description_str(f"q: {loss_q:.4f}; p: {loss_p:.4f}")

        p_prev = pred[:, -1]
        q_prev = F.cross_entropy(pred - (~valid_mask * 1e12), s, reduction='none')
        q_prev[s == 64] = 1
        win_prev = win

        if iter % 100000 == 0:
            torch.save(model.state_dict(), bk_dir + f"/checkpoint{iter//100000:04d}.pth")

if __name__ == '__main__':
    model = ReversiNet(train=True).to(device)
    # model = torch.nn.DataParallel(model)
    # model = torch.jit.script(model)

    # with open("w.pth", "rb") as f:
    #     state = torch.load(f, map_location=device)
    #     model.load_state_dict(state, strict=False)
    train(model)
    torch.save(model.state_dict(), bk_dir + "/w.pth")
