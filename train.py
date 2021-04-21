import torch
import torch.nn.functional as F
import reversi_layer_cpp
from torch.optim.lr_scheduler import MultiStepLR
from net import ReversiNet
import os
from tqdm import tqdm
import datetime


ITERS = 10000000
BATCH_SIZE = 128
device = 'cuda'

bk_dir = datetime.datetime.now().strftime("backup/%Y%m%d%H%M%S")
os.makedirs(bk_dir, exist_ok=True)


@torch.jit.script
@torch.no_grad()
def convert(w, b):
    x = w.new_zeros((w.size(0), 2, 64))
    x_mask = torch.arange(0, 64).unsqueeze(0).to(x.device)
    x[:, 0] = (w.unsqueeze(1) >> x_mask) & 1
    x[:, 1] = (b.unsqueeze(1) >> x_mask) & 1
    return x.view(-1, 2, 8, 8).float()


@torch.no_grad()
def evaluate(model, test_size):
    model.eval()
    data_b = torch.zeros((test_size,), dtype=torch.long)
    data_w = torch.zeros((test_size,), dtype=torch.long)
    data_b[:] = 0x0000000810000000
    data_w[:] = 0x0000001008000000
    is_std = True
    win_cnt = 0.
    lose_cnt = 0.
    while data_b.size(0):
        if is_std:
            win = reversi_layer_cpp.std(data_b, data_w)
            lose_cnt += torch.sum(win == 2)
            win_cnt += torch.sum(win == 0)
        else:
            pred = model(convert(data_b, data_w).to(device))
            _, win, _ = reversi_layer_cpp.forward(data_b, data_w, pred.to('cpu'))
            win_cnt += torch.sum(win == 2)
            lose_cnt += torch.sum(win == 0)
        data_b = data_b[win == 3]
        data_w = data_w[win == 3]
        is_std = not is_std

    win = reversi_layer_cpp.win(data_b, data_w)
    model.train()
    return win_cnt / (win_cnt + lose_cnt)


def train(model):
    import wandb
    wandb.init(project='reversi')

    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    sched = MultiStepLR(optim, 10000, [ITERS/10*8, ITERS/10*9])

    data_b = torch.zeros((BATCH_SIZE,), dtype=torch.long)
    data_w = torch.zeros((BATCH_SIZE,), dtype=torch.long)
    data_b[:] = 0x0000000810000000
    data_w[:] = 0x0000001008000000

    # init
    for i in range(BATCH_SIZE - 1):
        reversi_layer_cpp.forward(data_b[i:], data_w[i:], torch.ones((BATCH_SIZE - i, 65)))

    p_prev = None
    q_prev = None
    win_prev = None
    pbar = tqdm(range(ITERS))
    for iter in pbar:
        pred = model(convert(data_b, data_w).to(device)).to('cpu')
        s, win, valid_mask = reversi_layer_cpp.forward(data_b, data_w, pred)

        # Compute and print loss
        if win_prev is not None:
            with torch.no_grad():
                gt = 1 - pred[:, -1].sigmoid()
                gt[win_prev != 3] = win_prev[win_prev != 3].float() / 2
                residue = -F.softplus(pred[:, -1]) + F.softplus(-p_prev)
            loss_p = F.binary_cross_entropy_with_logits(p_prev, gt)
            loss_q = (residue * q_prev).mean()
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
        q_prev[s == 64] = 0
        win_prev = win

        if iter % 2000 == 0:
            if iter % 100000 == 0:
                torch.save(model.state_dict(), bk_dir + f"/checkpoint{iter//100000:04d}.pth")
            wandb.log({"win_std": evaluate(model, 100)})


if __name__ == '__main__':
    model = ReversiNet(train=True).to(device)
    # model = torch.nn.DataParallel(model)
    # model = torch.jit.script(model)

    # with open("w.pth", "rb") as f:
    #     state = torch.load(f, map_location=device)
    #     model.load_state_dict(state, strict=False)
    train(model)
    torch.save(model.state_dict(), bk_dir + "/w.pth")
