import torch
import time


class Logger(object):
    def __init__(self, epoch, mode, length, save_param=1,calculate_mean=False):
        self.epoch = epoch
        self.mode = mode
        self.length = length
        self.save_param = save_param
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x

    def __call__(self, loss, metrics, i):
        track_str = '\r{}{:5d} | {:5d}/{:<5d}| '.format(self.mode, self.epoch, i + 1, self.length)
        loss_str = 'loss: {:9.4f} | '.format(self.fn(loss, i))
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())
        print(track_str + loss_str + metric_str + '   ', end='')
        if i%self.save_param == 0:
            with open("log.txt","a") as f:
                f.write(track_str + loss_str + metric_str + '   ')
        if i + 1 == self.length:
            print('')


class BatchTimer(object):
    def __init__(self, rate=True, per_sample=True):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred, y):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)


def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


def pass_epoch(
    epoch = None, model=None, loss_fn=None, loader=None, optimizer=None, scheduler=None,save_every=1,
    batch_metrics={'time': BatchTimer()}, show_running=True,
    device='cpu'
):
    mode = 'Train' if model.training else 'Val'
    logger = Logger(epoch, mode, length=len(loader), save_param=save_every, calculate_mean=show_running)
    loss = 0
    metrics = {}

    for i_batch, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss_batch = loss_fn(y_pred, y)

        if model.training:
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn(y_pred, y).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]
        
        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
        if show_running:
            logger(loss, metrics, i_batch)
        else:
            logger(loss_batch, metrics_batch, i_batch)
    
    if model.training and scheduler is not None:
        scheduler.step()

    loss = loss / (i_batch + 1)
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}

    return loss, metrics


# def collate_pil(x): 
#     out_x, out_y = [], [] 
#     for xx, yy in x: 
#         out_x.append(xx) 
#         out_y.append(yy) 
#     return out_x, out_y 