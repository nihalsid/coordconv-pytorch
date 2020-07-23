import pytorch_lightning as pl
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataset import FlyingMNIST
from model import UniformSplit, SmallNetwork
from PIL import Image, ImageDraw


class MNISTRegressionTrainer(pl.LightningModule):

    def __init__(self, kwargs):
        super(MNISTRegressionTrainer, self).__init__()
        self.hparams = kwargs
        self.model = UniformSplit(coords=self.hparams.coords)
        self.loss = torch.nn.MSELoss(reduction='mean')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-5)
        return [optimizer], []

    def train_dataloader(self):
        dataset = FlyingMNIST(self.hparams.dataset, "train")
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True)

    def val_dataloader(self):
        dataset = FlyingMNIST(self.hparams.dataset, "val")
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def visualization_dataloader(self, split):
        dataset = FlyingMNIST(self.hparams.dataset, split)
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

    def forward(self, batch):
        pred = self.model(batch["image"].unsqueeze(1))
        return pred

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss = self.loss(pred, batch['target'])
        self.logger.log_train_loss(self.global_step, loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        loss = self.loss(pred, batch['target'])
        return {'val_loss': loss, 'hiddens': {'losses': loss.cpu().item()}}

    def validation_epoch_end(self, outputs):
        val_losses = []
        for output in outputs:
            val_losses.append(output['hiddens']['losses'])
        mean_loss = np.array(val_losses).mean(0)
        self.logger.log_val_loss(self.global_step, mean_loss)
        print(f"\n\nIter {(self.global_step // 1000):06d}K | Loss: {mean_loss:.3f}")
        self.visualize_outputs('val')
        return {'val_loss': mean_loss}

    def visualize_outputs(self, split):
        visualization_loader = self.visualization_dataloader(split)
        data_box_dim = 14
        if self.hparams.dataset.startswith('not_so_clvr'):
            data_box_dim = 4
        output_vis_path = Path("runs") / self.hparams.dataset / self.hparams.experiment / "visualizations"
        output_vis_path.mkdir(exist_ok=True, parents=True)
        output_vis_path = output_vis_path / f'{(self.global_step // 1000):05d}.jpg'
        image = Image.new("RGB", (64 * 3, 64 * 3))
        for t, batch in enumerate(tqdm(visualization_loader, desc='visualizing')):
            local_image = Image.new('RGB', (64, 64))
            local_image.paste(Image.fromarray(batch['image'][0].numpy() * 255))
            batch["image"] = batch["image"].cuda(self.device)
            target = batch["target"][0].numpy()
            pred = self.forward(batch)[0].squeeze().cpu().numpy()
            draw = ImageDraw.Draw(local_image)
            draw.rectangle([(0, 0), (63, 63)], outline=(255, 255, 255))
            draw.rectangle([(max(pred[1] - data_box_dim, 0), max(pred[0] - data_box_dim, 0)), (min(pred[1] + data_box_dim, 64), min(pred[0] + data_box_dim, 64))], outline=(255, 0, 0))
            draw.rectangle([(max(target[1] - data_box_dim, 0), max(target[0] - data_box_dim, 0)), (min(target[1] + data_box_dim, 64), min(target[0] + data_box_dim, 64))], outline=(0, 255, 0))
            image.paste(local_image, ((t // 3) * 64, (t % 3) * 64))
            if t == 9:
                break
        image.save(output_vis_path)


if __name__ == '__main__':
    from util import parse_arguments
    from util import NestedTensorboardLogger
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    import os

    args = parse_arguments()

    logger = NestedTensorboardLogger(save_dir=os.path.join("runs", args.dataset), name=args.experiment)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join("runs", args.dataset, args.experiment, 'checkpoints'), save_top_k=-1, verbose=False, period=args.save_epoch)

    model = MNISTRegressionTrainer(args)

    trainer = Trainer(gpus=args.gpu, early_stop_callback=None, num_sanity_val_steps=args.sanity_steps, checkpoint_callback=checkpoint_callback, max_epochs=args.max_epoch, limit_val_batches=args.val_check_percent,
                      val_check_interval=min(args.val_check_interval, 1.0), check_val_every_n_epoch=max(1, args.val_check_interval), resume_from_checkpoint=args.resume, distributed_backend=args.distributed_backend, logger=logger)
    trainer.fit(model)
