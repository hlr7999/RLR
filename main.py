import argparse

from torch.backends import cudnn
from utils.utils import *

from trainer import Trainer
from decoder_trainer import DecoderTrainer
from datasets.mvtec import MVTEC_CLASS_NAMES
from datasets.visa import VISA_CLASS_NAMES


def main(args):
    cudnn.benchmark = True
    init_seeds(3407)

    trainer = Trainer(args)

    if args.mode == 'train':
        if args.with_decoder:
            trainer = DecoderTrainer(args)
            trainer.train()
            return
        img_auc, pix_auc, img_aucs, pix_aucs = trainer.train()
    elif args.mode == 'test':
        img_auc, pix_auc, img_aucs, pix_aucs = trainer.test(vis=args.vis, checkpoint_path=args.save_path)

    print('======================Best Auc======================')
    class_names = MVTEC_CLASS_NAMES
    if args.dataset == 'visa':
        class_names = VISA_CLASS_NAMES
    for c, i, p in zip(class_names, img_aucs, pix_aucs):
        print(f"class: {c} | img auc: {i} | pix auc: {p}")
    print(f"Avg | img auc: {img_auc} | pix auc: {pix_auc}")
    print('=======Format=======')
    for i, p in zip(img_aucs, pix_aucs):
        print(f'{round(i*100, 2)}/{round(p*100, 2)}')
    print(f'{round(img_auc*100, 2)}/{round(pix_auc*100, 2)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='')
    # basic config
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--with_decoder', action='store_true', default=False)
    # dataset config
    parser.add_argument('--dataset', default='mvtec', type=str, choices=['mvtec', 'visa'])
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--inp_size', default=256, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    # model config
    parser.add_argument('--backbone_arch', default='tf_efficientnet_b6', type=str)
    parser.add_argument('--feature_levels', default=2, type=int)
    parser.add_argument('--out_indices', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--block_mode', type=str, default='parallel', choices=['parallel', 'serial'])
    parser.add_argument('--blocks', nargs='+', type=str, default=['mca', 'nsa'])
    parser.add_argument('--blocks_gate', type=str, default='none', choices=['none', 'gate', 'net'])
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--ref_len', type=int, default=1)
    parser.add_argument('--ref_loss', action='store_true', default=False)
    # trainer config
    parser.add_argument('--feature_jitter', type=float, default=0.0)
    parser.add_argument('--noise_prob', type=float, default=1.0)
    parser.add_argument('--no_avg', action='store_true', default=False)
    parser.add_argument('--with_mask', action='store_true', default=False)
    # misc
    parser.add_argument('--save_path', type=str, default='save')
    parser.add_argument('--save_prefix', type=str, default='')
    parser.add_argument('--vis', action='store_true', default=False)

    args = parser.parse_args()
    
    args.device = torch.device("cuda")
    args.img_size = (args.inp_size, args.inp_size)
    args.crop_size = (args.inp_size, args.inp_size)
    args.norm_mean, args.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    args.save_path = f'{args.root_path}/{args.save_path}'
    os.makedirs(args.save_path, exist_ok=True)

    assert args.feature_levels == len(args.out_indices)
    for b in args.blocks:
        assert b in ['mca', 'ca', 'msa', 'nsa', 'sa']

    args_dict = vars(args)
    print('------------ Options -------------')
    for k, v in sorted(args_dict.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    
    main(args)
