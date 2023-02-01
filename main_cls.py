import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm
from util import IOStream
from models.model_cls import Model
from data import ModelNet40
from visualdl import LogWriter


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)

    if args.mode == 'train':
        if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
            os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')

    os.system('cp models/model_cls.py checkpoints' + '/' + args.exp_name + '/' + 'model.py')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py')
    os.system('cp main_cls.py checkpoints' + '/' + args.exp_name + '/' + 'main.py')
    os.system('cp models/model_util.py checkpoints' + '/' + args.exp_name + '/' + 'model_util.py')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py')


def train(args, io):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed()%2**32
        np.random.seed(worker_seed)

    train_loader = DataLoader(
        ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                batch_size=args.batch_size, shuffle=True, drop_last=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(
        ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                    batch_size=args.test_batch_size, shuffle=True, drop_last=False, worker_init_fn=seed_worker)

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    model = Model(args, output_channels=40).to(device)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.decay)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr / 100)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=30, gamma=0.5)
    elif args.scheduler == 'multistep':
        scheduler = MultiStepLR(opt, [160, 210], gamma=0.1)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    io.cprint("The number of trainable parameters: %.6f(M)" % (count_parameters(model) / (1024 ** 2)))

    best_acc = 0.0
    best_bal_acc = 0.0
    best_z_acc = 0.0
    best_z_bal_acc = 0.0
    test_feat_l = []
    test_feat_g = []
    with LogWriter(logdir='checkpoints/%s/log/train' % args.exp_name) as writer:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            loss_cls = 0.0
            loss_dir_local = 0.0
            loss_dir_global = 0.0
            loss_orth_local = 0.0
            loss_orth_global = 0.0
            loss_feat_local = 0.0
            loss_feat_global = 0.0
            count = 0.0
            train_pred = []
            train_true = []

            for batch_data in tqdm(train_loader, total=len(train_loader)):
                data, label = batch_data
                batch_size = data.shape[0]
                data, label = data.to(device), label.to(device).squeeze()

                if args.train_rot == 'z':
                    trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to(device)
                elif args.train_rot == 'so3':
                    trot = Rotate(R=random_rotations(batch_size)).to(device)

                data = trot.transform_points(data)
                opt.zero_grad()

                loss, logits, loss_list = model(data, label, train=True)
                loss.backward()
                opt.step()

                preds = logits.max(dim=1)[1]
                train_true.append(label.detach().cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())

                count += batch_size
                train_loss += loss.item() * batch_size
                loss_cls += loss_list[0].item() * batch_size
                loss_dir_local += loss_list[1].item() * batch_size
                loss_dir_global += loss_list[2].item() * batch_size
                loss_orth_local += loss_list[3].item() * batch_size
                loss_orth_global += loss_list[4].item() * batch_size
                loss_feat_local += loss_list[5].item() * batch_size
                loss_feat_global += loss_list[6].item() * batch_size

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            train_acc = accuracy_score(train_true, train_pred)
            train_bal_acc = balanced_accuracy_score(train_true, train_pred)
            train_loss = train_loss / count
            loss_cls = loss_cls / count
            loss_dir_local = loss_dir_local / count
            loss_dir_global = loss_dir_global / count
            loss_orth_local = loss_orth_local / count
            loss_orth_global = loss_orth_global / count
            loss_feat_local = loss_feat_local / count
            loss_feat_global = loss_feat_global / count

            test_feat_l.append(loss_feat_local)
            test_feat_g.append(loss_feat_global)


            io.cprint('[Train %d, local loss dir: %.6f, loss orth: %.6f, loss feat: %.6f ]' % (epoch, loss_dir_local, loss_orth_local, loss_feat_local))
            io.cprint('[Train %d, global loss dir: %.6f, loss orth: %.6f, loss feat: %.6f ]' % (epoch, loss_dir_global, loss_orth_global, loss_feat_global))
            io.cprint('[Train %d, train loss: %.6f, cls loss: %.6f, cls acc: %.6f, cls bal acc: %.6f]' % (epoch, train_loss, loss_cls, train_acc, train_bal_acc))

            writer.add_scalar(tag='train_loss', step=epoch, value=train_loss)
            writer.add_scalar(tag='train_acc', step=epoch, value=train_acc)
            writer.add_scalar(tag='train_bal_acc', step=epoch, value=train_bal_acc)
            writer.add_scalar(tag='train_loss_dir_local', step=epoch, value=loss_dir_local)
            writer.add_scalar(tag='train_loss_dir_global', step=epoch, value=loss_dir_global)
            writer.add_scalar(tag='train_loss_orth_local', step=epoch, value=loss_orth_local)
            writer.add_scalar(tag='train_loss_orth_global', step=epoch, value=loss_orth_global)
            writer.add_scalar(tag='train_loss_feat_local', step=epoch, value=loss_feat_local)
            writer.add_scalar(tag='train_loss_feat_global', step=epoch, value=loss_feat_global)

            if args.scheduler == 'cos':
                scheduler.step()
            elif args.scheduler == 'step':
                if opt.param_groups[0]['lr'] > 1e-6:
                    scheduler.step()

            model.eval()
            test_loss = 0.0
            loss_cls = 0.0
            loss_dir_local = 0.0
            loss_dir_global = 0.0
            loss_orth_local = 0.0
            loss_orth_global = 0.0
            loss_feat_local = 0.0
            loss_feat_global = 0.0
            count = 0.0
            test_true = []
            test_pred = []
            with torch.no_grad():
                for batch_data in tqdm(test_loader, total=len(test_loader)):
                    data, label = batch_data
                    batch_size = data.shape[0]
                    data, label = data.to(device), label.to(device).squeeze()

                    trot = Rotate(R=random_rotations(batch_size)).to(device)
                    data = trot.transform_points(data)
                    loss, logits, loss_list = model(data, label, train=False)
                    count += batch_size
                    test_loss += loss.item() * batch_size
                    loss_cls += loss_list[0].item() * batch_size
                    loss_dir_local += loss_list[1].item() * batch_size
                    loss_dir_global += loss_list[2].item() * batch_size
                    loss_orth_local += loss_list[3].item() * batch_size
                    loss_orth_global += loss_list[4].item() * batch_size
                    loss_feat_local += loss_list[5].item() * batch_size
                    loss_feat_global += loss_list[6].item() * batch_size

                    preds = logits.max(dim=1)[1]
                    test_true.append(label.detach().cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)
                test_loss = test_loss / count
                test_acc = accuracy_score(test_true, test_pred)
                test_bal_acc = balanced_accuracy_score(test_true, test_pred)

                loss_cls = loss_cls / count
                loss_dir_local = loss_dir_local / count
                loss_dir_global = loss_dir_global / count
                loss_orth_local = loss_orth_local / count
                loss_orth_global = loss_orth_global / count
                loss_feat_local = loss_feat_local / count
                loss_feat_global = loss_feat_global / count

                writer.add_scalar(tag='test_loss', step=epoch, value=test_loss)
                writer.add_scalar(tag='test_acc', step=epoch, value=test_acc)
                writer.add_scalar(tag='test_bal_acc', step=epoch, value=test_bal_acc)
                writer.add_scalar(tag='test_loss_dir_local', step=epoch, value=loss_dir_local)
                writer.add_scalar(tag='test_loss_dir_global', step=epoch, value=loss_dir_global)
                writer.add_scalar(tag='test_loss_orth_local', step=epoch, value=loss_orth_local)
                writer.add_scalar(tag='test_loss_orth_global', step=epoch, value=loss_orth_global)
                writer.add_scalar(tag='test_loss_feat_local', step=epoch, value=loss_feat_local)
                writer.add_scalar(tag='test_loss_feat_global', step=epoch, value=loss_feat_global)

                if best_acc <= test_acc:
                    best_acc = test_acc
                    torch.save(model.state_dict(), 'checkpoints/%s/models/best_acc_model.pth' % args.exp_name)
                if best_bal_acc <= test_bal_acc:
                    best_bal_acc = test_bal_acc

                #io.cprint('[Test %d, local loss dir: %.6f \t loss orth: %.6f \t loss feat: %.6f ]' % (epoch, loss_dir_local, loss_orth_local, loss_feat_local))
                #io.cprint('[Test %d, global loss dir: %.6f \t loss orth: %.6f \t loss feat: %.6f ]' % (epoch, loss_dir_global, loss_orth_global, loss_feat_global))
                io.cprint('[Test %d, cls_loss: %.6f, acc: %.6f, bal_acc: %.6f \t Best acc: %.6f, Best balanced acc: %.6f]' % (epoch, loss_cls, test_acc, test_bal_acc, best_acc, best_bal_acc))

            if args.test_rot=="z":
                test_loss = 0.0
                loss_cls = 0.0
                count = 0.0
                test_true = []
                test_pred = []
                with torch.no_grad():
                    for batch_data in tqdm(test_loader, total=len(test_loader)):
                        data, label = batch_data
                        batch_size = data.shape[0]
                        data, label = data.to(device), label.to(device).squeeze()

                        trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to(device)
                        data = trot.transform_points(data)
                        loss, logits, loss_list = model(data, label, train=False)
                        count += batch_size
                        test_loss += loss.item() * batch_size
                        loss_cls += loss_list[0].item() * batch_size
                        preds = logits.max(dim=1)[1]
                        test_true.append(label.detach().cpu().numpy())
                        test_pred.append(preds.detach().cpu().numpy())

                    test_true = np.concatenate(test_true)
                    test_pred = np.concatenate(test_pred)
                    test_loss = test_loss / count
                    loss_cls = loss_cls / count
                    test_acc = accuracy_score(test_true, test_pred)
                    test_bal_acc = balanced_accuracy_score(test_true, test_pred)

                    if best_z_acc <= test_acc:
                        best_z_acc = test_acc
                        torch.save(model.state_dict(), 'checkpoints/%s/models/best_z_acc_model.pth' % args.exp_name)
                    if best_z_bal_acc <= test_bal_acc:
                        best_z_bal_acc = test_bal_acc


                    io.cprint('[Test %d, z rot: cls_loss: %.6f, acc: %.6f, bal_acc: %.6f \t Best acc: %.6f, Best balanced acc: %.6f]' % (epoch, loss_cls, test_acc, test_bal_acc, best_z_acc, best_z_bal_acc))


def val(args, test_loader, model, epoch, best_acc, best_bal_acc, logger, rot='z'):
    model.eval()
    test_loss = 0.0
    count = 0.0
    test_true = []
    test_pred = []

    loss_cls_v = 0.0
    loss_dir_local_v = 0.0
    loss_dir_global_v = 0.0
    loss_orth_local_v = 0.0
    loss_orth_global_v = 0.0
    loss_feat_local_v = 0.0
    loss_feat_global_v = 0.0

    with torch.no_grad():
        for batch_data in tqdm(test_loader, total=len(test_loader)):
            data, label = batch_data
            batch_size = data.shape[0]
            data, label = data.to(device), label.to(device).squeeze()

            trot = None
            if rot == 'z':
                trot = RotateAxisAngle(angle=torch.rand(data.shape[0]) * 360, axis="Z", degrees=True).to(device)
            elif rot == 'so3':
                trot = Rotate(R=random_rotations(batch_size)).to(device)
            if trot is not None:
                data = trot.transform_points(data)

            loss, logits, losslist = model(data, label, train=False)
            count += batch_size

            loss_cls_v += loss_list[0] * batch_size
            loss_dir_local_v += loss_list[1] * batch_size
            loss_dir_global_v += loss_list[2] * batch_size
            loss_orth_local_v += loss_list[3] * batch_size
            loss_orth_global_v += loss_list[4] * batch_size
            loss_feat_local_v += loss_list[5] * batch_size
            loss_feat_global_v += loss_list[6] * batch_size


            test_loss += loss.item() * batch_size
            logits = logits.detach().cpu()
            preds = logits.max(dim=1)[1]
            test_true.append(label.detach().cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_loss = test_loss / count
        test_acc = accuracy_score(test_true, test_pred)
        test_bal_acc = balanced_accuracy_score(test_true, test_pred)

        loss_cls_v = loss_cls_v / count
        loss_dir_local_v = loss_dir_local_v / count
        loss_dir_global_v = loss_dir_global_v / count
        loss_orth_local_v = loss_orth_local_v / count
        loss_orth_global_v = loss_orth_global_v / count
        loss_feat_local_v = loss_feat_local_v / count
        loss_feat_global_v = loss_feat_global_v / count

        logger.add_scalar(tag='val_z_loss', step=epoch, value=test_loss)
        logger.add_scalar(tag='val_z_acc', step=epoch, value=test_acc)
        logger.add_scalar(tag='val_z_bal_acc', step=epoch, value=test_bal_acc)

        if best_acc <= test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/best_acc_model.pth' % args.exp_name)
        if best_bal_acc <= test_bal_acc:
            best_bal_acc = test_bal_acc

        io.cprint('[Under %s: Best acc: %.6f \t Best balanced acc: %.6f]' % (rot, best_acc, best_bal_acc))
        return best_acc, best_bal_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp_cls',
                        help='Name of the experiment')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='training mode')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='Size of batch for test')
    parser.add_argument('--epochs', type=int, default=250,
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true',
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--decay', type=float, default=1e-6, metavar='N',
                        help='weight_decay in optimizer(default: 1e-6, 1e-4 if using sgd)')
    parser.add_argument('--scheduler', type=str, default='cos', choices=['cos', 'step', 'multistep'],
                        help='Scheduler to use, [cos, step, multistep]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--train_rot', type=str, default='z', choices=['aligned', 'z', 'so3'],
                        help='Rotation augmentation to input data')
    parser.add_argument('--test_rot', type=str, default='z', choices=['z', 'so3'],
                        help='test with z & so3 [z], or so3 only [so3]')

    parser.add_argument('--loss_feat_l', type=float, default=0.0, metavar='N',
                        help='invariant loss weight for local scale invariant feature')
    parser.add_argument('--loss_feat_g', type=float, default=0.0, metavar='N',
                        help='invariant loss weight for global scale invariant feature')
    parser.add_argument('--loss_dir_l', type=float, default=1.0, metavar='N',
                        help='equivariant loss weight for local scale orientation')
    parser.add_argument('--loss_dir_g', type=float, default=1.0, metavar='N',
                        help='equivariant loss weight for global scale orientation')
    parser.add_argument('--loss_orth_l', type=float, default=0.2, metavar='N',
                        help='loss for MSE between predicted direction')
    parser.add_argument('--loss_orth_g', type=float, default=0.1, metavar='N',
                        help='loss for MSE between predicted direction')
    parser.add_argument('--loss_cls', type=float, default=1.0, metavar='N',
                        help='Cross entropy loss for cls')

    parser.add_argument('--num_points', type=int, default=1024, metavar='N')
    parser.add_argument('--local_S', type=int, default=256,
                        help='Num of patches to generate (N_l)')
    parser.add_argument('--k_local', type=int, default=64, metavar='N',
                        help='Num of nearest neighbors to use for KNN when generating local patches (k_l)')
    parser.add_argument('--k_global', type=int, default=32, metavar='N',
                        help='Num of points sampled for global patches (N_g)')
    parser.add_argument('--k_local_layer', type=int, default=32, metavar='N',
                        help='Num of neighbors searching for edge conv in intra-learning (k_intra)')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                        help='drop out rate')
    parser.add_argument('--emb_dims', type=int, default=1024,
                        help='embedding dimension')

    parser.add_argument('--use_ball_query', action='store_true',
                        help='use ball query for generation of local sacle patches')
    parser.add_argument('--radius', type=float, default=0.2, metavar='N',
                        help='radius for ball query')
    parser.add_argument('--invar_block', type=int, default=2,
                        help='2 or 3')
    parser.add_argument('--combin_block', type=int, default=3,
                        help='2 or 3')

    args = parser.parse_args()
    _init_()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    else:
        io.cprint('Using CPU')

    torch.cuda.empty_cache()
    train(args, io)
    torch.cuda.empty_cache()
