import argparse

from network import *

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument("-v", action="store_true", required=False)
parser.add_argument("-train_feat", dest="train_feat_fn", type=argparse.FileType('r'), required=True)
parser.add_argument("-train_target", dest="train_target_fn", type=argparse.FileType('r'), required=True)
parser.add_argument("-dev_feat", dest="dev_feat_fn", type=argparse.FileType('r'), required=True)
parser.add_argument("-dev_target", dest="dev_target_fn", type=argparse.FileType('r'), required=True)
parser.add_argument("-epochs", type=int, required=True)
parser.add_argument("-learnrate", type=float, required=True)
parser.add_argument("-nunits", dest="num_hidden_units", type=int, required=True)
parser.add_argument("-type", dest="problem_mode", required=True)
parser.add_argument("-hidden_act", dest="hidden_unit_activation", required=True)
parser.add_argument("-init_range", type=float, required=True)
parser.add_argument("-num_classes", dest="C", type=int, required=False)
parser.add_argument("-mb", dest="minibatch_size", default=0, type=int, required=False)
parser.add_argument("-nlayers", dest="num_hidden_layers", type=int, required=False)

args = parser.parse_args()

train_feat = np.loadtxt(args.train_feat_fn)
train_target = np.loadtxt(args.train_target_fn)
dev_feat = np.loadtxt(args.dev_feat_fn)
dev_target = np.loadtxt(args.dev_target_fn)

D = train_feat.shape[1] if train_feat.ndim == 2 else 1
# C = train_target.shape[1] if train_target.ndim == 2 else 1
C = args.C
# print(f"input size={D}, output size={C}")

network = Network(D, C, args.problem_mode, args.hidden_unit_activation,
                  args.learnrate, args.init_range, args.num_hidden_units,
                  args.num_hidden_layers, args.epochs, args.minibatch_size, args.v)
network.fit(train_feat, train_target, dev_feat, dev_target)
# print(network)

