from __future__ import absolute_import, division, print_function
from yacs.config import CfgNode as CN

_C = CN()
_C.name = "default"
_C.output_dir = "./output"
_C.proctitle = "title"
_C.valid_step = 5
_C.save_step = -1
_C.show_step = 20
_C.pin_memory = True
_C.input_size = (28, 28)
_C.color_space = "Gray"
_C.cpu_mode = False
_C.eval_mode = False
_C.seed_num = 0
_C.pretrained = None
_C.save_only_result = False
_C.ddp = False
_C.port = 12355
_C.rank = -1
_C.world_size = -1
_C.dp = False
_C.mixed_precision = False

_C.dataset = CN()
_C.dataset.dataset = "CIFAR100"
_C.dataset.num_classes = 100
_C.dataset.num_classes_1 = 0
_C.dataset.num_classes_2 = 0
_C.dataset.random_hierarchy = False
_C.dataset.root = "/data/hoyong"
_C.dataset.hier_type = "default"
_C.dataset.grouping_file = "none"

_C.backbone = CN()
_C.backbone.type = "LeNet5"
_C.backbone.in_features = 784
_C.backbone.in_channels = 1
_C.backbone.backbone_freeze = False

_C.pooling = CN()
_C.pooling.type = "Identity"
_C.reshape = CN()
_C.reshape.type = "Identity"
_C.classifier = CN()
_C.classifier.type = "FC"
_C.classifier.bias = True
_C.scaling = CN()
_C.scaling.type = "Identity"

_C.loss = CN()
_C.loss.loss_type = "CrossEntropyCustom"
_C.loss.lambda_coarse = 0.3
_C.loss.soft_beta = 0.3
_C.loss.lambda_warmup = False
_C.loss.lambda_decay = False
_C.loss.coarse_detach = False
_C.loss.center_loss_weight = 0.0
_C.loss.two_stage_mode = "joint"
_C.loss.dual_temp_coarse = 1.0
_C.loss.dual_temp_fine = 1.0
_C.loss.nc_reg_weight = 0.0

_C.loss.LDAM = CN()
_C.loss.LDAM.drw_epoch = 160
_C.loss.LDAM.max_margin = 0.5

_C.train = CN()
_C.train.batch_size = 32
_C.train.num_epochs = 60
_C.train.shuffle = True
_C.train.num_workers = 8
_C.train.tensorboard = CN()
_C.train.tensorboard.enable = True
_C.train.trainer = CN()
_C.train.trainer.type = "default"
_C.train.trainer.mixup_alpha = 0.2

_C.train.optimizer = CN()
_C.train.optimizer.type = "SGD"
_C.train.optimizer.base_lr = 0.001
_C.train.optimizer.momentum = 0.9
_C.train.optimizer.weight_decay = 1e-4

_C.train.lr_scheduler = CN()
_C.train.lr_scheduler.type = "multistep"
_C.train.lr_scheduler.lr_step = [40, 50]
_C.train.lr_scheduler.lr_factor = 0.1
_C.train.lr_scheduler.warm_epoch = 5
_C.train.lr_scheduler.cosine_decay_end = 0
_C.train.lr_scheduler.eta_min = 1e-4

_C.test = CN()
_C.test.batch_size = 32
_C.test.num_workers = 8
_C.test.model_file = ""

_C.transforms = CN()
_C.transforms.train_transforms = ("random_resized_crop", "random_horizontal_flip")
_C.transforms.test_transforms = ("shorter_resize_for_crop", "center_crop")
_C.transforms.process_detail = CN()
_C.transforms.process_detail.random_crop = CN()
_C.transforms.process_detail.random_crop.padding = 4
_C.transforms.process_detail.random_resized_crop = CN()
_C.transforms.process_detail.random_resized_crop.scale = (0.08, 1.0)
_C.transforms.process_detail.random_resized_crop.ratio = (0.75, 1.333333333)
_C.transforms.process_detail.normalize = CN()
_C.transforms.process_detail.normalize.mean = [0.286,]
_C.transforms.process_detail.normalize.std = [0.353,]
_C.transforms.process_detail.random_rotation = CN()
_C.transforms.process_detail.random_rotation.degrees = 15

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
