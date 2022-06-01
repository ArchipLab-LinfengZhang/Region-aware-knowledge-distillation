import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from pytorch_wavelets import DWTForward, DWTInverse
from . import networks
from packaging import version
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import numpy as np
from .vgg import *
import torchvision

cross_entropy_loss = torch.nn.CrossEntropyLoss()


def KL(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

def get_sample_ids(s, t, temperature=0.1):
    s = s.permute(0, 2, 3, 1).flatten(1, 2)
    t = t.permute(0, 2, 3, 1).flatten(1, 2)
    #   b x wh x c
    s_attention = torch.mean(torch.abs(s), [2])
    t_attention = torch.mean(torch.abs(t), [2])
    #score = s_attention + t_attention + torch.abs(s_attention - t_attention)
    score = t_attention
    score = score.view(-1)
    value, indexs = torch.sort(score, dim=0, descending=True)
    return [indexs[:64]]


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):

        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        batchSize = feat_q.shape[0]
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        self.opt.nce_includes_all_negatives_from_minibatch = False
        self.opt.nce_T = 0.07

        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        return loss.mean()


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[0]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)

                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids



class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt, teacher=None):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        if teacher is not None:
            self.teacher = teacher
        self.opt = opt
        if self.opt.distill is True:
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'kd', 'pip']
            visual_names_A = ['real_A', 'fake_B', 'rec_A', 'teacher_fake_B']
            visual_names_B = ['real_B', 'fake_A', 'rec_B', 'teacher_fake_A']
            if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
                visual_names_A.append('idt_B')
                visual_names_B.append('idt_A')
            self.visual_names = visual_names_A + visual_names_B
        else:
            # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
            self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
            # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
            visual_names_A = ['real_A', 'fake_B', 'rec_A']
            visual_names_B = ['real_B', 'fake_A', 'rec_B']
            if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
                visual_names_A.append('idt_B')
                visual_names_B.append('idt_A')
            self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if opt.distill:
            student_channel_size = [
                3, 3,
                self.opt.ngf, self.opt.ngf, self.opt.ngf,
                self.opt.ngf * 2, self.opt.ngf * 2, self.opt.ngf * 2,
                self.opt.ngf * 4, self.opt.ngf * 4, self.opt.ngf * 4, self.opt.ngf * 4,
                self.opt.ngf * 4, self.opt.ngf * 4, self.opt.ngf * 4,
                self.opt.ngf * 4, self.opt.ngf * 4, self.opt.ngf * 4, self.opt.ngf * 4,
                self.opt.ngf * 4,
                self.opt.ngf * 2, self.opt.ngf * 2, self.opt.ngf * 2,
                self.opt.ngf, self.opt.ngf, self.opt.ngf, self.opt.ngf,
                3, 3
            ]
            teacher_channel_size = [
                3, 3,
                self.teacher.opt.ngf, self.teacher.opt.ngf, self.teacher.opt.ngf,
                self.teacher.opt.ngf * 2, self.teacher.opt.ngf * 2, self.teacher.opt.ngf * 2,
                self.teacher.opt.ngf * 4, self.teacher.opt.ngf * 4, self.teacher.opt.ngf * 4, self.teacher.opt.ngf * 4,
                self.teacher.opt.ngf * 4, self.teacher.opt.ngf * 4, self.teacher.opt.ngf * 4,
                self.teacher.opt.ngf * 4, self.teacher.opt.ngf * 4, self.teacher.opt.ngf * 4, self.teacher.opt.ngf * 4,
                self.teacher.opt.ngf * 4,
                self.teacher.opt.ngf * 2, self.teacher.opt.ngf * 2, self.teacher.opt.ngf * 2,
                self.teacher.opt.ngf, self.teacher.opt.ngf, self.teacher.opt.ngf, self.teacher.opt.ngf,
                3, 3
            ]
        # for perceptual distillation
        if opt.lpip!=0:
            if opt.pretrain_model == "supervised_vgg":
                self.pretrain_model = torchvision.models.vgg19_bn(pretrained=True)
                #self.pretrain_model = vgg19_bn().cuda()
                #self.pretrain_model.load_state_dict(torch.load("/home/zhanglf/.cache/torch/hub/checkpoints/vgg19_bn-c79401a0.pth"))
            '''
            if opt.pretrain_model == "supervised_resnet":
                self.pretrain_model = torchvision.models.resnet50(pretrained=True)
            if opt.pretrain_model == "unsupervised_resnet":
                pass
            if opt.pretrain_model == 'supervised_vit':
                pass
            if opt.pretrain_model == 'unsupervised_vit':
                pass
            '''
        if opt.distill and opt.kd_contrastive:
            self.con_criterion = PatchNCELoss(self.opt)
            student_layer_list, teacher_layer_list = [], []
            for index in self.opt.choice_of_feature:
                student_layer_list.append(PatchSampleF(nc=student_channel_size[index]))
                teacher_layer_list.append(PatchSampleF(nc=student_channel_size[index], use_mlp=True))
            self.teacher_sampler_A = torch.nn.ModuleList(teacher_layer_list)
            self.student_sampler_A = torch.nn.ModuleList(student_layer_list)

            student_layer_list, teacher_layer_list = [], []
            for index in self.opt.choice_of_feature:
                student_layer_list.append(PatchSampleF(nc=student_channel_size[index]))
                teacher_layer_list.append(PatchSampleF(nc=student_channel_size[index], use_mlp=True))
            self.teacher_sampler_B = torch.nn.ModuleList(teacher_layer_list)
            self.student_sampler_B = torch.nn.ModuleList(student_layer_list)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            g_param_list = [self.netG_A.parameters(), self.netG_B.parameters()]
            if self.opt.distill and self.opt.kd_feature_distillation:
                g_param_list.append(self.feature_distillation_adaptation_layer_for_netG_A.parameters())
                g_param_list.append(self.feature_distillation_adaptation_layer_for_netG_B.parameters())
            if self.opt.distill and self.opt.kd_channel_attention_distillation:
                g_param_list.append(self.channel_attention_adaptation_for_netG_A.parameters())
                g_param_list.append(self.channel_attention_adaptation_for_netG_B.parameters())

            if self.opt.distill and self.opt.kd_gram:
                g_param_list.append(self.gram_adapt_A.parameters())
                g_param_list.append(self.gram_adapt_B.parameters())

            if self.opt.kd_contrastive and self.opt.distill:
                g_param_list.append(self.teacher_sampler_A.parameters())
                g_param_list.append(self.student_sampler_A.parameters())
                g_param_list.append(self.teacher_sampler_B.parameters())
                g_param_list.append(self.student_sampler_B.parameters())

            self.optimizer_G = torch.optim.Adam(itertools.chain(*g_param_list), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def get_wavelet_loss(self, a, b):
        loss = 0.0
        student_l, student_h = self.xfm(a)
        teacher_l, teacher_h = self.xfm(b)
        for index in range(len(student_h)):
            loss += torch.nn.functional.l1_loss(student_h[index], teacher_h[index])
        return loss

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.opt.distill:
            AtoB = self.opt.direction == 'AtoB'
            self.teacher.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.teacher.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.teacher.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A, record_feature=True)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B, record_feature=True)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        if self.opt.distill:
            self.loss_G += self.loss_kd + self.loss_pip
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        if self.opt.distill:
            with torch.no_grad():
                self.teacher.forward()
                self.teacher_fake_B = self.teacher.fake_B
                self.teacher_fake_A = self.teacher.fake_A

                student_feature_netG_A = [self.netG_A.module.feature_buffer[index] for index in
                                          self.opt.choice_of_feature]
                teacher_feature_netG_A = [self.teacher.netG_A.module.feature_buffer[index] for index in
                                          self.opt.choice_of_feature]
                student_feature_netG_B = [self.netG_B.module.feature_buffer[index] for index in
                                          self.opt.choice_of_feature]
                teacher_feature_netG_B = [self.teacher.netG_B.module.feature_buffer[index] for index in
                                          self.opt.choice_of_feature]

            self.loss_kd = 0.0
            layer_num = len(student_feature_netG_A)
            if self.opt.kd_contrastive != 0:
                for index in range(len(student_feature_netG_A)):
                    ids = get_sample_ids(student_feature_netG_A[index], teacher_feature_netG_A[index])
                    student_patch, ids = self.student_sampler_A[index]([student_feature_netG_A[index]], patch_ids=ids)
                    teacher_patch, ids = self.teacher_sampler_A[index]([teacher_feature_netG_A[index]], patch_ids=ids)
                    self.loss_kd += self.con_criterion(student_patch[0],
                                                       teacher_patch[0]) * self.opt.kd_contrastive * self.opt.kd_ratio / layer_num
                    
                    
                for index in range(len(student_feature_netG_B)):
                    ids = get_sample_ids(student_feature_netG_B[index], teacher_feature_netG_B[index])
                    student_patch, ids = self.student_sampler_B[index]([student_feature_netG_B[index]], patch_ids=ids)
                    teacher_patch, ids = self.teacher_sampler_B[index]([teacher_feature_netG_B[index]], patch_ids=ids)
                    self.loss_kd += self.con_criterion(student_patch[0],
                                                       teacher_patch[0]) * self.opt.kd_contrastive * self.opt.kd_ratio / layer_num

            student_feat = self.pretrain_model.features(self.fake_A)
            teacher_feat = self.pretrain_model.features(self.teacher.fake_A)
            self.loss_pip = F.l1_loss(student_feat, teacher_feat) * self.opt.lpip

            student_feat = self.pretrain_model.features(self.fake_B)
            teacher_feat = self.pretrain_model.features(self.teacher.fake_B)
            self.loss_pip += F.l1_loss(student_feat, teacher_feat) * self.opt.lpip


        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
