from datetime import datetime
from random import shuffle

from scipy import signal
from torch.utils.tensorboard import SummaryWriter

from .discriminator import *
from .generator import *
from .ops import *
from ..datasets import *
from ..utils import *
from config import opts, device


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1DResBlock') != -1:
        print('Initializing weights of convresblock to 0.0, 0.02')
        for k, p in m.named_parameters():
            if 'weight' in k and 1 in k:
                p.data.normal_(0.0, 0.02)
    elif classname.find('Conv1d') != -1:
        print('Initialzing weight to 0.0, 0.02 for module: ', m)
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            print('bias to 0 for module: ', m)
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        print('Initializing FC weight to xavier uniform')
        nn.init.xavier_uniform_(m.weight.data)


def wsegan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1DResBlock') != -1:
        print('Initializing weights of convresblock to 0.0, 0.02')
        for k, p in m.named_parameters():
            if 'weight' in k and 1 in k:
                nn.init.xavier_uniform_(p.data)
    elif classname.find('Conv1d') != -1 or classname.find('ConvTranspose1d') != -1:
        print('Initialzing weight to XU for module: ', m)
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('Conv1DResBlock') == -1 and classname.find('Linear') != -1:
        print('Initializing FC weight to XU')
        nn.init.xavier_uniform_(m.weight.data)


# def z_dropout(m):
#     classname = m.__class__.__name__
#     if classname.find('Dropout') != -1:
#         # let it active
#         m.train()
#     else:
#         m.eval()

opts_important = f"epoch{opts.epoch}_batch_size{opts.batch_size}_dnorm_type{opts.dnorm_type}"


class SEGAN(Model):
    def __init__(self, opts, name='SEGAN', generator=None, discriminator=None):
        super(SEGAN, self).__init__(name)
        self.save_path = opts.save_path
        self.preemph = opts.preemph
        self.reg_loss = getattr(F, opts.reg_loss)
        if generator is None:
            # Build G and D
            self.G = Generator(1, opts.genc_fmaps, opts.gkwidth, opts.genc_poolings, opts.gdec_fmaps,
                               opts.gdec_kwidth, opts.gdec_poolings, z_dim=opts.z_dim, no_z=opts.no_z,
                               skip=(not opts.no_skip), bias=opts.bias,
                               skip_init=opts.skip_init, skip_type=opts.skip_type,
                               skip_merge=opts.skip_merge, skip_kwidth=opts.skip_kwidth)
        else:
            self.G = generator
        self.G.apply(weights_init)
        print('Generator: ', self.G)

        if discriminator is None:
            dkwidth = opts.gkwidth if opts.dkwidth is None else opts.dkwidth
            self.D = Discriminator(2, opts.denc_fmaps, dkwidth,
                                   poolings=opts.denc_poolings, pool_type=opts.dpool_type, pool_slen=opts.dpool_slen,
                                   norm_type=opts.dnorm_type, phase_shift=opts.phase_shift, sinc_conv=opts.sinc_conv)
        else:
            self.D = discriminator
        self.D.apply(weights_init)
        print('Discriminator: ', self.D)

        self.G, self.D = self.G.to(device), self.D.to(device)

        # create writer
        tensorboard_file_name = f"{os.path.join(self.save_path, 'train')}_SEGAN_{opts_important}_{datetime.time(datetime.now())}"
        self.writer = SummaryWriter(log_dir=tensorboard_file_name, comment=opts_important)

    def get_n_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def __sizeof__(self):
        return self.G.parameters.__sizeof__()

    def generate(self, inwav, z=None):
        self.G.eval()
        N = 16384
        x = np.zeros((1, 1, N))
        c_res = None
        slice_idx = torch.zeros(1)
        for beg_i in range(0, inwav.shape[2], N):
            if inwav.shape[2] - beg_i < N:
                length = inwav.shape[2] - beg_i
                pad = N - length
            else:
                length = N
                pad = 0
            if pad > 0:
                x[0, 0] = torch.cat((inwav[0, 0, beg_i:beg_i + length], torch.zeros(pad).to(device)), dim=0)
            else:
                x[0, 0] = inwav[0, 0, beg_i:beg_i + length]
            # x = torch.FloatTensor(x)
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            x = x.to(device)
            canvas_w, hall = self.infer_G(x, z=z, ret_hid=True)
            nums = [int(k.split('_')[1]) for k in hall.keys() if 'enc' in k and 'zc' not in k]
            g_c = hall['enc_{}'.format(max(nums))]
            if z is None and hasattr(self.G, 'z'):
                # if z was created inside G as first inference
                z = self.G.z
            if pad > 0:
                canvas_w = canvas_w[0, 0, :-pad]
            canvas_w = canvas_w.data.cpu().numpy().squeeze()
            c_res = canvas_w if c_res is None else np.concatenate((c_res, canvas_w))
            slice_idx += 1
        # de-emph
        c_res = de_emphasize(c_res, self.preemph)
        return c_res, g_c

    def discriminate(self, cwav, nwav):
        self.D.eval()
        d_in = torch.cat((cwav, nwav), dim=1)
        d_veredict, _ = self.D(d_in)
        return d_veredict

    def infer_G(self, nwav, cwav=None, z=None, ret_hid=False):
        if ret_hid:
            Genh, hall = self.G(nwav, z=z, ret_hid=ret_hid)
            return Genh, hall
        else:
            Genh = self.G(nwav, z=z, ret_hid=ret_hid)
            return Genh

    def infer_D(self, x_, ref):
        D_in = torch.cat((x_, ref), dim=1)
        return self.D(D_in)

    def gen_train_samples(self, clean_samples, noisy_samples, z_sample, iteration=None):
        if z_sample is not None:
            canvas_w = self.infer_G(noisy_samples, clean_samples, z=z_sample)
        else:
            canvas_w = self.infer_G(noisy_samples, clean_samples)
        sample_dif = noisy_samples - clean_samples
        # sample wavs
        for m in range(noisy_samples.size(0)):
            m_canvas = de_emphasize(canvas_w[m, 0].cpu().data.numpy(), self.preemph)
            print('w{} max: {} min: {}'.format(m, m_canvas.max(), m_canvas.min()))
            wavfile.write(os.path.join(self.save_path, 'sample_{}-{}.wav'.format(iteration, m)), int(16e3), m_canvas)
            m_clean = de_emphasize(clean_samples[m, 0].cpu().data.numpy(), self.preemph)
            m_noisy = de_emphasize(noisy_samples[m, 0].cpu().data.numpy(), self.preemph)
            m_dif = de_emphasize(sample_dif[m, 0].cpu().data.numpy(), self.preemph)
            m_gtruth_path = os.path.join(self.save_path, 'gtruth_{}.wav'.format(m))
            if not os.path.exists(m_gtruth_path):
                wavfile.write(os.path.join(self.save_path, 'gtruth_{}.wav'.format(m)), int(16e3), m_clean)
                wavfile.write(os.path.join(self.save_path, 'noisy_{}.wav'.format(m)), int(16e3), m_noisy)
                wavfile.write(os.path.join(self.save_path, 'dif_{}.wav'.format(m)), int(16e3), m_dif)

    def build_optimizers(self, opts):
        if opts.opt == 'rmsprop':
            Gopt = optim.RMSprop(self.G.parameters(), lr=opts.g_lr)
            Dopt = optim.RMSprop(self.D.parameters(), lr=opts.d_lr)
        elif opts.opt == 'adam':
            Gopt = optim.Adam(self.G.parameters(), lr=opts.g_lr, betas=(0, 0.9))
            Dopt = optim.Adam(self.D.parameters(), lr=opts.d_lr, betas=(0, 0.9))
        else:
            raise ValueError('Unrecognized optimizer {}'.format(opts.opt))
        return Gopt, Dopt

    def train(self, opts, dloader, criterion, l1_init, l1_dec_step, l1_dec_epoch, log_freq, va_dloader=None):
        """ Train the SEGAN """

        # # create writer
        # tensorboard_file_name = f"{os.path.join(self.save_path, 'train')}_SEGAN_{opts_important}_{datetime.time(datetime.now())}"
        # self.writer = SummaryWriter(log_dir=tensorboard_file_name, comment=opts_important)

        # Build the optimizers
        Gopt, Dopt = self.build_optimizers(opts)

        # attach opts to models so that they are saved altogether in ckpts
        self.G.optim = Gopt
        self.D.optim = Dopt

        # Build savers for end of epoch, storing up to 3 epochs each
        eoe_g_saver = Saver(self.G, opts.save_path, max_ckpts=3, optimizer=self.G.optim, prefix='EOE_G-')
        eoe_d_saver = Saver(self.D, opts.save_path, max_ckpts=3, optimizer=self.D.optim, prefix='EOE_D-')
        # num_batches = len(dloader)
        l1_weight = l1_init
        iteration = 1
        timings, evals, noisy_evals = [], {}, {}
        noisy_samples, clean_samples, z_sample = None, None, None
        patience = opts.patience
        best_val_obj = 0
        # acumulator for exponential avg of valid curve
        # acum_val_obj = 0
        # make label tensor
        label = torch.ones(opts.batch_size)
        label = label.to(device)

        for epoch in range(1, opts.epoch + 1):
            beg_t = timeit.default_timer()
            self.G.train()
            self.D.train()
            for bidx, batch in enumerate(dloader, start=1):
                if epoch >= l1_dec_epoch:
                    if l1_weight > 0:
                        l1_weight -= l1_dec_step
                        # ensure it is 0 if it goes < 0
                        l1_weight = max(0, l1_weight)
                sample = batch
                if len(sample) == 4:
                    uttname, clean, noisy, slice_idx = batch
                else:
                    raise ValueError('Returned {} elements per sample?'.format(len(sample)))
                clean, noisy = clean.unsqueeze(1), noisy.unsqueeze(1)
                label.resize_(clean.size(0)).fill_(1)
                clean, noisy = clean.to(device), noisy.to(device)
                if noisy_samples is None:
                    noisy_samples, clean_samples = noisy[:20, :, :].contiguous(), clean[:20, :, :].contiguous()
                ## Discriminator critic train
                Genh, d_fake, d_fake_loss, d_real, d_real_loss = \
                    self.train_critic_once(Dopt, clean, criterion, label, noisy)

                # # improve discriminator and prevent over-train problem.
                # # Not: generator getting worse after long time
                # d_loss_critic = self.critic_iters / epoch  # 10  # dynamic discriminator loss control.
                # if d_fake_loss > d_loss_critic or d_real_loss > d_loss_critic or not epoch-1 % self.critic_iters:
                #     # for _ in range(1, self.critic_iters + 1):
                #     for _ in range(1, self.critic_iters):
                #         Genh, d_fake, d_fake_loss, d_real, d_real_loss = \
                #             self.train_critic_once(Dopt, clean, criterion, label, noisy)

                # d_loss = d_fake_loss + d_real_loss

                ## (3) G real update
                Gopt.zero_grad()
                lab = label.fill_(1)
                d_fake_, _ = self.infer_D(Genh, noisy)
                g_adv_loss = criterion(d_fake_.view(-1), lab)
                # g_l1_loss = l1_weight * F.l1_loss(Genh, clean)
                g_l1_loss = l1_weight * self.reg_loss(Genh, clean)
                g_loss = g_adv_loss + g_l1_loss
                g_loss.backward()
                Gopt.step()

                ## report
                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                beg_t = timeit.default_timer()
                if z_sample is None and not self.G.no_z:
                    # capture sample now that we know shape after first
                    # inference
                    z_sample = self.G.z[:20, :, :].contiguous()
                    print('z_sample size: ', z_sample.size())
                    z_sample = z_sample.to(device)
                if bidx % log_freq == 0 or bidx >= len(dloader):
                    d_real_loss_v = d_real_loss.cpu().item()
                    d_fake_loss_v = d_fake_loss.cpu().item()
                    g_adv_loss_v = g_adv_loss.cpu().item()
                    g_l1_loss_v = g_l1_loss.cpu().item()
                    log = '(Iter {}) Batch {}/{} (Epoch {}) d_real:{:.4f}, d_fake:{:.4f}, ' \
                        .format(iteration, bidx, len(dloader), epoch, d_real_loss_v, d_fake_loss_v)
                    log += 'g_adv:{:.4f}, g_l1:{:.4f} l1_w: {:.2f}, btime: {:.4f} s, mbtime: {:.4f} s' \
                        .format(g_adv_loss_v, g_l1_loss_v, l1_weight, timings[-1], np.mean(timings))
                    print(log)
                    self.writer.add_scalar('D_real', d_real_loss_v, iteration)
                    self.writer.add_scalar('D_fake', d_fake_loss_v, iteration)
                    self.writer.add_scalar('G_adv', g_adv_loss_v, iteration)
                    # TODO: full sinc-conv ValueError: autodetected range of [nan, nan] is not finite
                    # (Iter 8968) Batch 300 / 788(Epoch 12) d_real: nan, d_fake: nan, g_adv: nan, g_l1: nan l1_w: 100.00, btime: 4.3358 s, mbtime: 4.5765 s
                    self.writer.add_scalar('G_l1', g_l1_loss_v, iteration)

                    self.writer.add_histogram('D_fake__hist', d_fake_.cpu().data, iteration, bins='sturges')
                    self.writer.add_histogram('D_fake_hist', d_fake.cpu().data, iteration, bins='sturges')
                    self.writer.add_histogram('D_real_hist', d_real.cpu().data, iteration, bins='sturges')
                    self.writer.add_histogram('Gz', Genh.cpu().data, iteration, bins='sturges')
                    self.writer.add_histogram('clean', clean.cpu().data, iteration, bins='sturges')
                    self.writer.add_histogram('noisy', noisy.cpu().data, iteration, bins='sturges')

                    # get D and G weights and plot their norms by layer and
                    # global
                    def model_weights_norm(model, total_name):
                        total_GW_norm = 0
                        for k, v in model.named_parameters():
                            if 'weight' in k:
                                W = v.data
                                W_norm = torch.norm(W)
                                self.writer.add_scalar('{}_Wnorm'.format(k), W_norm, iteration)
                                total_GW_norm += W_norm
                        self.writer.add_scalar('{}_Wnorm'.format(total_name), total_GW_norm, iteration)

                    model_weights_norm(self.G, 'Gtotal')
                    model_weights_norm(self.D, 'Dtotal')
                    if not opts.no_train_gen:
                        # canvas_w = self.G(noisy_samples, z=z_sample)
                        self.gen_train_samples(clean_samples, noisy_samples, z_sample, iteration=iteration)
                iteration += 1

            if va_dloader is not None:
                if len(noisy_evals) == 0:
                    evals_, noisy_evals_ = self.evaluate(opts, va_dloader, log_freq, do_noisy=True)
                    for k, v in noisy_evals_.items():
                        if k not in noisy_evals:
                            noisy_evals[k] = []
                        noisy_evals[k] += v
                        self.writer.add_scalar('noisy-{}'.format(k), noisy_evals[k][-1], epoch)
                else:
                    evals_ = self.evaluate(opts, va_dloader, log_freq, do_noisy=False)
                for k, v in evals_.items():
                    if k not in evals:
                        evals[k] = []
                    evals[k] += v
                    self.writer.add_scalar('Genh-{}'.format(k), evals[k][-1], epoch)
                val_obj = evals['covl'][-1] + evals['pesq'][-1] + evals['ssnr'][-1]
                self.writer.add_scalar('Genh-val_obj', val_obj, epoch)
                if val_obj > best_val_obj:
                    print('Val obj (COVL + SSNR + PESQ) improved {} -> {}'.format(best_val_obj, val_obj))
                    best_val_obj = val_obj
                    patience = opts.patience
                    # save models with true valid curve is minimum
                    self.G.save(self.save_path, iteration, True)
                    self.D.save(self.save_path, iteration, True)
                else:
                    patience -= 1
                    print('Val loss did not improve. Patience{}/{}'.format(patience, opts.patience))
                    if patience <= 0:
                        print('STOPPING SEGAN TRAIN: OUT OF PATIENCE.')
                        break

            # save models in end of epoch with EOE savers
            self.G.save(self.save_path, iteration, saver=eoe_g_saver)
            self.D.save(self.save_path, iteration, saver=eoe_d_saver)

    def train_critic_once(self, Dopt, clean, criterion, label, noisy):
        ## (1) D real update
        Dopt.zero_grad()
        total_d_fake_loss, total_d_real_loss = 0, 0
        Genh = self.infer_G(noisy, clean)
        lab = label
        d_real, _ = self.infer_D(clean, noisy)
        d_real_loss = criterion(d_real.view(-1), lab)
        d_real_loss.backward()
        total_d_real_loss += d_real_loss
        ## (2) D fake update
        d_fake, _ = self.infer_D(Genh.detach(), noisy)
        lab = label.fill_(0)
        d_fake_loss = criterion(d_fake.view(-1), lab)
        d_fake_loss.backward()
        total_d_fake_loss += d_fake_loss
        Dopt.step()
        return Genh, d_fake, d_fake_loss, d_real, d_real_loss

    def save(self, save_path, step, best_val=False, saver=None):
        super(SEGAN, self).save(save_path, step, best_val=False, saver=None)

    def evaluate(self, opts, dloader, log_freq, do_noisy=False, max_samples=1):
        """ Objective evaluation with PESQ, SSNR, COVL, CBAK and CSIG """
        if not self.G is None:
            self.G.eval()
        if not self.D is None:
            self.D.eval()
        evals = {'pesq': [], 'ssnr': [], 'csig': [],
                 'cbak': [], 'covl': []}
        # pesqs, ssnrs = [], []
        if do_noisy:
            noisy_evals = {'pesq': [], 'ssnr': [], 'csig': [],
                           'cbak': [], 'covl': []}
            npesqs, nssnrs = [], []
        if not hasattr(self, 'pool'):
            self.pool = mp.Pool(opts.eval_workers)
        # total_s = 0
        # timings = []
        with torch.no_grad():
            # going over dataset ONCE
            for bidx, batch in enumerate(dloader, start=1):
                sample = batch
                if len(sample) == 4:
                    uttname, clean, noisy, slice_idx = batch
                else:
                    raise ValueError('Returned {} elements per sample?'.format(len(sample)))
                noisy = noisy.unsqueeze(1)
                clean, noisy = clean.to(device), noisy.to(device)
                Genh = self.infer_G(noisy).squeeze(1)
                clean_npy = clean.cpu().data.numpy()
                Genh_npy = Genh.cpu().data.numpy()
                clean_npy = np.apply_along_axis(de_emphasize, 0, clean_npy, self.preemph)
                Genh_npy = np.apply_along_axis(de_emphasize, 0, Genh_npy, self.preemph)
                beg_t = timeit.default_timer()
                if do_noisy:
                    noisy_npy = noisy.cpu().data.numpy()
                    noisy_npy = np.apply_along_axis(de_emphasize, 0, noisy_npy, self.preemph)
                    args = [(clean_npy[i], Genh_npy[i], noisy_npy[i]) for i in range(clean.size(0))]
                else:
                    args = [(clean_npy[i], Genh_npy[i], None) for i in range(clean.size(0))]
                map_ret = self.pool.map(composite_helper, args)  # error, None type is not iteratable

                end_t = timeit.default_timer()
                print('Time to process eval with {} samples : {} s'.format(clean.size(0), end_t - beg_t))
                if bidx >= max_samples: break

            def fill_ret_dict(ret_dict, in_dict):
                for k, v in in_dict.items():
                    ret_dict[k].append(v)

            if do_noisy:
                for eval_, noisy_eval_ in map_ret:
                    fill_ret_dict(evals, eval_)
                    fill_ret_dict(noisy_evals, noisy_eval_)
                return evals, noisy_evals
            else:
                for eval_ in map_ret:
                    fill_ret_dict(evals, eval_)
                return evals


# Whisper SEGAN
# This is the samples page of the Whisper SEGAN project, where whispered voices suffering from impairments after a total laryngectomy are reconstructed with intonation and more expressiveness than a regression baseline by means of a speech enhancement GAN. The code for this project can found

class WSEGAN(SEGAN):
    """Whisper SEGAN"""
    def __init__(self, opts, name='WSEGAN', generator=None, discriminator=None):
        self.lbd = 1
        self.misalign_pair = opts.misalign_pair
        self.interf_pair = opts.interf_pair
        self.pow_weight = opts.pow_weight
        self.vanilla_gan = opts.vanilla_gan
        self.n_fft = opts.n_fft
        super(WSEGAN, self).__init__(opts, name, None, None)
        self.G.apply(wsegan_weights_init)
        self.D.apply(wsegan_weights_init)

        self.G, self.D = self.G.to(device), self.D.to(device)

        # create writer
        tensorboard_file_name = f"{os.path.join(self.save_path, 'train')}_WSEGAN_{opts_important}_{datetime.time(datetime.now())}"
        self.writer = SummaryWriter(log_dir=tensorboard_file_name, comment=opts_important)

    def get_n_params(self):
        super(WSEGAN, self).get_n_params()

    def sample_dloader(self, dloader):
        sample = next(dloader.__iter__())
        batch = sample
        uttname, clean, noisy, slice_idx = batch
        clean, noisy = clean.unsqueeze(1).to(device), noisy.unsqueeze(1).to(device)
        slice_idx = slice_idx.to(device)
        return uttname, clean, noisy, slice_idx

    def infer_G(self, nwav, cwav=None, z=None, ret_hid=False):
        return self.G(nwav, z=z, ret_hid=ret_hid)

    def train(self, opts, dloader, criterion, l1_init, l1_dec_step, l1_dec_epoch, log_freq, va_dloader=None):
        """ Train the SEGAN """

        # Build the optimizers
        Gopt, Dopt = self.build_optimizers(opts)

        # attach opts to models so that they are saved altogether in ckpts
        self.G.optim = Gopt
        self.D.optim = Dopt

        # Build savers for end of epoch, storing up to 3 epochs each
        eoe_g_saver = Saver(self.G, opts.save_path, max_ckpts=3, optimizer=self.G.optim, prefix='EOE_G-')
        eoe_d_saver = Saver(self.D, opts.save_path, max_ckpts=3, optimizer=self.D.optim, prefix='EOE_D-')
        num_batches = len(dloader)
        l1_weight = l1_init
        iteration = 1
        timings = []
        evals = {}
        noisy_evals = {}
        noisy_samples = None
        clean_samples = None
        z_sample = None
        patience = opts.patience
        best_val_obj = np.inf

        for iteration in range(1, opts.epoch * len(dloader) + 1):
            beg_t = timeit.default_timer()
            uttname, clean, noisy, slice_idx = self.sample_dloader(dloader)
            bsz = clean.size(0)
            clean, noisy = clean.to(device), noisy.to(device)
            ## discriminator update
            if self.vanilla_gan:
                cost = F.binary_cross_entropy_with_logits
            else:
                cost = F.mse_loss

            Genh, bsz, cost, d_loss = self.train_critic_once(Dopt, cost, bsz, clean, noisy)
            # # Not: generator getting worse
            # if not iteration-1 % self.critic_iters:  # prevent over-train problem.
            #     for _ in range(1, self.critic_iters):
            #         Genh, bsz, cost, d_loss = self.train_critic_once(Dopt, cost, bsz, clean, noisy)

            ## generator
            Gopt.zero_grad()
            d_fake_, _ = self.infer_D(Genh, noisy)
            g_adv_loss = cost(d_fake_, torch.ones(d_fake_.size()).to(device))

            # POWER Loss -----------------------------------
            # make stft of gtruth
            clean_stft = torch.stft(clean.squeeze(1),
                                    n_fft=min(clean.size(-1), self.n_fft),
                                    hop_length=160,
                                    win_length=320,
                                    normalized=True)
            clean_mod = torch.norm(clean_stft, 2, dim=3)
            clean_mod_pow = 10 * torch.log10(clean_mod ** 2 + 10e-20)
            Genh_stft = torch.stft(Genh.squeeze(1),
                                   n_fft=min(Genh.size(-1), self.n_fft),
                                   hop_length=160,
                                   win_length=320, normalized=True)
            Genh_mod = torch.norm(Genh_stft, 2, dim=3)
            Genh_mod_pow = 10 * torch.log10(Genh_mod ** 2 + 10e-20)
            pow_loss = self.pow_weight * F.l1_loss(Genh_mod_pow, clean_mod_pow)
            G_cost = g_adv_loss + pow_loss
            if l1_weight > 0:
                # look for additive files to build batch mask
                mask = torch.zeros(bsz, 1, Genh.size(2))
                mask = mask.to(device)
                for utt_i, uttn in enumerate(uttname):
                    if 'additive' in uttn:
                        mask[utt_i, 0, :] = 1.
                den_loss = l1_weight * F.l1_loss(Genh * mask, clean * mask)
                G_cost += den_loss
            else:
                den_loss = torch.zeros(1)
            G_cost.backward()
            Gopt.step()
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            if noisy_samples is None:
                noisy_samples = noisy[:20, :, :].contiguous()
                clean_samples = clean[:20, :, :].contiguous()
            if z_sample is None and not self.G.no_z:
                # capture sample now that we know shape after first
                # inference
                z_sample = self.G.z[:20, :, :].contiguous()
                print('z_sample size: ', z_sample.size())
                z_sample = z_sample.to(device)
            if iteration % log_freq == 0:
                log = 'Iter {}/{} ({} bpe) d_loss:{:.4f}, g_loss: {:.4f}, pow_loss: {:.4f}, den_loss: {:.4f} ' \
                    .format(iteration, len(dloader) * opts.epoch, len(dloader), d_loss.item(), G_cost.item(),
                            pow_loss.item(), den_loss.item())

                log += 'btime: {:.4f} s, mbtime: {:.4f} s'.format(timings[-1], np.mean(timings))
                print(log)
                ## tensorboard
                self.writer.add_scalar('D_loss', d_loss.item(), iteration)
                self.writer.add_scalar('G_loss', G_cost.item(), iteration)
                self.writer.add_scalar('G_adv_loss', g_adv_loss.item(), iteration)
                self.writer.add_scalar('G_pow_loss', pow_loss.item(), iteration)
                self.writer.add_histogram('clean_mod_pow', clean_mod_pow.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('Genh_mod_pow', Genh_mod_pow.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('Gz', Genh.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('clean', clean.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('noisy', noisy.cpu().data, iteration, bins='sturges')
                if hasattr(self.G, 'skips'):
                    for skip_id, alpha in self.G.skips.items():
                        skip = alpha[0]
                        if skip.skip_type == 0:
                            self.writer.add_histogram('skip_alpha_{}'.format(skip_id), skip.skip_k.data,
                                                      iteration, bins='sturges')

                # get D and G weights and plot their norms by layer and global
                def model_weights_norm(model, total_name):
                    total_GW_norm = 0
                    for k, v in model.named_parameters():
                        if 'weight' in k:
                            W = v.data
                            W_norm = torch.norm(W)
                            self.writer.add_scalar('{}_Wnorm'.format(k), W_norm, iteration)
                            total_GW_norm += W_norm
                    self.writer.add_scalar('{}_Wnorm'.format(total_name), total_GW_norm, iteration)

                model_weights_norm(self.G, 'Gtotal')
                model_weights_norm(self.D, 'Dtotal')
                if not opts.no_train_gen:
                    self.gen_train_samples(clean_samples, noisy_samples, z_sample, iteration=iteration)
                # BEWARE: There is no evaluation in Whisper SEGAN (WSEGAN)
                # TODO: Perhaps add some MCD/F0 RMSE metric
            if iteration % len(dloader) == 0:
                # save models in end of epoch with EOE savers
                self.G.save(self.save_path, iteration, saver=eoe_g_saver)
                self.D.save(self.save_path, iteration, saver=eoe_d_saver)

    def train_critic_once(self, Dopt, cost, bsz, clean, noisy):
        # TODO: Check the correction. There is may be a mistake!
        # grads
        Dopt.zero_grad()
        d_real, _ = self.infer_D(clean, noisy)
        rl_lab = torch.ones(d_real.size()).to(device)
        d_real_loss = cost(d_real, rl_lab)
        Genh = self.infer_G(noisy, clean)
        fake = Genh.detach()
        d_fake, _ = self.infer_D(fake, noisy)
        fk_lab = torch.zeros(d_fake.size()).to(device)
        d_fake_loss = cost(d_fake, fk_lab)
        d_weight = 0.5  # count only d_fake and d_real
        d_loss = d_fake_loss + d_real_loss
        if self.misalign_pair:
            clean_shuf = list(torch.chunk(clean, clean.size(0), dim=0))
            shuffle(clean_shuf)
            clean_shuf = torch.cat(clean_shuf, dim=0)
            d_fake_shuf, _ = self.infer_D(clean, clean_shuf)
            d_fake_shuf_loss = cost(d_fake_shuf, fk_lab)
            d_weight = 1 / 3  # count 3 components now
            d_loss += d_fake_shuf_loss
        if self.interf_pair:
            # put interferring squared signals with random amplitude and
            # freq as fake signals mixed with clean data
            # TODO: Beware with hard-coded values! possibly improve this
            freqs = [250, 1000, 4000]
            amps = [0.01, 0.05, 0.1, 1]
            bsz = clean.size(0)
            squares = []
            t = np.linspace(0, 2, 32000)
            for _ in range(bsz):
                f_ = random.choice(freqs)
                a_ = random.choice(amps)
                sq = a_ * signal.square(2 * np.pi * f_ * t)
                sq = sq[:clean.size(-1)].reshape((1, -1))
                squares.append(torch.FloatTensor(sq))
            squares = torch.cat(squares, dim=0).unsqueeze(1)
            squares = squares.to(device)
            interf = clean + squares
            d_fake_inter, _ = self.infer_D(interf, noisy)
            d_fake_inter_loss = cost(d_fake_inter, fk_lab)
            d_weight = 1 / 4  # count 4 components in d loss now
            d_loss += d_fake_inter_loss
        d_loss = d_weight * d_loss
        d_loss.backward()
        Dopt.step()
        return Genh, bsz, cost, d_loss

    def save(self, save_path, step, best_val=False, saver=None):
        super(WSEGAN, self).save(save_path, step, best_val=False, saver=None)

    def generate(self, inwav, z=None):
        # simplified inference without chunking
        # if self.z_dropout:
        #    self.G.apply(z_dropout)
        # else:
        self.G.eval()
        ori_len = inwav.size(2)
        p_wav = make_divN(inwav.transpose(1, 2), 1024).transpose(1, 2)
        c_res, hall = self.infer_G(p_wav, z=z, ret_hid=True)
        c_res = c_res[0, 0, :ori_len].cpu().data.numpy()
        c_res = de_emphasize(c_res, self.preemph)
        return c_res, hall


class AEWSEGAN(WSEGAN):
    """ Auto-Encoder model """

    def __init__(self, opts, name='AEWSEGAN', generator=None, discriminator=None):
        super(AEWSEGAN, self).__init__(opts, name=name, generator=generator, discriminator=discriminator)
        # delete discriminator
        self.D = None
        self.l1_loss = opts.reg_loss

        self.G = self.G.to(device)
        # self.D = self.D.to(device)

        # create writer
        tensorboard_file_name = f"{os.path.join(self.save_path, 'train')}_AEWSEGAN_{opts_important}_{datetime.time(datetime.now())}"
        self.writer = SummaryWriter(log_dir=tensorboard_file_name, comment=opts_important)

    def get_n_params(self):
        super(AEWSEGAN, self).get_n_params()

    def train(self, opts, dloader, criterion, l1_init, l1_dec_step, l1_dec_epoch, log_freq, va_dloader=None):
        """ Train the SEGAN """
        if opts.opt == 'rmsprop':
            Gopt = optim.RMSprop(self.G.parameters(), lr=opts.g_lr)
        elif opts.opt == 'adam':
            Gopt = optim.Adam(self.G.parameters(), lr=opts.g_lr, betas=(0.5, 0.9))
        else:
            raise ValueError('Unrecognized optimizer {}'.format(opts.opt))

        # attach opts to models so that they are saved altogether in ckpts
        self.G.optim = Gopt

        # Build savers for end of epoch, storing up to 3 epochs each
        eoe_g_saver = Saver(self.G, opts.save_path, max_ckpts=3, optimizer=self.G.optim, prefix='EOE_G-')
        # num_batches = len(dloader)
        # l2_weight = l1_init
        iteration = 1
        timings = []
        evals, noisy_evals = {}, {}
        noisy_samples, clean_samples, z_sample = None, None, None
        # patience = opts.patience
        best_val_obj = np.inf
        # acumulator for exponential avg of valid curve
        # acum_val_obj = 0
        G = self.G

        for iteration in range(1, opts.epoch * len(dloader) + 1):
            beg_t = timeit.default_timer()
            uttname, clean, noisy, slice_idx = self.sample_dloader(dloader)
            # bsz = clean.size(0)
            Genh = self.infer_G(noisy, clean)
            if self.l1_loss:
                loss = F.l1_loss(Genh, clean)
            else:
                loss = F.mse_loss(Genh, clean)

            Gopt.zero_grad()
            loss.backward()
            Gopt.step()
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            # beg_t = timeit.default_timer()
            if noisy_samples is None:
                noisy_samples = noisy[:20, :, :].contiguous()
                clean_samples = clean[:20, :, :].contiguous()
            if z_sample is None and not G.no_z:
                # capture sample now that we know shape after first
                # inference
                z_sample = G.z[:20, :, :].contiguous()
                print('z_sample size: ', z_sample.size())
                z_sample = z_sample.to(device)
            if iteration % log_freq == 0:
                # POWER Loss (not used to backward) -----------------------------------
                # make stft of gtruth
                clean_stft = torch.stft(clean.squeeze(1), n_fft=min(clean.size(-1), self.n_fft),
                                        hop_length=160, win_length=320, normalized=True)
                clean_mod = torch.norm(clean_stft, 2, dim=3)
                clean_mod_pow = 10 * torch.log10(clean_mod ** 2 + 10e-20)
                Genh_stft = torch.stft(Genh.detach().squeeze(1), n_fft=min(Genh.size(-1), self.n_fft),
                                       hop_length=160, win_length=320, normalized=True)
                Genh_mod = torch.norm(Genh_stft, 2, dim=3)
                Genh_mod_pow = 10 * torch.log10(Genh_mod ** 2 + 10e-20)
                pow_loss = F.l1_loss(Genh_mod_pow, clean_mod_pow)
                log = 'Iter {}/{} ({} bpe) g_l2_loss:{:.4f}, pow_loss: {:.4f}, '.format(iteration,
                                                                                        len(dloader) * opts.epoch,
                                                                                        len(dloader), loss.item(),
                                                                                        pow_loss.item())

                log += 'btime: {:.4f} s, mbtime: {:.4f} s'.format(timings[-1], np.mean(timings))
                ## tensorboard
                print(log)
                self.writer.add_scalar('g_l2/l1_loss', loss.item(), iteration)
                self.writer.add_scalar('G_pow_loss', pow_loss.item(), iteration)
                self.writer.add_histogram('clean_mod_pow', clean_mod_pow.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('Genh_mod_pow', Genh_mod_pow.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('Gz', Genh.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('clean', clean.cpu().data, iteration, bins='sturges')
                self.writer.add_histogram('noisy', noisy.cpu().data, iteration, bins='sturges')
                if hasattr(G, 'skips'):
                    for skip_id, alpha in G.skips.items():
                        skip = alpha[0]
                        if skip.skip_type == 0:
                            self.writer.add_histogram('skip_alpha_{}'.format(skip_id),
                                                      skip.skip_k.data, iteration, bins='sturges')

                # # get D and G weights and plot their norms by layer and global
                # def model_weights_norm(model, total_name):
                #     total_GW_norm = 0
                #     for k, v in model.named_parameters():
                #         if 'weight' in k:
                #             W = v.data
                #             W_norm = torch.norm(W)
                #             self.writer.add_scalar('{}_Wnorm'.format(k), W_norm, iteration)
                #             total_GW_norm += W_norm
                #     self.writer.add_scalar('{}_Wnorm'.format(total_name), total_GW_norm, iteration)

                # model_weights_norm(self.G, 'Gtotal')
                # model_weights_norm(self.D, 'Dtotal')
                if not opts.no_train_gen:
                    # canvas_w = self.G(noisy_samples, z=z_sample)
                    self.gen_train_samples(clean_samples, noisy_samples, z_sample, iteration=iteration)
                if va_dloader is not None:
                    if len(noisy_evals) == 0:
                        sd, nsd = self.evaluate(opts, va_dloader, log_freq,
                                                do_noisy=True)  # error, None Type is not iteretable
                        self.writer.add_scalar('noisy_SD', nsd, iteration)
                    else:
                        sd = self.evaluate(opts, va_dloader, log_freq, do_noisy=False)
                    self.writer.add_scalar('Genh_SD', sd, iteration)
                    print('Eval SD: {:.3f} dB, NSD: {:.3f} dB'.format(sd, nsd))
                    if sd < best_val_obj:
                        self.G.save(self.save_path, iteration, True)
                        best_val_obj = sd
            if iteration % len(dloader) == 0:
                # save models in end of epoch with EOE savers
                self.G.save(self.save_path, iteration, saver=eoe_g_saver)

    def save(self, save_path, step, best_val=False, saver=None):
        super(AEWSEGAN, self).save(save_path, step, best_val=False, saver=None)
