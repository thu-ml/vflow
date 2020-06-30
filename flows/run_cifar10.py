import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
from tensorflow.distributions import Normal
import time
from utils import save_tiled_imgs, print_params

from flow_training import train, evaluate, load_data, iterbatches
from flows import (
    Flow, Compose, Inverse, ImgProc, Sigmoid, AffineCoupling, MixLogisticAttnCoupling,
    TupleFlip, CheckerboardSplit, ChannelSplit, SpaceToDepth, Norm, Pointwise, ElemwiseAffine,
    conv2d, gated_conv, nin, gate, layernorm, VarConfig, get_var, gaussian_sample_logp, sumflat
)


def construct(*, filters, dequant_filters, posterior_filters, blocks, posterior_blocks, components, extra_dims=0, x_dims=3, x_max_val=256):
    coupling_kwargs = dict(filters=filters, blocks=blocks, components=components)
    posterior_coupling_kwargs = dict(filters=posterior_filters, blocks=posterior_blocks, components=components)
    dequant_coupling_kwargs = dict(filters=filters, blocks=2, components=components)

    class PosteriorLatent(Flow):
        def __init__(self):
            class NormalPosterior:
                def forward(self, x, *, vcfg, dropout_p=0., verbose=True, context=None):
                    z, logq_z = gaussian_sample_logp(x.shape.as_list()[:-1] + [extra_dims])
                    return z, logq_z

            class FlowPosterior:
                def __init__(self):
                    self.inited = False
                    def shallow_processor(x, *, dropout_p, vcfg):
                        context = {}

                        (this, that), _ = CheckerboardSplit().forward(x)
                        x = conv2d(tf.concat([this, that], 3), name='proj_q', num_units=32, vcfg=vcfg)
                        for i in range(3):
                            x = gated_conv(x, name=f'q_c{i}', vcfg=vcfg, dropout_p=dropout_p, use_nin=False, a=None)
                        context[f'{x.shape.as_list()[1]}_{x.shape.as_list()[2]}'] = x

                        return context

                    self.context_proc = tf.make_template("context_proc", shallow_processor)

                    self.q_flow = Compose([
                        CheckerboardSplit(),
                        Norm(), Pointwise(), MixLogisticAttnCoupling(**posterior_coupling_kwargs), TupleFlip(),
                        Norm(), Pointwise(), MixLogisticAttnCoupling(**posterior_coupling_kwargs), TupleFlip(),
                        Norm(), Pointwise(), MixLogisticAttnCoupling(**posterior_coupling_kwargs), TupleFlip(),
                        Norm(), Pointwise(), MixLogisticAttnCoupling(**posterior_coupling_kwargs), TupleFlip(),
                        Inverse(CheckerboardSplit()),
                        Sigmoid(),
                    ])

                def forward(self, x, *, vcfg, dropout_p=0., verbose=True, context=None):
                    '''
                    z = f(u)
                    u = f^{-1}(z)
                    p(u) = p(z) * |det(J_f^{-1}^{-1}(u))| = p(z) * |det(J_f(u))|
                    '''
                    assert context is None
                    eps, eps_logp = gaussian_sample_logp(x.shape.as_list()[:-1] + [extra_dims])
                    xd, logd = self.q_flow.forward(
                        eps,
                        context=self.context_proc(x, dropout_p=dropout_p, vcfg=vcfg),
                        dropout_p=dropout_p, verbose=verbose, vcfg=vcfg
                    )
                    return xd, eps_logp - logd
                
            # self.posterior_flow = NormalPosterior()
            self.posterior_flow = FlowPosterior()

        def forward(self, x, **flow_kwargs):
            """
            return: z, shape (1, batch_size, h, w, extra_dims)
                    logq_z, shape (batch_size)
            """
            z, logq_z = self.posterior_flow.forward(x, **flow_kwargs)
            return tf.reshape(z, [1,] + x.shape.as_list()[:-1] + [extra_dims]), tf.reshape(logq_z, [-1])


    class UnifDequant(Flow):
        def forward(self, x, **kwargs):
            x = x + tf.random_uniform(x.shape.as_list())
            x, logd = ImgProc(max_val=x_max_val).forward(x)
            return x, logd


    class Dequant(Flow):
        def __init__(self):
            def shallow_processor(x, *, dropout_p, vcfg):
                x = x / (x_max_val * 1.0) - 0.5
                context = {}

                (this, that), _ = CheckerboardSplit().forward(x)
                x = conv2d(tf.concat([this, that], 3), name='proj', num_units=32, vcfg=vcfg)
                for i in range(3):
                    x = gated_conv(x, name=f'c{i}', vcfg=vcfg, dropout_p=dropout_p, use_nin=False, a=None)
                context[f'{x.shape.as_list()[1]}_{x.shape.as_list()[2]}'] = x

                return context

            self.context_proc = tf.make_template("context_proc", shallow_processor)

            self.dequant_flow = Compose([
                CheckerboardSplit(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**dequant_coupling_kwargs), TupleFlip(),
                Inverse(CheckerboardSplit()),
                Sigmoid(),
            ])

        def forward(self, x, *, vcfg, dropout_p=0., verbose=True, context=None):
            assert context is None
            eps, eps_logp = gaussian_sample_logp(x.shape.as_list())
            xd, logd = self.dequant_flow.forward(
                eps,
                context=self.context_proc(x, dropout_p=dropout_p, vcfg=vcfg),
                dropout_p=dropout_p, verbose=verbose, vcfg=vcfg
            )
            assert eps.shape == x.shape and logd.shape == eps_logp.shape == [x.shape[0]]
            x = x + xd
            dequant_logd = logd - eps_logp
            x, logd = ImgProc(max_val=x_max_val).forward(x)
            return x, dequant_logd + logd


    class SpecialAffine(Flow):
        def __init__(self, init_scale=1., blocks=3):
            self.inited = False
            def f(x, *, vcfg:VarConfig, dropout_p=0., verbose=True, context=None):
                assert context is None
                vcfg = VarConfig(init=(vcfg.init and not self.inited), ema=vcfg.ema, dtype=vcfg.dtype)
                if vcfg.init and verbose:
                    self.inited = True
                    # debug stuff
                    xmean, xvar = tf.nn.moments(x, axes=list(range(len(x.shape))))
                    x = tf.Print(
                        x, [tf.shape(x), xmean, tf.sqrt(xvar), tf.reduce_min(x), tf.reduce_max(x)],
                        message='{} (shape/mean/std/min/max) '.format(self.generative_net.variable_scope.name), summarize=10
                    )
                B, H, W, C = x.shape.as_list()
                x = conv2d(x, name='p_proj_in', num_units=filters, vcfg=vcfg)
                for i in range(blocks):
                    with tf.variable_scope(f'p_block{i}'):
                        x = gated_conv(x, name='p_conv', a=context, use_nin=True, dropout_p=dropout_p, vcfg=vcfg)
                        x = layernorm(x, name='p_ln', vcfg=vcfg)
                x = conv2d(x, name='p_proj_out', num_units=x_dims*2, init_scale=init_scale, vcfg=vcfg)
                x = tf.reshape(x, [B, H, W, x_dims*2])
                return x

            self.generative_net = tf.make_template(self.__class__.__name__, f)

        def decode(self, z, flow_kwargs):
            mean, logstd = tf.split(self.generative_net(z, **flow_kwargs), num_or_size_splits=2, axis=3)
            std = tf.exp(logstd) + 1e-6
            return mean, logstd, std

        def forward(self, xz, flow_kwargs):
            '''
            (x, z) -> (epsilon_x, epsilon_z)
            '''
            x = xz[:,:,:,:x_dims]
            z = xz[:,:,:,x_dims:]
            mean, logstd, std = self.decode(z, flow_kwargs)
            epsilon_x = (x - mean) / std
            epsilon_z = z
            logd = -sumflat(logstd)
            return tf.concat([epsilon_x, epsilon_z], axis=-1), logd

        def inverse(self, xz, flow_kwargs):
            epsilon_x = xz[:,:,:,:x_dims]
            z = xz[:,:,:,x_dims:]
            mean, logstd, std = self.decode(z, flow_kwargs)
            x = epsilon_x * std + mean
            logd = sumflat(logstd)
            return tf.concat([x, z], axis=-1), logd


    class SpecialAffineNULL(Flow):
        def forward(self, xz, flow_kwargs):
            return xz, tf.constant(0.)

        def inverse(self, xz, flow_kwargs):
            return xz, tf.constant(0.)


    class Generative(Flow):
        def __init__(self):
            self.vaeflow = SpecialAffineNULL()
            self.flow = Compose([
                # 32x32x6 (3+3)
                CheckerboardSplit(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                Inverse(CheckerboardSplit()),

                ChannelSplit(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                Inverse(ChannelSplit()),

                # 16x16x24
                SpaceToDepth(),

                CheckerboardSplit(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                Inverse(CheckerboardSplit()),

                ChannelSplit(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                Norm(), Pointwise(), MixLogisticAttnCoupling(**coupling_kwargs), TupleFlip(),
                Inverse(ChannelSplit()),
            ])

        def forward(self, x, z, flow_kwargs):
            '''
            y = f(xz)
            xz = f^{-1}(y)
            p(xz) = p(y) * |det(J_f^{-1}^{-1}(xz))| = p(y) * |det(J_f(xz))|
            '''
            H, W, Cx = x.shape.as_list()[-3:]
            x = tf.reshape(x, [-1, H, W, Cx])
            z = tf.reshape(z, [-1, H, W, extra_dims])

            xz = tf.concat([x, z], axis=-1)
            y, logd_flow = self.vaeflow.forward(xz, flow_kwargs)
            y, ld = self.flow.forward(y, **flow_kwargs)
            logd_flow += ld
            logpy = sumflat(Normal(0., 1.).log_prob(y))
            return y, logpy + logd_flow

        def inverse(self, y, flow_kwargs):
            '''
            g = f^{-1}
            y = g^{-1}(xz)
            p(y) = p(xz) * |det(J_g^{-1}^{-1}(y))| = p(xz) * |det(J_g(y))|
            '''
            y = tf.reshape(y, [-1] + y.shape.as_list()[-3:])
            logpy = sumflat(Normal(0., 1.).log_prob(y))

            xz = y
            xz = tf.reshape(xz, [-1, 16, 16, 4*(x_dims+extra_dims)])
            xz, ld = self.flow.inverse(xz, **flow_kwargs)
            xz, logd_flow = self.vaeflow.inverse(xz, flow_kwargs)
            logd_flow += ld
            return xz, logpy + logd_flow

        def sample(self, num_samples, flow_kwargs):
            y = tf.random_normal([num_samples, 32, 32, x_dims + extra_dims])
            xz, logp_xz = self.inverse(y, flow_kwargs)
            x = xz[:,:,:,:x_dims]
            x, logd = ImgProc(max_val=x_max_val).inverse(x)
            return x, logp_xz + logd


    dequant_flow = Dequant()
    # dequant_flow = UnifDequant()
    posterior_flow = PosteriorLatent()

    flow = Generative()

    return dequant_flow, flow, posterior_flow


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_checkpoint', type=str, default=None)
    args = parser.parse_args()

    batch_size = 64
    max_lr = 0.0012
    min_lr = 3e-4
    warmup_steps = 2000
    lr_decay = 0.99999

    def lr_schedule(step):
        global curr_lr
        if step < warmup_steps:
            return max_lr * step / warmup_steps
        elif step >= warmup_steps and step <= (25 * warmup_steps):
            curr_lr =  max_lr
            return max_lr
        elif step > (25 * warmup_steps) and curr_lr > min_lr:
            curr_lr *= lr_decay
            return curr_lr
        return curr_lr

    dropout_p = 0.2
    blocks = 10
    posterior_blocks = 3
    filters = dequant_filters = 96
    posterior_filters = 96
    components = 32  # logistic mixture components
    ema_decay = 0.999

    extra_dims = 3

    def flow_constructor():
        return construct(filters=filters, dequant_filters=dequant_filters, posterior_filters=posterior_filters, blocks=blocks, posterior_blocks=posterior_blocks, components=components, extra_dims=extra_dims)

    if args.eval_checkpoint:
        evaluate(flow_constructor=flow_constructor, seed=0, restore_checkpoint=args.eval_checkpoint, total_bs=1024)
        return

    train(
        flow_constructor=flow_constructor,
        logdir=f'~/logs/vflow_fbdq{dequant_filters}_blocks{blocks}_f{filters}_lr{max_lr}_drop{dropout_p}_extra_dims{extra_dims}_posterior_blocks{posterior_blocks}_batchsize{batch_size}',
        lr_schedule=lr_schedule,
        dropout_p=dropout_p,
        seed=0,
        init_bs=batch_size,
        total_bs=batch_size,
        val_total_bs=batch_size,
        ema_decay=ema_decay,
        steps_per_log=100,
        epochs_per_val=1,
        max_grad_norm=1.,
    )
    return

if __name__ == '__main__':
    main()
