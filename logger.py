from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import os
import shutil
import torch
import torchvision
import numpy as np
from termcolor import colored

# NOTE:
# Logger writes:
# - train.log / eval.log : JSONL
# - console line for train/eval using FORMAT_CONFIG
# - TensorBoard scalars under tb/
#
# IMPORTANT:
# MetersGroup will remove prefix train/ or eval/ and then replace "/" with "_".
# So key "train/headway/p_in_1p5_2p5" becomes "headway_p_in_1p5_2p5" in JSON & console mapping.

FORMAT_CONFIG = {
    'rl': {
        'train': [
            ('episode', 'E', 'int'),
            ('step', 'S', 'int'),
            ('FiT', 'FiT', 'int'),

            # Safety TTC
            ('safety_min_TTC', 'minTTC', 'float3'),
            ('safety_TTC5', 'TTC5', 'float3'),
            ('safety_p_ttc_danger', 'pTTC<=2', 'float3'),

            # Validity
            ('cf_valid_cf_ratio', 'vCF', 'float4'),

            # Headway goal (core)
            ('headway_THW_mean', 'THWμ', 'float3'),
            ('headway_THW_p50_headway_THW_p95', 'THW50/95', 'pair_float3'),
            ('headway_p_in_1p5_2p5', 'pTHW[1.5,2.5]', 'float3'),
            ('headway_p_safe_in_1p5_2p5', 'pSafeTHW', 'float3'),
            ('headway_p_lt_1p0', 'pTHW<1.0', 'float3'),

            # Stability
            ('stability_MAE_dv', 'MAE|dv|', 'float3'),

            # Robustness
            ('robust_same_lane_ratio', 'sameRoute', 'float4'),
            ('robust_junction_uncertain_ratio', 'junc?', 'float4'),
            ('robust_offlane_streak_max', 'offStr', 'int'),
            ('robust_leader_missing_streak_max', 'missStr', 'int'),
            ('leader_leader_kick_count', 'leadKick', 'int'),

            # Speed
            ('speed_mean_kmh', 'v̄kmh', 'float2'),
            ('speed_max_kmh', 'vmax', 'float2'),

            # RL main
            ('ep_reward', 'ER', 'float'),
            ('batch_reward', 'BR', 'float'),
            ('actor_loss', 'A_LOSS', 'float'),
            ('critic_loss', 'CR_LOSS', 'float'),
            ('curl_loss', 'CU_LOSS', 'float'),
            ('cpc_loss', 'CPC_LOSS', 'float'),
        ],
        'eval': [
            ('step', 'S', 'int'),
            ('mean_ep_reward', 'MER', 'float'),
            ('max_ep_reward', 'BER', 'float'),
            ('min_ep_reward', 'mER', 'float'),
            ('std_ep_reward', 'sdER', 'float'),
        ]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0.0
        self._count = 0

    def update(self, value, n=1):
        try:
            v = float(value)
        except Exception:
            v = float('nan')
        self._sum += v
        self._count += int(n)

    def value(self):
        if self._count <= 0:
            return float('nan')
        return self._sum / float(self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = {}
        for key, meter in self._meters.items():
            if key.startswith('train/'):
                key2 = key[len('train/'):]
            elif key.startswith('eval/'):
                key2 = key[len('eval/'):]
            else:
                key2 = key
            key2 = key2.replace('/', '_')
            data[key2] = meter.value()
        return data

    def _apply_compat_mapping(self, data: dict):
        # ✅ loss keys compatibility (CURL-SAC often logs train_actor_loss/train_critic_loss)
        if ('actor_loss' not in data) and ('train_actor_loss' in data):
            data['actor_loss'] = data.get('train_actor_loss')

        if ('critic_loss' not in data) and ('train_critic_loss' in data):
            data['critic_loss'] = data.get('train_critic_loss')

        if ('curl_loss' not in data) and ('train_curl_loss' in data):
            data['curl_loss'] = data.get('train_curl_loss')

        if ('cpc_loss' not in data) and ('train_cpc_loss' in data):
            data['cpc_loss'] = data.get('train_cpc_loss')

        # ✅ older keys compatibility (if someone still logs train/min_TTC, map it)
        if ('safety_min_TTC' not in data) and ('min_TTC' in data):
            data['safety_min_TTC'] = data.get('min_TTC')
        if ('safety_TTC5' not in data) and ('TTC5' in data):
            data['safety_TTC5'] = data.get('TTC5')

        if ('cf_valid_cf_ratio' not in data) and ('valid_cf_ratio' in data):
            data['cf_valid_cf_ratio'] = data.get('valid_cf_ratio')

        if ('stability_MAE_dv' not in data) and ('MAE_dv' in data):
            data['stability_MAE_dv'] = data.get('MAE_dv')

        if ('headway_THW_mean' not in data) and ('THW_mean' in data):
            data['headway_THW_mean'] = data.get('THW_mean')
        if ('headway_THW_p50' not in data) and ('THW_p50' in data):
            data['headway_THW_p50'] = data.get('THW_p50')
        if ('headway_THW_p95' not in data) and ('THW_p95' in data):
            data['headway_THW_p95'] = data.get('THW_p95')

        if ('speed_mean_kmh' not in data) and ('mean_kmh' in data):
            data['speed_mean_kmh'] = data.get('mean_kmh')
        if ('speed_max_kmh' not in data) and ('max_kmh' in data):
            data['speed_max_kmh'] = data.get('max_kmh')

        if ('robust_same_lane_ratio' not in data) and ('same_lane_ratio' in data):
            data['robust_same_lane_ratio'] = data.get('same_lane_ratio')
        if ('robust_junction_uncertain_ratio' not in data) and ('junction_uncertain_ratio' in data):
            data['robust_junction_uncertain_ratio'] = data.get('junction_uncertain_ratio')
        if ('robust_offlane_streak_max' not in data) and ('offlane_streak_max' in data):
            data['robust_offlane_streak_max'] = data.get('offlane_streak_max')
        if ('robust_leader_missing_streak_max' not in data) and ('leader_missing_streak_max' in data):
            data['robust_leader_missing_streak_max'] = data.get('leader_missing_streak_max')
        if ('leader_leader_kick_count' not in data) and ('leader_kick_count' in data):
            data['leader_leader_kick_count'] = data.get('leader_kick_count')

        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def _to_int(self, x, default=0):
        try:
            if x is None:
                return default
            xf = float(x)
            if np.isnan(xf):
                return default
            return int(xf)
        except Exception:
            return default

    def _to_float(self, x, default=float('nan')):
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _fmt_floatN(self, x, ndigits=3):
        try:
            if x is None:
                return "nan"
            xf = float(x)
            if np.isnan(xf):
                return "nan"
            fmt = "{:." + str(int(ndigits)) + "f}"
            return fmt.format(xf)
        except Exception:
            return "nan"

    def _format(self, key, value, ty):
        template = '%s: '

        if ty == 'int':
            template += '%d'
            return template % (key, self._to_int(value, default=0))

        elif ty == 'float':
            template += '%.04f'
            v = self._to_float(value, default=float('nan'))
            return template % (key, float(v))

        elif ty == 'float2':
            template += '%.02f'
            v = self._to_float(value, default=float('nan'))
            return template % (key, float(v))

        elif ty == 'float3':
            template += '%.03f'
            v = self._to_float(value, default=float('nan'))
            return template % (key, float(v))

        elif ty == 'float4':
            template += '%.04f'
            v = self._to_float(value, default=float('nan'))
            return template % (key, float(v))

        elif ty == 'pair_float3':
            if isinstance(value, (tuple, list)) and len(value) == 2:
                a, b = value
            else:
                a, b = (float('nan'), float('nan'))
            return f"{key}: {self._fmt_floatN(a, 3)}/{self._fmt_floatN(b, 3)}"

        else:
            raise Exception('invalid format type: %s' % ty)

    def _dump_to_console(self, data, prefix):
        prefix_col = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix_col)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, float('nan'))
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return

        data = self._prime_meters()
        data['step'] = int(step)

        data = self._apply_compat_mapping(data)

        if prefix == 'train':
            for k in ('actor_loss', 'critic_loss', 'curl_loss', 'cpc_loss'):
                if k not in data:
                    data[k] = float('nan')
            p50 = data.get('headway_THW_p50', float('nan'))
            p95 = data.get('headway_THW_p95', float('nan'))
            data['headway_THW_p50_headway_THW_p95'] = (p50, p95)

        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb=True, config='rl'):
        self._log_dir = log_dir

        if use_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None

        self._train_mg = MetersGroup(
            os.path.join(log_dir, 'train.log'),
            formating=FORMAT_CONFIG[config]['train']
        )
        self._eval_mg = MetersGroup(
            os.path.join(log_dir, 'eval.log'),
            formating=FORMAT_CONFIG[config]['eval']
        )

        # FiT legend
        try:
            with open(os.path.join(log_dir, 'train.log'), 'a', encoding='utf-8') as f:
                f.write('# FiT legend: 0=running/none, 1=collision, 2=time_up, 3=stall, 4=too_close\n')
        except Exception:
            pass

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_image(self, key, image, step):
        if self._sw is not None:
            assert image.dim() == 3
            grid = torchvision.utils.make_grid(image.unsqueeze(1))
            self._sw.add_image(key, grid, step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def log(self, key, value, step, n=1):
        assert key.startswith('train') or key.startswith('eval')
        if isinstance(value, torch.Tensor):
            value = value.item()

        try:
            v = float(value) / float(n)
        except Exception:
            v = float('nan')

        # TensorBoard scalar
        self._try_sw_log(key, v, step)

        # JSONL / console meters
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step):
        self.log_histogram(key + '_w', param.weight.data, step)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step)
        if hasattr(param, 'bias'):
            self.log_histogram(key + '_b', param.bias.data, step)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step)

    def log_image(self, key, image, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_image(key, image, step)

    def log_video(self, key, frames, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step):
        assert key.startswith('train') or key.startswith('eval')
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step):
        self._train_mg.dump(step, 'train')
        self._eval_mg.dump(step, 'eval')
        try:
            if self._sw is not None:
                self._sw.flush()
        except Exception:
            pass
