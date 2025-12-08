import torch
import logging
try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
except:
    deepspeed = None
    DeepSpeedEngine = None
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.llm.model.adapter_builder import AdapterModel

logger = logging.getLogger(__name__)


class LLMTrainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model=model,
                         data=data,
                         device=device,
                         config=config,
                         only_for_eval=only_for_eval,
                         monitor=monitor)
        self._unlearn_bases = {}
        self._warned_deepspeed_unlearn = False
        self._unlearn_kind = 'union'
        self._unlearn_round = 0
        self._proj_strength = 1.0
        self._unlearn_proj_dtype = 'float32'

    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            if not ctx.cfg.llm.deepspeed.use:
                ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        if ctx.cfg.llm.deepspeed.use:
            # Enable deepspeed
            # TODO: save ctx.optimizer and ctx.scheduler
            # TODO: should clients share the same `ctx.model_engine`?
            assert deepspeed is not None, "Please install deepspeed."
            if not hasattr(ctx, 'model_engine'):
                ctx.model_engine, ctx.optimizer, _, ctx.scheduler = \
                    deepspeed.initialize(
                        config=ctx.cfg.llm.deepspeed.ds_config,
                        model=ctx.model,
                        model_parameters=filter(lambda p: p.requires_grad,
                                                ctx.model.parameters()),
                    )
            # Enable all cards from 0
            ctx.device = ctx.model_engine.local_rank
            if ctx.cfg.train.is_enable_half:
                ctx.fp16 = ctx.model_engine.fp16_enabled()
        else:
            # prepare model and optimizer
            ctx.model.to(ctx.device)
            if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
                # Initialize optimizer here to avoid the reuse of optimizers
                # across different routines
                ctx.optimizer = get_optimizer(
                    ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
                ctx.scheduler = get_scheduler(
                    ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        # prepare statistics
        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        if ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        logits = outputs.logits
        loss = outputs.loss

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        if ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
        else:
            ctx.optimizer.zero_grad()
            ctx.loss_task.backward()
            self._project_gradients(ctx)

            if ctx.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                               ctx.grad_clip)

            ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                # Retry with new data in train and finetune
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
            f'{ctx.cur_split}_loss': ctx.loss_batch_total,
            f'{ctx.cur_split}_total': ctx.num_samples,
            f'{ctx.cur_split}_avg_loss': avg_loss,
        }
        setattr(ctx, 'eval_metrics', eval_results)

        # TODO: make this as a hook function
        # Move trainable part to `cpu`, which can save memory but cost time
        if ctx.cfg.llm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')

    def _hook_on_batch_forward_flop_count(self, ctx):
        """
        The monitoring hook to calculate the flops during the fl course

        Note:
          For customized cases that the forward process is not only \
          based on ctx.model, please override this function (inheritance \
          case) or replace this hook (plug-in case)

          The modified attributes and according operations are shown below:
            ==================================  ===========================
            Attribute                           Operation
            ==================================  ===========================
            ``ctx.monitor``                     Track average flops
            ==================================  ===========================
        """

        # The process may occupy a large amount of video memory
        # if the garbage collection is not triggered in time
        # when there is plenty of video memory left. Set
        # `eval.count_flops = False` to avoid this.
        if not isinstance(ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Please check whether this is you want.")
            return

        if self.cfg.eval.count_flops and ctx.monitor.flops_per_sample == 0:
            # calculate the flops_per_sample
            try:
                input_ids = ctx.data_batch['input_ids'].to(ctx.device)
                labels = ctx.data_batch['labels'].to(ctx.device)
                attention_mask = ctx.data_batch['attention_mask'].to(
                    ctx.device)
                from fvcore.nn import FlopCountAnalysis
                if isinstance(ctx.model, AdapterModel):
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model.model,
                        inputs=(input_ids, attention_mask)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, inputs=(input_ids, attention_mask)).total()
                ctx.monitor.track_avg_flops(flops_one_batch, ctx.batch_size)
            except Exception as e:
                logger.warning("When using count flops functions, torch's "
                               "garbage collection mechanism may not be "
                               "timely resulting in OOM, please set "
                               "`cfg.eval.count_flops` to `False` "
                               "to avoid error or warning like this.")
                logger.error(e)
                # Raise warning at the first failure
                logger.warning(
                    "current flop count implementation is for general LLM "
                    "trainer case: "
                    "1) ctx.data_batch contains [input_ids, labels, "
                    "attn_mask]; and 2) the ctx.model takes first two "
                    "arguments should be and attention_mask. "
                    "If ctx.model is an adapter model, the model in 2) has "
                    "been replaced by ctx.model.model. "
                    "Please check the forward format or implement your own "
                    "flop_count function")
                ctx.monitor.flops_per_sample = -1

        # by default, we assume the data has the same input shape,
        # thus simply multiply the flops to avoid redundant forward
        ctx.monitor.total_flops += ctx.monitor.flops_per_sample * \
            ctx.batch_size

    def set_unlearn_bases(self, bases):
        if not bases:
            self._unlearn_bases = {}
            self._unlearn_kind = 'union'
            self._proj_strength = 0.0
            return

        q_dict, meta = self._parse_unlearn_payload(bases)
        if not q_dict:
            self._unlearn_bases = {}
            self._unlearn_kind = 'union'
            self._proj_strength = 0.0
            return

        processed = {}
        for name, tensor in q_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.ndim != 2:
                continue
            processed[name] = tensor.detach().to(dtype=torch.float32).clone()
        self._unlearn_bases = processed
        self._unlearn_kind = meta.get('kind', meta.get('broadcast', 'union'))
        self._unlearn_proj_dtype = meta.get('proj_dtype',
                                            self._unlearn_proj_dtype)
        round_idx = meta.get('round')
        if round_idx is not None:
            try:
                self._unlearn_round = int(round_idx)
            except (TypeError, ValueError):
                self._unlearn_round = 0
        self._proj_strength = self._compute_proj_strength(self._unlearn_round)

    def _project_gradients(self, ctx):
        if not self.cfg.train.unlearn.project_grads:
            return
        if not self._unlearn_bases:
            return
        strength = getattr(self, '_proj_strength', 1.0)
        rho = getattr(self.cfg.train.unlearn, 'proj_rho', 1.0)
        try:
            rho_val = float(rho)
        except (TypeError, ValueError):
            rho_val = 1.0
        strength = strength * rho_val
        if strength <= 0:
            return
        if ctx.cfg.llm.deepspeed.use:
            if not self._warned_deepspeed_unlearn:
                logger.warning('UNLEARN gradient projection is disabled '
                               'when DeepSpeed is enabled.')
                self._warned_deepspeed_unlearn = True
            return

        device = ctx.device
        proj_dtype = self._dtype_from_name(self._unlearn_proj_dtype)
        with torch.no_grad():
            for name, param in ctx.model.named_parameters():
                if param.grad is None:
                    continue
                basis = self._unlearn_bases.get(name)
                if basis is None or param.grad.ndim != 2:
                    continue
                Q = basis.to(device=device, dtype=proj_dtype)
                grad_fp = param.grad.data.to(dtype=proj_dtype)
                projection = (grad_fp @ Q) @ Q.transpose(0, 1)
                adjusted = grad_fp - strength * projection
                param.grad.data = adjusted.to(dtype=param.grad.dtype)

    def _parse_unlearn_payload(self, payload):
        candidate = payload
        if isinstance(candidate, list):
            candidate = next(
                (item for item in candidate if isinstance(item, dict)),
                {})
        if not isinstance(candidate, dict):
            return {}, {}
        if 'Q' in candidate and isinstance(candidate['Q'], dict):
            meta = {k: v for k, v in candidate.items() if k != 'Q'}
            q_dict = candidate['Q']
        else:
            meta = {}
            q_dict = candidate
        return q_dict, meta

    def _compute_proj_strength(self, round_idx: int) -> float:
        schedule = getattr(self.cfg.train.unlearn, 'proj_schedule', None)
        if schedule is None:
            return 1.0
        mode = getattr(schedule, 'type', 'none')
        if isinstance(mode, str) and mode.lower() == 'linear':
            total_rounds = int(getattr(schedule, 'rounds', 0))
            if total_rounds <= 0:
                return 1.0
            idx = max(0, int(round_idx))
            return min(1.0, idx / float(total_rounds))
        return 1.0

    @staticmethod
    def _dtype_from_name(name: str) -> torch.dtype:
        mapping = {
            'float32': torch.float32,
            'fp32': torch.float32,
            'float16': torch.float16,
            'fp16': torch.float16,
            'half': torch.float16,
            'float64': torch.float64,
            'fp64': torch.float64,
            'double': torch.float64,
            'bfloat16': torch.bfloat16,
        }
        return mapping.get(name.lower(), torch.float32)


def call_llm_trainer(trainer_type):
    if trainer_type == 'llmtrainer':
        trainer_builder = LLMTrainer
        return trainer_builder


register_trainer('llmtrainer', call_llm_trainer)
