import torch
import typing, warnings
from torch_pruning.pruner import MetaPruner


class TaylorStepPruner(MetaPruner):
    def set_storage(self):
        self.groups_imps = []

    def store_importance(self, is_first=True, add=True):

        for i, group in enumerate(self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types)):
            if self._check_sparsity(group): # check pruning ratio
                group = self._downstream_node_as_root_if_unbind(group)
                ch_groups = self._get_channel_groups(group)

                # print(group)

                imp = self.estimate_importance(group, ch_groups=ch_groups)
                # print(imp)
                if is_first:
                    self.groups_imps.append(imp)
                else:
                    self.groups_imps[i] += imp

        # print(self.groups_imps[0].sum())

    def prune_local(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            warnings.warn("Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return
        for i, group in enumerate(self.DG.get_all_groups(ignored_layers=self.ignored_layers,
                                                         root_module_types=self.root_module_types)):
            if self._check_sparsity(group):  # check pruning ratio

                group = self._downstream_node_as_root_if_unbind(group)

                module = group[0][0].target.module
                pruning_fn = group[0][0].handler
                ch_groups = self._get_channel_groups(group)

                # imp = self.estimate_importance(group, ch_groups=ch_groups)
                imp = self.groups_imps[i]
                # print(imp.sum())
                if imp is None: continue

                if self.DG.is_out_channel_pruning_fn(pruning_fn):
                    current_channels = self.DG.get_out_channels(module)
                    target_sparsity = self.get_target_sparsity(module)
                    n_pruned = current_channels - int(
                        self.layer_init_out_ch[module] *
                        (1 - target_sparsity)
                    )
                else:
                    current_channels = self.DG.get_in_channels(module)
                    target_sparsity = self.get_target_sparsity(module)
                    n_pruned = current_channels - int(
                        self.layer_init_in_ch[module] *
                        (1 - target_sparsity)
                    )

                # round to the nearest multiple of round_to
                if self.round_to:
                    n_pruned = self._round_to(n_pruned, current_channels, self.round_to)

                if n_pruned <= 0:
                    continue

                if ch_groups > 1:  # independent pruning for each group
                    group_size = current_channels // ch_groups
                    pruning_idxs = []
                    n_pruned_per_group = n_pruned // ch_groups  # max(1, n_pruned // ch_groups)
                    if n_pruned_per_group == 0: continue  # skip
                    for chg in range(ch_groups):
                        sub_group_imp = imp[chg * group_size: (chg + 1) * group_size]
                        sub_imp_argsort = torch.argsort(sub_group_imp)
                        sub_pruning_idxs = sub_imp_argsort[:n_pruned_per_group] + chg * group_size  # offset
                        pruning_idxs.append(sub_pruning_idxs)
                    pruning_idxs = torch.cat(pruning_idxs, 0)
                else:  # no channel grouping
                    imp_argsort = torch.argsort(imp)
                    pruning_idxs = imp_argsort[:n_pruned]

                group = self.DG.get_pruning_group(
                    module, pruning_fn, pruning_idxs.tolist())

                if self.DG.check_pruning_group(group):
                    yield group



