"""Bounded diagnostic observations copied by retained prefill and ordinary B1 decode graphs."""

from dataclasses import dataclass

import torch

# Device-side gating keeps ordinary diagnostic requests off the reduction path.
# triton_op remains visible to torch.compile and is warmed with graph capture.
try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only mathematical qualification remains available.
    triton = None

if triton is not None:
    @triton.jit
    def _search_neurons(X, Positions, Enabled, SelectedPosition, Peak, PeakPosition,
                        SumAbs, Selected, WIDTH: tl.constexpr, TOKENS: tl.constexpr,
                        STRIDE: tl.constexpr, COLUMN_STRIDE: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
        if tl.load(Enabled):
            row = tl.arange(0, ROWS)
            col = tl.program_id(0) * COLS + tl.arange(0, COLS)
            positions = tl.load(Positions + row, row < TOKENS, -1)
            valid = (row < TOKENS) & (positions >= 0)
            value = tl.load(X + row[:, None] * STRIDE + col[None, :] * COLUMN_STRIDE, valid[:, None] & (col[None, :] < WIDTH), 0.).to(tl.float32)
            absolute = tl.abs(value)
            # Propagate any nonfinite input to host validation via SumAbs.
            maximum = tl.max(absolute, axis=0)
            first = tl.min(tl.where((absolute == maximum[None, :]) & valid[:, None], row[:, None], 2147483647), axis=0)
            signed = tl.load(X + first * STRIDE + col * COLUMN_STRIDE, (first < TOKENS) & (col < WIDTH), 0.)
            position = tl.load(Positions + first, first < TOKENS, -1)
            selected_row = tl.min(tl.where(valid & (positions == tl.load(SelectedPosition)), row, 2147483647), axis=0)
            selected = tl.load(X + selected_row * STRIDE + col * COLUMN_STRIDE,
                               (selected_row < TOKENS) & (col < WIDTH), 0.).to(tl.float32)
            tl.store(Peak + col, signed, col < WIDTH)
            tl.store(PeakPosition + col, position, col < WIDTH)
            tl.store(SumAbs + col, tl.sum(absolute, axis=0), col < WIDTH)
            tl.store(Selected + col, selected, col < WIDTH)

    @triton.jit
    def _search_tokens(X, Positions, Enabled, Peak, Neuron, WIDTH: tl.constexpr,
                       STRIDE: tl.constexpr, COLUMN_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
        if tl.load(Enabled):
            row = tl.program_id(0)
            col = tl.arange(0, BLOCK)
            live = tl.load(Positions + row) >= 0
            value = tl.load(X + row * STRIDE + col * COLUMN_STRIDE, (col < WIDTH) & live, 0.).to(tl.float32)
            maximum = tl.max(tl.abs(value), axis=0)
            first = tl.min(tl.where((tl.abs(value) == maximum) & (col < WIDTH), col, 2147483647), axis=0)
            signed = tl.load(X + row * STRIDE + first * COLUMN_STRIDE, (first < WIDTH) & live, 0.)
            tl.store(Peak + row, signed)
            tl.store(Neuron + row, first)

    @torch.library.triton_op('flare::activation_search_reduce', mutates_args={
        'peak', 'peak_position', 'sum_abs', 'token_peak', 'token_neuron', 'selected'})
    def _search_reduce(value: torch.Tensor, positions: torch.Tensor, enabled: torch.Tensor,
                       selected_position: torch.Tensor, peak: torch.Tensor, peak_position: torch.Tensor,
                       sum_abs: torch.Tensor, token_peak: torch.Tensor, token_neuron: torch.Tensor,
                       selected: torch.Tensor) -> None:
        from torch.library import wrap_triton
        tokens, width = value.shape
        wrap_triton(_search_neurons)[(triton.cdiv(width, 32),)](
            value, positions, enabled, selected_position, peak, peak_position, sum_abs, selected,
            width, tokens, value.stride(0), value.stride(1), triton.next_power_of_2(tokens), 32, num_warps=4)
        wrap_triton(_search_tokens)[(tokens,)](value, positions, enabled, token_peak, token_neuron,
            width, value.stride(0), value.stride(1), triton.next_power_of_2(width), num_warps=4)


@dataclass(frozen=True)
class ActivationCapture:
    token_indices: torch.Tensor
    token_enabled: torch.Tensor
    token_values: torch.Tensor
    search_enabled: torch.Tensor
    search_position: torch.Tensor
    search_peak: torch.Tensor
    search_peak_position: torch.Tensor
    search_sum_abs: torch.Tensor
    search_token_peak: torch.Tensor
    search_token_neuron: torch.Tensor
    search_selected: torch.Tensor
    residual_last: torch.Tensor
    pre_mlp_last: torch.Tensor
    mlp_input_last: torch.Tensor
    residual_l2: torch.Tensor
    positions: torch.Tensor
    last_index: torch.Tensor
    live: torch.Tensor

    @classmethod
    def allocate(cls, layers, tokens, hidden, device, *, intermediate):
        return cls(torch.zeros((layers, 32), dtype=torch.int64, device=device),
                   torch.zeros((layers, 32), dtype=torch.bool, device=device),
                   torch.zeros((layers, tokens, 32), dtype=torch.float32, device=device),
                   torch.zeros(1, dtype=torch.bool, device=device),
                   torch.full((1,), -1, dtype=torch.int64, device=device),
                   torch.zeros((layers, intermediate), dtype=torch.float32, device=device),
                   torch.full((layers, intermediate), -1, dtype=torch.int64, device=device),
                   torch.zeros((layers, intermediate), dtype=torch.float32, device=device),
                   torch.zeros((layers, tokens), dtype=torch.float32, device=device),
                   torch.zeros((layers, tokens), dtype=torch.int64, device=device),
                   torch.zeros((layers, intermediate), dtype=torch.float32, device=device),
                   torch.zeros((layers, hidden), dtype=torch.float32, device=device),
                   torch.zeros((layers, hidden), dtype=torch.float32, device=device),
                   torch.zeros((layers, intermediate), dtype=torch.float32, device=device),
                   torch.zeros((layers, tokens), dtype=torch.float32, device=device),
                   torch.full((tokens,), -1, dtype=torch.int64, device=device),
                   torch.zeros(1, dtype=torch.int64, device=device),
                   torch.zeros(1, dtype=torch.bool, device=device))

    def rows(self, tokens):
        return type(self)(self.token_indices, self.token_enabled, self.token_values[:, :tokens],
                          self.search_enabled, self.search_position, self.search_peak, self.search_peak_position,
                          self.search_sum_abs, self.search_token_peak[:, :tokens], self.search_token_neuron[:, :tokens],
                          self.search_selected, self.residual_last, self.pre_mlp_last, self.mlp_input_last, self.residual_l2[:, :tokens],
                          self.positions[:tokens], self.last_index, self.live)

    def begin(self, metadata):
        self.positions.copy_(metadata.positions)
        self.last_index.copy_((metadata.live_counts[:1] - 1).clamp(0, self.positions.numel() - 1))
        self.live.copy_(metadata.live_counts[:1] > 0)

    def record_pre_mlp(self, value, layer):
        self.pre_mlp_last[layer].copy_(torch.where(self.live, value[self.last_index].reshape(-1), 0))

    def record_mlp_input(self, value, layer):
        if value.device.type == "cuda":
            _search_reduce(value, self.positions, self.search_enabled, self.search_position,
                           self.search_peak[layer], self.search_peak_position[layer], self.search_sum_abs[layer],
                           self.search_token_peak[layer], self.search_token_neuron[layer], self.search_selected[layer])
        else:
            # Static graph operations: reduce every live token and every neuron.
            # No [layers,tokens,neurons] capture plane is allocated. These are chunk
            # summaries; the host accumulates them after each ordinary prefill.
            valid_rows = (self.positions >= 0) & self.search_enabled
            masked = torch.where(valid_rows[:, None], value.float(), 0)
            absolute = masked.abs()
            peak_rows = absolute.argmax(dim=0)
            self.search_peak[layer].copy_(masked.gather(0, peak_rows[None, :]).squeeze(0))
            self.search_peak_position[layer].copy_(self.positions[peak_rows])
            self.search_sum_abs[layer].copy_(absolute.sum(dim=0))
            peak_neurons = absolute.argmax(dim=1)
            self.search_token_neuron[layer].copy_(peak_neurons)
            self.search_token_peak[layer].copy_(masked.gather(1, peak_neurons[:, None]).squeeze(1))
            selected_row = (self.positions == self.search_position) & valid_rows
            self.search_selected[layer].copy_(torch.where(selected_row.any(),
                masked[selected_row.to(torch.int64).argmax()], 0))
        selected = value.index_select(-1, self.token_indices[layer])
        valid = (self.positions >= 0)[:, None] & self.token_enabled[layer][None, :]
        self.token_values[layer].copy_(torch.where(valid, selected, 0))
        self.mlp_input_last[layer].copy_(torch.where(self.live, value[self.last_index].reshape(-1), 0))

    def record_residual(self, value, layer):
        self.residual_last[layer].copy_(torch.where(self.live, value[self.last_index].reshape(-1), 0))
        self.residual_l2[layer].copy_(torch.where(self.positions >= 0, value.square().sum(-1).sqrt(), 0))


@dataclass(frozen=True)
class RequestActivationCapture:
    """Ordinary-decode observations indexed by physical request slot.

    Separate from legacy B1 capture: no selected-neuron or all-token search
    reduction is pooled across requests. Each live slot must own exactly one
    packed token. Invalid mappings and duplicate owners produce explicit error
    masks; absent/ambiguous slots have zero planes and sentinel evidence.
    The consumer must reject a replay when mapping_error is true.
    """
    row_slot_ids: torch.Tensor
    row_generations: torch.Tensor
    request_generations: torch.Tensor
    physical_slots: torch.Tensor
    token_rows: torch.Tensor
    token_slots: torch.Tensor
    token_positions: torch.Tensor
    token_ids: torch.Tensor
    token_generations: torch.Tensor
    token_live: torch.Tensor
    invalid_tokens: torch.Tensor
    slot_match_counts: torch.Tensor
    slot_last_index: torch.Tensor
    slot_live: torch.Tensor
    slot_positions: torch.Tensor
    slot_token_ids: torch.Tensor
    slot_generations: torch.Tensor
    mapping_error: torch.Tensor
    residual_last: torch.Tensor
    pre_mlp_last: torch.Tensor
    mlp_input_last: torch.Tensor

    @classmethod
    def allocate(cls, layers, capacity, hidden, device, *, intermediate,
                 row_slot_ids, row_generations, request_generations):
        if (any(type(n) is not int or n < 1 for n in (layers, capacity, hidden, intermediate))
                or capacity not in (1, 2, 4, 8, 16, 32, 64)):
            raise ValueError('Capture dimensions require a supported positive request capacity')
        device = torch.device(device)
        for value in (row_slot_ids, row_generations, request_generations):
            if (value.ndim != 1 or value.numel() != capacity
                    or value.dtype not in (torch.int32, torch.int64)
                    or value.device != torch.empty(0, device=device).device):
                raise ValueError('Request capture must bind the full native row/epoch buffers')
        def zeros(shape, dtype=torch.int64):
            return torch.zeros(shape, dtype=dtype, device=device)
        def sentinel():
            return torch.full((capacity,), -1, dtype=torch.int64, device=device)
        return cls(row_slot_ids, row_generations, request_generations,
                   torch.arange(capacity, dtype=torch.int64, device=device),
                   sentinel(), sentinel(), sentinel(), sentinel(), zeros(capacity),
                   zeros(capacity, torch.bool), zeros(capacity, torch.bool),
                   zeros(capacity), zeros(capacity), zeros(capacity, torch.bool),
                   sentinel(), sentinel(), zeros(capacity), zeros(1, torch.bool),
                   zeros((layers, capacity, hidden), torch.float32),
                   zeros((layers, capacity, hidden), torch.float32),
                   zeros((layers, capacity, intermediate), torch.float32))

    def rows(self, tokens):
        if type(tokens) is not int or not 1 <= tokens <= self.physical_slots.numel():
            raise ValueError('Decode token capacity must fit physical request capacity')
        # Output planes remain physical-slot indexed even in a B1/B2 view.
        return type(self)(self.row_slot_ids, self.row_generations, self.request_generations,
            self.physical_slots, self.token_rows[:tokens], self.token_slots[:tokens],
            self.token_positions[:tokens], self.token_ids[:tokens], self.token_generations[:tokens],
            self.token_live[:tokens], self.invalid_tokens[:tokens], self.slot_match_counts,
            self.slot_last_index, self.slot_live, self.slot_positions, self.slot_token_ids,
            self.slot_generations, self.mapping_error, self.residual_last,
            self.pre_mlp_last, self.mlp_input_last)

    def begin(self, metadata):
        if (metadata.positions.shape != self.token_positions.shape
                or metadata.requests.shape != self.token_rows.shape
                or metadata.token_ids.shape != self.token_ids.shape):
            raise ValueError('Request capture only supports its prepared ordinary decode shape')
        rows = metadata.requests.to(torch.int64)
        row_valid = ((rows >= 0) & (rows < self.token_positions.numel())
                     & (rows < metadata.live_counts[1]))
        safe_rows = rows.clamp(0, self.row_slot_ids.numel() - 1)
        slots = self.row_slot_ids[safe_rows].to(torch.int64)
        slot_valid = (slots >= 0) & (slots < self.physical_slots.numel())
        safe_slots = slots.clamp(0, self.physical_slots.numel() - 1)
        epochs = self.row_generations[safe_rows].to(torch.int64)
        live = metadata.positions >= 0
        valid = live & row_valid & slot_valid & (epochs > 0)
        valid = valid & (epochs == self.request_generations[safe_slots])
        matches = (self.physical_slots[:, None] == slots[None, :]) & valid[None, :]
        counts = matches.to(torch.int64).sum(dim=1)
        indices = matches.to(torch.int64).argmax(dim=1)
        unique = counts == 1
        self.token_rows.copy_(rows)
        self.token_slots.copy_(slots)
        self.token_positions.copy_(metadata.positions)
        self.token_ids.copy_(metadata.token_ids)
        self.token_generations.copy_(epochs)
        self.token_live.copy_(live)
        self.invalid_tokens.copy_(live & ~valid)
        self.slot_match_counts.copy_(counts)
        self.slot_last_index.copy_(indices)
        self.slot_live.copy_(unique)
        self.slot_positions.copy_(torch.where(unique, metadata.positions[indices], -1))
        self.slot_token_ids.copy_(torch.where(unique, metadata.token_ids[indices], -1))
        self.slot_generations.copy_(torch.where(unique, epochs[indices], 0))
        self.mapping_error.copy_(((live & ~valid).any() | (counts > 1).any()
                                 | (live.sum() != metadata.live_counts[0])
                                 | (metadata.live_counts[1] < 0)
                                 | (metadata.live_counts[1] > self.token_positions.numel())).reshape(1))
        # Clearing here also removes stale observations if a later layer aborts.
        self.residual_last.zero_()
        self.pre_mlp_last.zero_()
        self.mlp_input_last.zero_()

    def record_pre_mlp(self, value, layer):
        self.pre_mlp_last[layer].copy_(torch.where(self.slot_live[:, None], value[self.slot_last_index], 0))

    def record_mlp_input(self, value, layer):
        self.mlp_input_last[layer].copy_(torch.where(self.slot_live[:, None], value[self.slot_last_index], 0))

    def record_residual(self, value, layer):
        self.residual_last[layer].copy_(torch.where(self.slot_live[:, None], value[self.slot_last_index], 0))
