#include <torch/extension.h>

#include <vector>

typedef uint64_t Grid;

int countSetBits(Grid n) {
  int count = 0;
  while (n) {
    count++;
    n ^= n & -n;
  }
  return count;
}

Grid pure_avail(Grid b, Grid w) {
  // const Grid sasks[] = {~0x0101010101010101, ~0x80808080808080FF, ~0x00000000000000FF,
  //                       ~0x01010101010101FF, ~0x8080808080808080, ~0xFF01010101010101,
  //                       ~0xFF00000000000000, ~0xFF80808080808080};
  // const int shifts[] = {1, 7, 8, 9, 1, 7, 8, 9};
  Grid avail = 0, alive;
  alive = b;
  while (alive) {
    alive = ((alive & ~0x0101010101010101) >> 1) & w;
    avail |= ((alive & ~0x0101010101010101) >> 1) & ~w & ~b;
  }
  alive = b;
  while (alive) {
    alive = ((alive & ~0x80808080808080FF) >> 7) & w;
    avail |= ((alive & ~0x80808080808080FF) >> 7) & ~w & ~b;
  }
  alive = b;
  while (alive) {
    alive = ((alive & ~0x00000000000000FF) >> 8) & w;
    avail |= ((alive & ~0x00000000000000FF) >> 8) & ~w & ~b;
  }
  alive = b;
  while (alive) {
    alive = ((alive & ~0x01010101010101FF) >> 9) & w;
    avail |= ((alive & ~0x01010101010101FF) >> 9) & ~w & ~b;
  }

  alive = b;
  while (alive) {
    alive = ((alive & ~0x8080808080808080) << 1) & w;
    avail |= ((alive & ~0x8080808080808080) << 1) & ~w & ~b;
  }
  alive = b;
  while (alive) {
    alive = ((alive & ~0xFF01010101010101) << 7) & w;
    avail |= ((alive & ~0xFF01010101010101) << 7) & ~w & ~b;
  }
  alive = b;
  while (alive) {
    alive = ((alive & ~0xFF00000000000000) << 8) & w;
    avail |= ((alive & ~0xFF00000000000000) << 8) & ~w & ~b;
  }
  alive = b;
  while (alive) {
    alive = ((alive & ~0xFF80808080808080) << 9) & w;
    avail |= ((alive & ~0xFF80808080808080) << 9) & ~w & ~b;
  }
  return avail;
}

Grid get_flipped(Grid b, Grid w, Grid s) {
  Grid f = s, alive, tmp;
  alive = s;
  tmp = 0;
  while (alive) {
    alive = ((alive & ~0x0101010101010101) >> 1) & w;
    tmp |= alive;
    if (((alive & ~0x0101010101010101) >> 1) & b) f |= tmp;
  }
  alive = s;
  tmp = 0;
  while (alive) {
    alive = ((alive & ~0x80808080808080FF) >> 7) & w;
    tmp |= alive;
    if (((alive & ~0x80808080808080FF) >> 7) & b) f |= tmp;
  }
  alive = s;
  tmp = 0;
  while (alive) {
    alive = ((alive & ~0x00000000000000FF) >> 8) & w;
    tmp |= alive;
    if (((alive & ~0x00000000000000FF) >> 8) & b) f |= tmp;
  }
  alive = s;
  tmp = 0;
  while (alive) {
    alive = ((alive & ~0x01010101010101FF) >> 9) & w;
    tmp |= alive;
    if (((alive & ~0x01010101010101FF) >> 9) & b) f |= tmp;
  }

  alive = s;
  tmp = 0;
  while (alive) {
    alive = ((alive & ~0x8080808080808080) << 1) & w;
    tmp |= alive;
    if (((alive & ~0x8080808080808080) << 1) & b) f |= tmp;
  }
  alive = s;
  tmp = 0;
  while (alive) {
    alive = ((alive & ~0xFF01010101010101) << 7) & w;
    tmp |= alive;
    if (((alive & ~0xFF01010101010101) << 7) & b) f |= tmp;
  }
  alive = s;
  tmp = 0;
  while (alive) {
    alive = ((alive & ~0xFF00000000000000) << 8) & w;
    tmp |= alive;
    if (((alive & ~0xFF00000000000000) << 8) & b) f |= tmp;
  }
  alive = s;
  tmp = 0;
  while (alive) {
    alive = ((alive & ~0xFF80808080808080) << 9) & w;
    tmp |= alive;
    if (((alive & ~0xFF80808080808080) << 9) & b) f |= tmp;
  }
  return f;
}

int (*eval)(Grid b, Grid w);

Grid rotate_cw(Grid b) {
  Grid cw = 0;
  for (size_t i = 0; i < 64; i++) {
    size_t x = i / 8, y = i % 8;
    size_t j = (7 - x) * 8 + (7 - y);
    cw |= ((b >> j) & 1) << i;
  }
  return cw;
}

Grid flip(Grid b) {
  Grid cw = 0;
  for (size_t i = 0; i < 64; i++) {
    size_t x = i / 8, y = i % 8;
    cw |= ((b >> (x * 8 + (7 - y))) & 1) << i;
  }
  return cw;
}

int eval_init(Grid b, Grid w) {
  Grid b_avail = pure_avail(b, w);
  Grid w_avail = pure_avail(w, b);
  return 32 * (countSetBits(b_avail) - countSetBits(w_avail)) + 1024 * (1 & (b >> 0)) +
         1024 * (1 & (b >> 7)) + 1024 * (1 & (b >> 63)) + 1024 * (1 & (b >> 56)) -
         1024 * (1 & (w >> 0)) - 1024 * (1 & (w >> 7)) - 1024 * (1 & (w >> 63)) -
         1024 * (1 & (w >> 56));
}

int eval_final(Grid b, Grid w) {
  Grid b_avail = pure_avail(b, w);
  Grid w_avail = pure_avail(w, b);
  if (!b_avail && !w_avail) return countSetBits(b) > countSetBits(w) ? 65536 : -65536;
  return 32 * (countSetBits(b_avail) - countSetBits(w_avail)) + countSetBits(b) - countSetBits(w) +
         1024 * (1 & (b >> 0)) + 1024 * (1 & (b >> 7)) + 1024 * (1 & (b >> 63)) +
         1024 * (1 & (b >> 56)) - 1024 * (1 & (w >> 0)) - 1024 * (1 & (w >> 7)) -
         1024 * (1 & (w >> 63)) - 1024 * (1 & (w >> 56));
}

int search(int depth, int alpha, int beta, Grid b, Grid w) {
  if (depth == 0) return eval(b, w);

  // Generate moves
  Grid b_avail = pure_avail(b, w);

  if (!b_avail) return -search(depth - 1, -beta, -alpha, w, b);

  while (b_avail) {
    Grid selected = b_avail & -b_avail;
    Grid flipped = get_flipped(b, w, selected);
    b_avail ^= selected;

    int val = -search(depth - 1, -beta, -alpha, w & ~flipped, b | flipped);

    if (val >= beta) return beta;
    if (val > alpha) alpha = val;
  }
  return alpha;
}

Grid std_ai(Grid b, Grid w) {
  int depth = 6;
  eval = countSetBits(w) + countSetBits(b) + depth < 50 ? eval_init : eval_final;

  // cerr << depth << endl;
  int alpha = -2100000000;
  int beta = 2100000000;
  Grid argmax = 0;

  // Generate moves
  Grid b_avail = pure_avail(b, w);
  while (b_avail) {
    Grid selected = b_avail & -b_avail;
    Grid flipped = get_flipped(b, w, selected);
    b_avail ^= selected;

    int val = -search(depth - 1, -beta, -alpha, w & ~flipped, b | flipped);

    if (val > alpha) {
      alpha = val;
      argmax = selected;
    }
  }
  return argmax;
}

#include <iostream>

torch::Tensor reversi_std(torch::Tensor data_b, torch::Tensor data_w) {
  auto result = torch::zeros_like(data_b);
  auto batch_size = data_b.size(0);

  auto b_it = data_b.contiguous().data_ptr<int64_t>();
  auto w_it = data_w.contiguous().data_ptr<int64_t>();
  auto result_it = result.contiguous().data_ptr<int64_t>();
  for (int i = 0; i < batch_size; i++) {
    Grid b = b_it[i];
    Grid w = w_it[i];
    Grid avail = pure_avail(b, w);
    if (avail) {
      Grid s = std_ai(b, w);
      Grid f = get_flipped(b, w, s);
      b |= f;
      w &= ~f;
    }
    if (!pure_avail(b, w) && !pure_avail(w, b)) {
      if (countSetBits(w) == countSetBits(b))
        result_it[i] = 1;
      else
        result_it[i] = 2 * (countSetBits(b) > countSetBits(w));
      b_it[i] = 0x0000000810000000;
      w_it[i] = 0x0000001008000000;
    } else {
      result_it[i] = 3;
      b_it[i] = (int64_t)(w);
      w_it[i] = (int64_t)(b);
    }
  }
  return result;
}

torch::Tensor reversi_win(torch::Tensor data_b, torch::Tensor data_w) {
  auto batch_size = data_b.size(0);
  auto result = torch::zeros_like(data_b);

  auto b_it = data_b.contiguous().data_ptr<int64_t>();
  auto w_it = data_w.contiguous().data_ptr<int64_t>();
  for (int i = 0; i < batch_size; i++) {
    Grid b = b_it[i];
    Grid w = w_it[i];
    if (!pure_avail(b, w) && !pure_avail(w, b)) {
      result[i] = countSetBits(b) > countSetBits(w);
    } else {
      result[i] = 3;
    }
  }
  return result;
}

#include <cmath>

std::vector<torch::Tensor> reversi_forward(torch::Tensor data_b, torch::Tensor data_w,
                                           torch::Tensor pred) {
  auto batch_size = data_b.size(0);
  auto selection = torch::zeros_like(data_b);
  auto result = torch::zeros_like(data_b);

  auto options = torch::TensorOptions().dtype(torch::kBool);
  auto valid_mask = torch::zeros_like(pred, options);

  auto b_it = data_b.contiguous().data_ptr<int64_t>();
  auto w_it = data_w.contiguous().data_ptr<int64_t>();
  auto result_it = result.contiguous().data_ptr<int64_t>();
  auto selection_it = selection.contiguous().data_ptr<int64_t>();
  auto valid_mask_it = valid_mask.contiguous().data_ptr<bool>();
  for (int i = 0; i < batch_size; i++) {
    auto pred_it = pred.select(0, i).contiguous().data_ptr<float>();
    Grid b = b_it[i];
    Grid w = w_it[i];
    Grid avail = pure_avail(b, w);
    if (!avail) {
      selection_it[i] = 64;
      b_it[i] = (int64_t)(w);
      w_it[i] = (int64_t)(b);
      valid_mask_it[i * 65 + 64] = 1;
    } else {
      float prob_sum = 0;
      for (int j = 0; j < 64; j++) {
        if ((avail >> j) & 1) {
          prob_sum += pred_it[j];
          valid_mask_it[i * 65 + j] = 1;
        }
      }

      float _rand = rand() * prob_sum / RAND_MAX;
      for (int j = 0; j < 64; j++) {
        if ((avail >> j) & 1) {
          _rand -= pred_it[j];
          selection_it[i] = j;
          if (_rand <= 0) break;
        }
      }
      Grid f = get_flipped(b, w, 1ULL << selection_it[i]);
      b_it[i] = (int64_t)(w & ~f);
      w_it[i] = (int64_t)(b | f);
    }
    b = b_it[i];
    w = w_it[i];
    if (!pure_avail(b, w) && !pure_avail(w, b)) {
      if (countSetBits(w) == countSetBits(b))
        result_it[i] = 1;
      else
        result_it[i] = 2 * (countSetBits(w) > countSetBits(b));
      b_it[i] = 0x0000000810000000;
      w_it[i] = 0x0000001008000000;
    } else {
      result_it[i] = 3;
      // if (rand() % 2) {
      //   b_it[i] = flip(b_it[i]);
      //   w_it[i] = flip(w_it[i]);
      // }
      // if (rand() % 2) {
      //   b_it[i] = rotate_cw(b_it[i]);
      //   w_it[i] = rotate_cw(w_it[i]);
      // }
    }
  }

  return {selection, result, valid_mask};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &reversi_forward, "reversi forward");
  m.def("std", &reversi_std, "reversi std");
  m.def("win", &reversi_win, "reversi win");
}
