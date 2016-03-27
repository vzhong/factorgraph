factorgraph
====================

Factor graph and (loopy) believe propagation in Lua. For reference:

- [Factor Graphs and Algorithms](http://www.psi.toronto.edu/~psi/pubs2/1999%20and%20before/134.pdf) by Frey et al.
- [Factor Graphs and the Sum-Product Algorithm](http://vision.unipv.it/IA2/Factor%20graphs%20and%20the%20sum-product%20algorithm.pdf) by Kshischang et al.

# Dependencies

- [torchlib](https://github.com/vzhong/torchlib)

# Installation

You can install `factorgraph` as follows:

`git clone https://github.com/vzhong/factorgraph.git && cd torchlib && luarocks make`

# Usage

This package is namespaced. To use it:

```lua
local fg = require 'factorgraph'

local b = fg.Variable(2, 'B')
local a = fg.Variable(3, 'A')

local P_b = fg.Factor(torch.Tensor{0.3, 0.7}, {b}, 'P_B')

local P_a_given_b = fg.Factor(torch.Tensor{{0.2, 0.8}, {0.4, 0.6}, {0.1, 0.9}}, {a, b}, 'P_A_given_B')

local g = fg.Graph({a, b}, {P_a_given_b, P_b})

print('converged in '..g:forward{max_iter=10}..' iterations')
for node, p in pairs(g:marginals()) do
  print(p)
end
```
